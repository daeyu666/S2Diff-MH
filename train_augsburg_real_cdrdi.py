"""Self-supervised Augsburg Real-C and strict Augsburg-2 Wald-CDRDI training.

Wald targets are observed 30 m EnMAP-like HSI, with 90 m LR-HSI and real
Sentinel-2 aggregated to 30 m. EnMAP10 is never used as a Wald label.
Optional synthetic shifts augment the unsupervised measurement only; no flow
truth is used for regression or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Dict

import torch
import torch.nn.functional as F

from augsburg_real import build_augsburg_real_loaders
from cdrdi_geometry import deformation_regularizer, forward_warp, jacobian_determinant, spectral_project
from degradations.effective_gaussian import EffectiveGaussianDegradation
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver, image_gradients
from utils import CSVLogger, ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Augsburg-Real Real-C geometry adaptation")
    p.add_argument("--stage", choices=["train", "test"], default="train")
    p.add_argument("--cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--psf_json", default="./data/calibration/AugsburgReal_effective_psf.json")
    p.add_argument("--radiometry_json", default="./data/calibration/AugsburgReal_radiometry.json")
    p.add_argument("--init_checkpoint", default="", help="Legacy synthetic Stage-C warm start; forbidden for Wald")
    p.add_argument("--from_scratch", action="store_true", help="Required for strictly Wald-only CDRDI training")
    p.add_argument("--augment_shift_px", type=float, default=0.0,
                   help="Training-only random x/y shift of real MSI in 30 m Wald pixels; no flow labels")
    p.add_argument("--geometry_checkpoint", default="", help="Trained Augsburg-Real C checkpoint for --stage test")
    p.add_argument("--checkpoint_root", default="./checkpoints/augsburg_real")
    p.add_argument("--log_root", default="./logs/augsburg_real")
    p.add_argument("--save_name", default="AugsburgReal_C_recursive_realclosure")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)

    p.add_argument("--train_patch_size", type=int, default=96)
    p.add_argument("--train_stride", type=int, default=48)
    p.add_argument("--eval_patch_size", type=int, default=192)
    p.add_argument("--min_valid_fraction", type=float, default=0.80)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--train_steps", type=int, default=6)
    p.add_argument("--eval_steps", type=int, default=9)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_norm", type=float, default=1.0)
    p.add_argument("--lambda_grad", type=float, default=0.5)
    p.add_argument("--lambda_raw", type=float, default=0.1)
    p.add_argument("--lambda_def", type=float, default=1e-4)
    p.add_argument("--lambda_jac", type=float, default=1e-3)
    p.add_argument("--lambda_rigid", type=float, default=0.0,
                   help="Penalty on normalized rigid displacement and rotation")
    p.add_argument("--lambda_local", type=float, default=0.0,
                   help="Penalty on normalized local-field magnitude")
    p.add_argument("--selection_lambda_geometry", type=float, default=0.005,
                   help="Wald model-selection motion penalty coefficient")
    p.add_argument("--selection_min_jac", type=float, default=0.5,
                   help="Wald checkpoint rejection threshold for local Jacobian")
    p.add_argument("--selection_max_motion_fraction", type=float, default=0.9,
                   help="Wald checkpoint rejection bound as fraction of geometry cap")
    p.add_argument("--jac_margin", type=float, default=0.1)
    p.add_argument("--local_window", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--resume", default="")
    return p.parse_args()


def _load_radiometry(path: str, device):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    gain = torch.tensor(payload["gain"], dtype=torch.float32, device=device).view(1, 4, 1, 1)
    bias = torch.tensor(payload["bias"], dtype=torch.float32, device=device).view(1, 4, 1, 1)
    return gain, bias


def _load_sigma(path: str) -> float:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("scale_ratio", 0)) != 3:
        raise ValueError("Augsburg-Real PSF calibration must use scale_ratio=3")
    return float(payload["terminal_sigma_hr_pixels"])


def _checkpoint_extra(path: str) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload.get("extra", {}) if isinstance(payload, dict) else {}


def _wald_metadata(cache_root: str) -> bool:
    meta = []
    for split in ("train", "validation", "test"):
        path = os.path.join(cache_root, split, "meta.json")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        meta.append(data)
    is_wald = meta[0].get("msi_source") == "real_Sentinel_2_Wald_30m"
    if is_wald:
        for split, entry in zip(("train", "validation", "test"), meta):
            if (entry.get("msi_source") != "real_Sentinel_2_Wald_30m"
                or entry.get("target") != "30m_EnMAP_like"
                or entry.get("gt_source") != "observed_30m_HSI_only"
                or int(entry.get("scale_ratio", -1)) != 3):
                raise ValueError(f"Invalid Wald-only provenance in {split}/meta.json")
    return is_wald


def _wald_checkpoint_guard(path: str, *, stage: str) -> dict:
    extra = _checkpoint_extra(path)
    if (extra.get("stage") != "Augsburg2-Wald-CDRDI"
        or extra.get("msi_source") != "real_Sentinel_2_Wald_30m"
        or extra.get("geometry_pixel_size_m") != 30):
        raise ValueError(
            f"{stage} requires a strictly Wald-trained CDRDI checkpoint, not legacy Real-C"
        )
    return extra


def _assert_wald_checkpoint_settings(extra: dict, args, *, sigma: float) -> None:
    config = extra.get("geometry_config", {})
    expected = {
        "base_channels": args.base_channels,
        "control_grid": args.control_grid,
        "max_translation": args.max_translation,
        "max_rotation_deg": args.max_rotation_deg,
        "max_local_px": args.max_local_px,
    }
    for key, value in expected.items():
        observed = config.get(key)
        if observed is None or abs(float(observed) - float(value)) > 1e-7:
            raise ValueError(
                f"Wald-CDRDI checkpoint {key}={observed} does not match requested {value}"
            )
    if abs(float(extra.get("effective_sigma", -1)) - float(sigma)) > 1e-7:
        raise ValueError("Wald-CDRDI checkpoint PSF does not match Wald PSF file")
    if os.path.normpath(extra.get("radiometry_json", "")) != os.path.normpath(args.radiometry_json):
        raise ValueError("Wald-CDRDI checkpoint radiometry file does not match training")


def _selection_score(metrics: dict, args) -> float:
    norm = lambda x, cap: x / max(float(cap), 1e-6)
    motion = (
        norm(metrics["dx_abs"], args.max_translation)
        + norm(metrics["dy_abs"], args.max_translation)
        + norm(metrics["theta_abs"], args.max_rotation_deg)
        + norm(metrics["local_mean"], args.max_local_px)
    )
    return float(metrics["norm"] + args.selection_lambda_geometry * motion)


def _wald_geometry_accepted(metrics: dict, args) -> bool:
    bound = args.selection_max_motion_fraction
    return (
        metrics["min_jac"] >= args.selection_min_jac
        and metrics["dx_abs"] <= bound * args.max_translation
        and metrics["dy_abs"] <= bound * args.max_translation
        and metrics["theta_abs"] <= bound * args.max_rotation_deg
        and metrics["local_mean"] <= bound * args.max_local_px
    )


def _lr_mask(mask_hr: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(mask_hr.float(), 3, 3) >= 0.999


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    selected = value[mask]
    if selected.numel() == 0:
        return value.new_zeros(())
    return selected.mean()


def _charbonnier(residual: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    value = torch.sqrt(residual * residual + eps * eps) - eps
    return _masked_mean(value, mask)


def _local_standardize(x: torch.Tensor, window: int, eps: float = 1e-4) -> torch.Tensor:
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("local_window must be a positive odd integer")
    pad = window // 2
    mu = F.avg_pool2d(x, window, stride=1, padding=pad)
    second = F.avg_pool2d(x * x, window, stride=1, padding=pad)
    var = (second - mu * mu).clamp_min(0.0)
    return (x - mu) / torch.sqrt(var + eps * eps)


def real_geometry_loss(outputs: Dict[str, object], target: torch.Tensor, mask: torch.Tensor, args):
    prediction = outputs["final_prediction"]
    target_n = _local_standardize(target, args.local_window)
    pred_n = _local_standardize(prediction, args.local_window)
    norm_loss = _charbonnier(target_n - pred_n, mask)

    tgx, tgy = image_gradients(target_n)
    pgx, pgy = image_gradients(pred_n)
    grad_loss = 0.5 * (
        _charbonnier(tgx - pgx, mask) + _charbonnier(tgy - pgy, mask)
    )
    raw_loss = _charbonnier(target - prediction, mask)
    local = outputs["final_local_field"]
    reg = deformation_regularizer(local)
    jac = jacobian_determinant(local)
    jac_penalty = torch.relu(float(args.jac_margin) - jac).mean()
    rigid = outputs["final_rigid"]
    normalized_rigid = torch.stack(
        (rigid[:, 0] / max(args.max_translation, 1e-6),
         rigid[:, 1] / max(args.max_translation, 1e-6),
         rigid[:, 2] / max(args.max_rotation_deg, 1e-6)), dim=1,
    )
    rigid_penalty = normalized_rigid.square().mean()
    local_penalty = (local / max(args.max_local_px, 1e-6)).square().mean()
    total = (
        args.lambda_norm * norm_loss
        + args.lambda_grad * grad_loss
        + args.lambda_raw * raw_loss
        + args.lambda_def * reg
        + args.lambda_jac * jac_penalty
        + args.lambda_rigid * rigid_penalty
        + args.lambda_local * local_penalty
    )
    return total, {
        "norm": norm_loss,
        "grad": grad_loss,
        "raw": raw_loss,
        "reg": reg,
        "jac_penalty": jac_penalty,
        "min_jac": jac.amin(),
        "rigid_penalty": rigid_penalty,
        "local_penalty": local_penalty,
    }


def _estimate_batch(model, batch, *, p0, srf, radiometry, steps: int, device, augment_shift_px=0.0):
    lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
    hr_msi = batch["hr_msi"].to(device, non_blocking=True)
    if radiometry is not None:
        gain, bias = radiometry
        hr_msi = gain * hr_msi + bias
    mask_hr = batch["valid_mask"].to(device, non_blocking=True)
    target = spectral_project(lr_hsi, srf)
    if augment_shift_px > 0:
        batch_size, _, height, width = hr_msi.shape
        delta = (torch.rand(batch_size, 2, device=device) * 2.0 - 1.0) * augment_shift_px
        angles = torch.zeros(batch_size, device=device, dtype=hr_msi.dtype)
        local = torch.zeros(batch_size, 2, height, width, device=device, dtype=hr_msi.dtype)
        hr_msi = forward_warp(hr_msi, delta[:, 0], delta[:, 1], angles, local)
    mask = _lr_mask(mask_hr)
    outputs = model(target, hr_msi, p0, steps=steps)
    return target, mask, outputs


def train_one_epoch(model, loader, optimizer, *, p0, srf, radiometry, args, device):
    model.train()
    sums = {k: 0.0 for k in ("loss", "norm", "grad", "raw", "reg", "jac", "rigid", "local")}
    count = 0
    for batch in loader:
        target, mask, outputs = _estimate_batch(
            model,
            batch,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            steps=args.train_steps,
            device=device,
            augment_shift_px=args.augment_shift_px,
        )
        loss, parts = real_geometry_loss(outputs, target, mask, args)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Augsburg-Real CDRDI loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        n = int(target.shape[0])
        sums["loss"] += float(loss.detach().item()) * n
        sums["norm"] += float(parts["norm"].detach().item()) * n
        sums["grad"] += float(parts["grad"].detach().item()) * n
        sums["raw"] += float(parts["raw"].detach().item()) * n
        sums["reg"] += float(parts["reg"].detach().item()) * n
        sums["jac"] += float(parts["min_jac"].detach().item()) * n
        sums["rigid"] += float(parts["rigid_penalty"].detach().item()) * n
        sums["local"] += float(parts["local_penalty"].detach().item()) * n
        count += n
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, *, p0, srf, radiometry, args, device):
    model.eval()
    keys = (
        "initial_norm", "norm", "grad", "raw",
        "dx_abs", "dy_abs", "theta_abs", "local_mean",
    )
    sums = {key: 0.0 for key in keys}
    total_weight = 0.0
    min_jac = float("inf")
    for batch in loader:
        target, mask, outputs = _estimate_batch(
            model,
            batch,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            steps=args.eval_steps,
            device=device,
        )
        _, parts = real_geometry_loss(outputs, target, mask, args)
        initial = outputs["initial_prediction"]
        initial_n = _local_standardize(initial, args.local_window)
        target_n = _local_standardize(target, args.local_window)
        initial_norm = float(_charbonnier(target_n - initial_n, mask).item())
        rigid = outputs["final_rigid"]
        local = outputs["final_local_field"]
        weight = float(mask[:, 0].sum().item())
        if weight <= 0.0:
            continue
        values = {
            "initial_norm": initial_norm,
            "norm": float(parts["norm"].item()),
            "grad": float(parts["grad"].item()),
            "raw": float(parts["raw"].item()),
            "dx_abs": float(rigid[:, 0].abs().mean().item()),
            "dy_abs": float(rigid[:, 1].abs().mean().item()),
            "theta_abs": float(rigid[:, 2].abs().mean().item()),
            "local_mean": float(
                torch.linalg.vector_norm(local, dim=1).mean().item()
            ),
        }
        for key, value in values.items():
            sums[key] += value * weight
        total_weight += weight
        min_jac = min(min_jac, float(parts["min_jac"].item()))
    if total_weight <= 0.0:
        raise ValueError("empty/invalid validation loader")
    out = {key: sums[key] / total_weight for key in keys}
    out["min_jac"] = min_jac
    out["closure_reduction"] = (
        1.0 - out["norm"] / max(out["initial_norm"], 1e-12)
    )
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    is_wald = _wald_metadata(args.cache_root)
    if is_wald:
        if not 0.0 <= args.augment_shift_px <= 0.75:
            raise ValueError("Wald augment_shift_px must be in [0,0.75] 30m MSI pixels")
        if args.psf_json == "./data/calibration/AugsburgReal_effective_psf.json":
            args.psf_json = os.path.join(args.cache_root, "wald_psf.json")
        if args.radiometry_json == "./data/calibration/AugsburgReal_radiometry.json":
            args.radiometry_json = "./data/calibration/Augsburg2_Wald_radiometry.json"
        if (args.max_translation > 1.5 or args.max_rotation_deg > 1.0
            or args.max_local_px > 1.0):
            raise ValueError(
                "Wald-CDRDI requires conservative geometry caps; use "
                "--max_translation 1 --max_rotation_deg 0.5 --max_local_px 0.5"
            )
        if args.init_checkpoint:
            raise ValueError("Strict Wald-CDRDI forbids legacy/EnMAP10 pretrained initialization")
        if args.stage == "train" and not (args.from_scratch or args.resume):
            raise ValueError("Strict Wald-CDRDI requires --from_scratch or Wald --resume")
        if args.lambda_rigid <= 0 or args.lambda_local <= 0:
            raise ValueError("Wald-CDRDI requires positive --lambda_rigid and --lambda_local")
        if args.selection_min_jac < 0.5:
            raise ValueError("Wald-CDRDI selection_min_jac must be >= 0.5")
        if args.resume:
            _wald_checkpoint_guard(args.resume, stage="resume")
    elif args.from_scratch and args.init_checkpoint:
        raise ValueError("--from_scratch and --init_checkpoint are mutually exclusive")
    if args.stage == "test" and is_wald and not args.geometry_checkpoint:
        raise ValueError("--stage test requires a Wald --geometry_checkpoint")
    sigma = _load_sigma(args.psf_json)
    if is_wald and (args.resume or args.stage == "test"):
        check_path = args.resume if args.resume else args.geometry_checkpoint
        extra = _wald_checkpoint_guard(check_path, stage="resume" if args.resume else "test")
        _assert_wald_checkpoint_settings(extra, args, sigma=sigma)
    radiometry = _load_radiometry(args.radiometry_json, device)

    train_loader, val_loader, test_loader, info = build_augsburg_real_loaders(
        args.cache_root,
        train_patch_size=args.train_patch_size,
        train_stride=args.train_stride,
        eval_patch_size=args.eval_patch_size,
        min_valid_fraction=args.min_valid_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=torch.float32)
    p0 = EffectiveGaussianDegradation(
        scale_ratio=3,
        terminal_sigma=sigma,
        truncate=3.0,
    ).to(device)
    model = LearnedPhysicalResidualSolver(
        4,
        base_channels=args.base_channels,
        control_grid=args.control_grid,
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px,
    ).to(device)
    if args.stage == "test":
        if not args.geometry_checkpoint:
            raise ValueError("--stage test requires --geometry_checkpoint")
        load_checkpoint(
            model,
            args.geometry_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        metrics = evaluate(
            model,
            test_loader,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            args=args,
            device=device,
        )
        print(
            "FINAL_REAL_C "
            f"INIT_NORM={metrics['initial_norm']:.8f} "
            f"FINAL_NORM={metrics['norm']:.8f} "
            f"REDUCTION={100*metrics['closure_reduction']:.3f}% "
            f"RAW={metrics['raw']:.8f} MIN_JAC={metrics['min_jac']:.6f} "
            f"DX={metrics['dx_abs']:.4f} DY={metrics['dy_abs']:.4f} "
            f"THETA={metrics['theta_abs']:.4f} LOCAL={metrics['local_mean']:.4f}"
        )
        return

    if not args.init_checkpoint and not args.resume and not args.from_scratch:
        raise ValueError("Real-C training requires --init_checkpoint, --from_scratch or --resume")
    if args.init_checkpoint and not args.resume:
        load_checkpoint(
            model,
            args.init_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ensure_dir(args.checkpoint_root)
    ensure_dir(args.log_root)
    checkpoint = os.path.join(args.checkpoint_root, args.save_name + ".pth")
    logger = CSVLogger(
        os.path.join(args.log_root, args.save_name + ".csv"),
        [
            "epoch", "train_loss", "train_norm", "train_grad", "train_raw",
            "val_initial_norm", "val_norm", "val_grad", "val_raw",
            "val_closure_reduction", "val_min_jac", "dx_abs", "dy_abs",
            "theta_abs", "local_mean", "val_selection_score",
            "selection_accepted", "best_val_score",
        ],
    )

    start_epoch = 0
    best = float("inf")
    if args.resume:
        start_epoch, stored = load_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
        if stored > 0:
            best = stored

    print(
        f"AUGSBURG_C_PROTOCOL={'Wald_30m' if is_wald else 'legacy_RealC'} "
        f"calibration={args.radiometry_json} train_patch={args.train_patch_size} "
        f"train_stride={args.train_stride} eval_patch={args.eval_patch_size} "
        f"geometry_caps=({args.max_translation},{args.max_rotation_deg},{args.max_local_px}) "
        f"augmentation={args.augment_shift_px} MSI_30m_pixels "
    )
    print(
        "AUGSBURG_REAL_C supervision=observable_physical_closure_only "
        f"sigma={sigma:.6f} scale=3 stages=1,2,3 train_steps={args.train_steps} "
        f"eval_steps={args.eval_steps}"
    )
    print(
        f"RADIOMETRY train_only_affine={radiometry is not None} file={args.radiometry_json}"
    )
    print(
        "LOSS geometry=shared_rigid_plus_bspline local_normalized_closure=True "
        "gradient_closure=True raw_reflectance_closure=True flow_GT=False"
    )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            args=args,
            device=device,
        )
        print(
            f"epoch={epoch:03d}/{args.epochs} loss={tr['loss']:.8f} "
            f"norm={tr['norm']:.8f} grad={tr['grad']:.8f} raw={tr['raw']:.8f}"
        )
        if epoch % args.eval_interval != 0 and epoch != args.epochs:
            continue

        va = evaluate(
            model,
            val_loader,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            args=args,
            device=device,
        )
        print(
            "REAL_C_VAL "
            f"INIT_NORM={va['initial_norm']:.8f} FINAL_NORM={va['norm']:.8f} "
            f"REDUCTION={100*va['closure_reduction']:.3f}% RAW={va['raw']:.8f} "
            f"MIN_JAC={va['min_jac']:.6f} DX={va['dx_abs']:.4f} "
            f"DY={va['dy_abs']:.4f} THETA={va['theta_abs']:.4f} "
            f"LOCAL={va['local_mean']:.4f}"
        )
        score = _selection_score(va, args) if is_wald else va["norm"]
        accepted = _wald_geometry_accepted(va, args) if is_wald else True
        print(
            f"REAL_C_SELECTION score={score:.8f} accepted={int(accepted)} "
            f"min_jac={va['min_jac']:.5f} (closure alone is not acceptance)"
        )
        logger.write(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_norm": tr["norm"],
                "train_grad": tr["grad"],
                "train_raw": tr["raw"],
                "val_initial_norm": va["initial_norm"],
                "val_norm": va["norm"],
                "val_grad": va["grad"],
                "val_raw": va["raw"],
                "val_closure_reduction": va["closure_reduction"],
                "val_min_jac": va["min_jac"],
                "dx_abs": va["dx_abs"],
                "dy_abs": va["dy_abs"],
                "theta_abs": va["theta_abs"],
                "local_mean": va["local_mean"],
                "val_selection_score": score,
                "selection_accepted": int(accepted),
                "best_val_score": min(best, score if accepted else float("inf")),
            }
        )
        if accepted and score < best:
            best = score
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best,
                checkpoint,
                extra={
                    "stage": "Augsburg2-Wald-CDRDI" if is_wald else "AugsburgReal-C",
                    "msi_source": "real_Sentinel_2_Wald_30m" if is_wald else "real_Sentinel_2",
                    "geometry_pixel_size_m": 30 if is_wald else 10,
                    "radiometry_json": args.radiometry_json,
                    "geometry_config": {
                        "base_channels": args.base_channels,
                        "control_grid": args.control_grid,
                        "max_translation": args.max_translation,
                        "max_rotation_deg": args.max_rotation_deg,
                        "max_local_px": args.max_local_px,
                    },
                    "observable_monitor": "local_normalized_closure",
                    "effective_sigma": sigma,
                    "scale_ratio": 3,
                    "stages": [1, 2, 3],
                    "train_steps": args.train_steps,
                    "eval_steps": args.eval_steps,
                    "flow_gt_used": False,
                },
            )
            print(f"SAVED_BEST {checkpoint} val_guarded_score={best:.8f}")


if __name__ == "__main__":
    main()
