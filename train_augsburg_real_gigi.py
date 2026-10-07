"""Real-E: terminal GIGI adaptation for Augsburg-Real.

Real-C geometry and Real-D2 diffusion are frozen.  The output remains in the
real Sentinel-2 10 m reference frame; geometry-compensated EnMAP10 metrics are
computed only after forward mapping the prediction to the HSI reference frame.
Observed LR-HSI is never inverse warped.
"""

from __future__ import annotations

import argparse
import math
import os
from statistics import mean

import torch
import torch.nn.functional as F

from augsburg_real import build_augsburg_real_loaders
from augsburg_real_process import build_augsburg_real_process
from cdrdi_geometry import forward_warp, spectral_project
from config import TrainConfig
from innovation1 import reconstruct_from_terminal_lr
from main import build_model
from models import (
    HeterogeneityGuidedSpectralRefiner,
    TerminalGIGISpectralRefiner,
    ranked_msi_heterogeneity,
)
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from train_augsburg_real_diffusion import (
    _apply_radiometry,
    _estimated_process,
    _estimate_geometry,
    _load_json,
    _lr_mask,
    _mask_to_msi,
    _masked_l1,
    _masked_metrics,
    _masked_sam,
    _radiometry,
    _warp_adjoint_normalized,
)
from utils import CSVLogger, ensure_dir, get_device, load_checkpoint, set_seed


VARIANTS = ("conv", "gigi", "gigi_hetero", "gigi_phy", "full")


def parse_args():
    p = argparse.ArgumentParser(description="Augsburg-Real Real-E GIGI adaptation")
    p.add_argument("--stage", choices=["train", "test"], default="train")
    p.add_argument("--variant", choices=VARIANTS, default="full")
    p.add_argument("--cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--psf_json", default="./data/calibration/AugsburgReal_effective_psf.json")
    p.add_argument("--radiometry_json", default="./data/calibration/AugsburgReal_radiometry.json")
    p.add_argument("--geometry_checkpoint", required=True)
    p.add_argument("--diffusion_checkpoint", required=True)
    p.add_argument("--init_refiner_checkpoint", default="", help="Synthetic Augsburg Stage-E checkpoint")
    p.add_argument("--refiner_checkpoint", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--checkpoint_root", default="./checkpoints/augsburg_real")
    p.add_argument("--log_root", default="./logs/augsburg_real")
    p.add_argument("--save_name", default="AugsburgReal_E_gigi_full")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)

    p.add_argument("--train_patch_size", type=int, default=96)
    p.add_argument("--train_stride", type=int, default=48)
    p.add_argument("--eval_patch_size", type=int, default=192)
    p.add_argument("--min_valid_fraction", type=float, default=0.80)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--geometry_steps", type=int, default=9)
    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)

    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--diffusion_base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--refine_hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--disable_tangent", action="store_true")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_ang", type=float, default=0.1)
    p.add_argument("--lambda_ref", type=float, default=0.5)
    p.add_argument("--lambda_phy", type=float, default=0.1)
    p.add_argument("--lambda_msi", type=float, default=0.1)
    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument("--monitor", choices=["ref_sam_high", "ref_sam", "ref_psnr"], default="ref_sam_high")
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=10)
    return p.parse_args()


def _diffusion_config(args):
    return TrainConfig(
        dataset="Augsburg",
        scale_ratio=3,
        diffusion_steps=args.diffusion_steps,
        predictor="raw_direct",
        base_channels=args.diffusion_base_channels,
        time_dim=args.time_dim,
        spectral_hidden=args.spectral_hidden,
        dropout=args.dropout,
        device=args.device,
    )


def _load_refiner_weights(model, path: str, device):
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    variant = state.get("variant", model.variant)
    if variant != model.variant:
        raise ValueError(f"refiner variant mismatch: checkpoint={variant} requested={model.variant}")
    payload = state.get("model")
    if payload is None:
        raise KeyError(f"{path} does not contain key 'model'")
    model.load_state_dict(payload, strict=True)
    print(f"REFINER_INIT {path}")


def _save_refiner(model, optimizer, epoch, best, path, args):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best),
            "monitor": args.monitor,
            "variant": model.variant,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "extra": vars(args),
        },
        path,
    )


def _load_training_refiner(model, path, device, optimizer=None):
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if state.get("variant", model.variant) != model.variant:
        raise ValueError("checkpoint/refiner variant mismatch")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0)), float(state.get("best_metric", float("inf")))


def _build_projector(info, args, device):
    model = HeterogeneityGuidedSpectralRefiner(
        n_bands=info["n_bands"],
        total_steps=args.diffusion_steps,
        hidden_channels=1,
        variant="full",
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _physical_residual(process, projector, y_h, base):
    predicted = process.terminal_observation(base)
    native = y_h - predicted
    lifted = process.terminal_state(native, target_size=tuple(base.shape[-2:]))
    broad = projector.project_broadshape(lifted)
    return projector.project_tangent(broad, base, preserve_broadshape=True)


def _hsi_to_msi(x, srf):
    return spectral_project(x, srf)


def _heterogeneity_to_reference(heterogeneity, rigid, local):
    return forward_warp(
        heterogeneity[:, None].float(),
        rigid[:, 0],
        rigid[:, 1],
        rigid[:, 2],
        local,
    )[:, 0]


def _pixel_sam_deg(pred, target, eps=1e-12):
    p = pred.float()
    t = target.float()
    dot = (p * t).sum(dim=1)
    pn = torch.linalg.vector_norm(p, dim=1)
    tn = torch.linalg.vector_norm(t, dim=1)
    valid = (pn > eps) & (tn > eps)
    angle = torch.full_like(pn, float("nan"))
    if valid.any():
        cos = (dot[valid] / (pn[valid] * tn[valid]).clamp_min(eps)).clamp(-1.0, 1.0)
        angle[valid] = torch.acos(cos) * (180.0 / math.pi)
    return angle


def _masked_region_sam(pred, target, mask, heterogeneity, fraction):
    angle = _pixel_sam_deg(pred, target)
    valid = (mask[:, 0] > 0.5) & torch.isfinite(angle) & torch.isfinite(heterogeneity)
    if not valid.any():
        return float("nan"), float("nan")
    h = heterogeneity[valid]
    a = angle[valid]
    lo = torch.quantile(h, float(fraction))
    hi = torch.quantile(h, 1.0 - float(fraction))
    high = a[h >= hi]
    low = a[h <= lo]
    return (
        float(high.mean().item()) if high.numel() else float("nan"),
        float(low.mean().item()) if low.numel() else float("nan"),
    )


def _update_mechanism(details, mask_msi, fraction):
    update = details["update"]
    heterogeneity = details["heterogeneity"]
    energy = torch.linalg.vector_norm(update.float(), dim=1)
    valid = mask_msi[:, 0] > 0.5
    h = heterogeneity[valid]
    e = energy[valid]
    if h.numel() == 0:
        return {
            "update_high": float("nan"),
            "update_low": float("nan"),
            "update_ratio": float("nan"),
        }
    lo = torch.quantile(h, float(fraction))
    hi = torch.quantile(h, 1.0 - float(fraction))
    high = e[h >= hi]
    low = e[h <= lo]
    high_mean = float(high.mean().item()) if high.numel() else float("nan")
    low_mean = float(low.mean().item()) if low.numel() else float("nan")
    ratio = (
        high_mean / max(low_mean, 1e-12)
        if math.isfinite(high_mean) and math.isfinite(low_mean)
        else float("nan")
    )
    return {
        "update_high": high_mean,
        "update_low": low_mean,
        "update_ratio": ratio,
    }


@torch.no_grad()
def _base_bundle(
    diffusion,
    geometry,
    projector,
    batch,
    *,
    base_process,
    p0,
    srf,
    radiometry,
    args,
    device,
):
    gt_ref = batch["gt"].to(device, non_blocking=True)
    y_h = batch["lr_hsi"].to(device, non_blocking=True)
    y_m = _apply_radiometry(
        batch["hr_msi"].to(device, non_blocking=True), radiometry
    )
    mask_ref = batch["valid_mask"].to(device, non_blocking=True) > 0.5
    rigid, local = _estimate_geometry(
        geometry,
        y_h,
        y_m,
        p0=p0,
        srf=srf,
        steps=args.geometry_steps,
    )
    process = _estimated_process(base_process, rigid, local)
    base = reconstruct_from_terminal_lr(
        diffusion,
        process,
        y_h,
        target_size=tuple(gt_ref.shape[-2:]),
        hr_msi=y_m,
    )
    r_phy = _physical_residual(process, projector, y_h, base)
    heterogeneity = ranked_msi_heterogeneity(y_m)
    gt_msi = _warp_adjoint_normalized(gt_ref, rigid, local)
    mask_msi = _mask_to_msi(mask_ref, rigid, local)
    return {
        "gt_ref": gt_ref,
        "gt_msi": gt_msi,
        "y_h": y_h,
        "y_m": y_m,
        "mask_ref": mask_ref,
        "mask_msi": mask_msi,
        "rigid": rigid,
        "local": local,
        "process": process,
        "base": base,
        "r_phy": r_phy,
        "heterogeneity": heterogeneity,
    }


def train_one_epoch(
    refiner,
    diffusion,
    geometry,
    projector,
    loader,
    optimizer,
    *,
    base_process,
    p0,
    srf,
    radiometry,
    args,
    device,
):
    refiner.train()
    diffusion.eval()
    geometry.eval()
    projector.eval()
    sums = {k: 0.0 for k in ("loss", "l1", "ang", "ref", "phy", "msi")}
    count = 0
    for batch in loader:
        with torch.no_grad():
            pack = _base_bundle(
                diffusion,
                geometry,
                projector,
                batch,
                base_process=base_process,
                p0=p0,
                srf=srf,
                radiometry=radiometry,
                args=args,
                device=device,
            )
        refined = refiner(
            pack["base"],
            pack["y_m"],
            pack["r_phy"],
            pack["heterogeneity"],
        )
        l1 = _masked_l1(refined, pack["gt_msi"], pack["mask_msi"])
        ang = _masked_sam(refined, pack["gt_msi"], pack["mask_msi"])
        ref_pred = forward_warp(
            refined,
            pack["rigid"][:, 0],
            pack["rigid"][:, 1],
            pack["rigid"][:, 2],
            pack["local"],
        )
        ref_loss = _masked_l1(
            ref_pred, pack["gt_ref"], pack["mask_ref"]
        )
        phy = _masked_l1(
            pack["process"].terminal_observation(refined),
            pack["y_h"],
            _lr_mask(pack["mask_ref"]),
        )
        msi = _masked_l1(
            _hsi_to_msi(refined, srf),
            pack["y_m"],
            pack["mask_msi"],
        )
        loss = (
            args.lambda_l1 * l1
            + args.lambda_ang * ang
            + args.lambda_ref * ref_loss
            + args.lambda_phy * phy
            + args.lambda_msi * msi
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Augsburg-Real GIGI loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            refiner.parameters(),
            args.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()

        n = pack["gt_ref"].shape[0]
        for key, value in (
            ("loss", loss),
            ("l1", l1),
            ("ang", ang),
            ("ref", ref_loss),
            ("phy", phy),
            ("msi", msi),
        ):
            sums[key] += float(value.detach().item()) * n
        count += n
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(
    refiner,
    diffusion,
    geometry,
    projector,
    loader,
    *,
    base_process,
    p0,
    srf,
    radiometry,
    args,
    device,
):
    refiner.eval()
    diffusion.eval()
    geometry.eval()
    projector.eval()
    rows = []
    for batch in loader:
        pack = _base_bundle(
            diffusion,
            geometry,
            projector,
            batch,
            base_process=base_process,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            args=args,
            device=device,
        )
        refined, details = refiner(
            pack["base"],
            pack["y_m"],
            pack["r_phy"],
            pack["heterogeneity"],
            return_details=True,
        )
        base_ref = forward_warp(
            pack["base"],
            pack["rigid"][:, 0],
            pack["rigid"][:, 1],
            pack["rigid"][:, 2],
            pack["local"],
        )
        refined_ref = forward_warp(
            refined,
            pack["rigid"][:, 0],
            pack["rigid"][:, 1],
            pack["rigid"][:, 2],
            pack["local"],
        )
        base_psnr, base_sam = _masked_metrics(
            base_ref, pack["gt_ref"], pack["mask_ref"]
        )
        ref_psnr, ref_sam = _masked_metrics(
            refined_ref, pack["gt_ref"], pack["mask_ref"]
        )
        hetero_ref = _heterogeneity_to_reference(
            pack["heterogeneity"], pack["rigid"], pack["local"]
        )
        base_high, base_low = _masked_region_sam(
            base_ref,
            pack["gt_ref"],
            pack["mask_ref"],
            hetero_ref,
            args.region_fraction,
        )
        high, low = _masked_region_sam(
            refined_ref,
            pack["gt_ref"],
            pack["mask_ref"],
            hetero_ref,
            args.region_fraction,
        )
        phy = float(
            _masked_l1(
                pack["process"].terminal_observation(refined),
                pack["y_h"],
                _lr_mask(pack["mask_ref"]),
            ).item()
        )
        msi = float(
            _masked_l1(
                _hsi_to_msi(refined, srf),
                pack["y_m"],
                pack["mask_msi"],
            ).item()
        )
        mech = _update_mechanism(
            details, pack["mask_msi"], args.region_fraction
        )
        rows.append(
            {
                "base_ref_psnr": base_psnr,
                "base_ref_sam": base_sam,
                "base_ref_sam_high": base_high,
                "base_ref_sam_low": base_low,
                "ref_psnr": ref_psnr,
                "ref_sam": ref_sam,
                "ref_sam_high": high,
                "ref_sam_low": low,
                "phy": phy,
                "msi": msi,
                **mech,
            }
        )
    if not rows:
        raise ValueError("empty evaluation loader")
    return {k: mean(row[k] for row in rows) for k in rows[0]}


def _build_frozen(args, info, device, base_process):
    srf = torch.as_tensor(
        info["srf_weights"], dtype=torch.float32, device=device
    )
    p0 = base_process.operator.to(device)
    geometry = LearnedPhysicalResidualSolver(
        4,
        base_channels=args.geometry_base_channels,
        control_grid=args.control_grid,
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px,
    ).to(device)
    load_checkpoint(
        geometry,
        args.geometry_checkpoint,
        map_location=str(device),
        load_optimizer=False,
    )
    geometry.eval()
    for parameter in geometry.parameters():
        parameter.requires_grad_(False)

    diffusion = build_model(_diffusion_config(args), info, device)
    load_checkpoint(
        diffusion,
        args.diffusion_checkpoint,
        map_location=str(device),
        load_optimizer=False,
    )
    diffusion.eval()
    for parameter in diffusion.parameters():
        parameter.requires_grad_(False)

    projector = _build_projector(info, args, device)
    return geometry, diffusion, projector, p0, srf


def _build_refiner(args, info, device):
    return TerminalGIGISpectralRefiner(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_msi_bands"],
        hidden_channels=args.refine_hidden,
        heads=args.heads,
        variant=args.variant,
        tangent_output=not args.disable_tangent,
    ).to(device)


def _best_initial(monitor):
    return float("-inf") if monitor == "ref_psnr" else float("inf")


def _better(value, best, monitor):
    return value > best if monitor == "ref_psnr" else value < best


def _run_eval_print(metrics, prefix):
    print(
        f"{prefix} BASE_REF_PSNR={metrics['base_ref_psnr']:.6f} "
        f"BASE_REF_SAM={metrics['base_ref_sam']:.6f} "
        f"BASE_REF_SAM_HIGH={metrics['base_ref_sam_high']:.6f} "
        f"REF_PSNR={metrics['ref_psnr']:.6f} "
        f"REF_SAM={metrics['ref_sam']:.6f} "
        f"REF_SAM_HIGH={metrics['ref_sam_high']:.6f} "
        f"REF_SAM_LOW={metrics['ref_sam_low']:.6f} "
        f"dPSNR={metrics['ref_psnr']-metrics['base_ref_psnr']:+.6f} "
        f"dSAM={metrics['ref_sam']-metrics['base_ref_sam']:+.6f} "
        f"dSAM_HIGH={metrics['ref_sam_high']-metrics['base_ref_sam_high']:+.6f} "
        f"PHY_L1={metrics['phy']:.8f} MSI_L1={metrics['msi']:.8f} "
        f"UPDATE_HL_RATIO={metrics['update_ratio']:.6f}"
    )


def train(args):
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    if not args.init_refiner_checkpoint and not args.resume:
        raise ValueError(
            "Real-E training requires --init_refiner_checkpoint or --resume"
        )
    set_seed(args.seed)
    device = get_device(args.device)
    sigma = float(
        _load_json(args.psf_json)["terminal_sigma_hr_pixels"]
    )
    radiometry = _radiometry(args.radiometry_json, device)
    train_loader, val_loader, _, info = build_augsburg_real_loaders(
        args.cache_root,
        train_patch_size=args.train_patch_size,
        train_stride=args.train_stride,
        eval_patch_size=args.eval_patch_size,
        min_valid_fraction=args.min_valid_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    base_process = build_augsburg_real_process(
        effective_sigma=sigma,
        diffusion_steps=args.diffusion_steps,
    )
    geometry, diffusion, projector, p0, srf = _build_frozen(
        args, info, device, base_process
    )
    refiner = _build_refiner(args, info, device)
    if not args.resume:
        _load_refiner_weights(
            refiner, args.init_refiner_checkpoint, device
        )
    optimizer = torch.optim.Adam(
        refiner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ensure_dir(args.checkpoint_root)
    ensure_dir(args.log_root)
    best_path = os.path.join(
        args.checkpoint_root, args.save_name + ".pth"
    )
    last_path = os.path.join(
        args.checkpoint_root, args.save_name + "_last.pth"
    )
    start_epoch = 0
    best = _best_initial(args.monitor)
    if args.resume:
        start_epoch, best = _load_training_refiner(
            refiner, args.resume, device, optimizer
        )

    logger = CSVLogger(
        os.path.join(args.log_root, args.save_name + ".csv"),
        [
            "epoch", "loss", "l1", "ang_rad", "ref_l1", "phy_l1", "msi_l1",
            "base_ref_psnr", "base_ref_sam", "base_ref_sam_high",
            "ref_psnr", "ref_sam", "ref_sam_high", "ref_sam_low",
            "dpsnr", "dsam", "dsam_high", "update_ratio", "best",
        ],
    )
    print(
        "AUGSBURG_REAL_E geometry=frozen diffusion=frozen "
        "terminal_GIGI_only output_frame=real_S2 "
        f"scale=3 stages={base_process.stages} sigma={sigma:.6f} "
        f"monitor={args.monitor}"
    )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr = train_one_epoch(
            refiner,
            diffusion,
            geometry,
            projector,
            train_loader,
            optimizer,
            base_process=base_process,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            args=args,
            device=device,
        )
        print(
            f"epoch={epoch:03d}/{args.epochs} loss={tr['loss']:.7f} "
            f"l1={tr['l1']:.7f} ang={tr['ang']:.7f} "
            f"ref={tr['ref']:.7f} phy={tr['phy']:.7f} "
            f"msi={tr['msi']:.7f}"
        )
        row = {
            "epoch": epoch,
            "loss": tr["loss"],
            "l1": tr["l1"],
            "ang_rad": tr["ang"],
            "ref_l1": tr["ref"],
            "phy_l1": tr["phy"],
            "msi_l1": tr["msi"],
            "base_ref_psnr": "",
            "base_ref_sam": "",
            "base_ref_sam_high": "",
            "ref_psnr": "",
            "ref_sam": "",
            "ref_sam_high": "",
            "ref_sam_low": "",
            "dpsnr": "",
            "dsam": "",
            "dsam_high": "",
            "update_ratio": "",
            "best": best,
        }
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            va = evaluate(
                refiner,
                diffusion,
                geometry,
                projector,
                val_loader,
                base_process=base_process,
                p0=p0,
                srf=srf,
                radiometry=radiometry,
                args=args,
                device=device,
            )
            _run_eval_print(va, "REAL_E_VAL")
            value = va[args.monitor]
            if _better(value, best, args.monitor):
                best = value
                _save_refiner(
                    refiner,
                    optimizer,
                    epoch,
                    best,
                    best_path,
                    args,
                )
                print(
                    f"SAVED_BEST {best_path} "
                    f"{args.monitor}={best:.6f}"
                )
            row.update(
                {
                    "base_ref_psnr": va["base_ref_psnr"],
                    "base_ref_sam": va["base_ref_sam"],
                    "base_ref_sam_high": va["base_ref_sam_high"],
                    "ref_psnr": va["ref_psnr"],
                    "ref_sam": va["ref_sam"],
                    "ref_sam_high": va["ref_sam_high"],
                    "ref_sam_low": va["ref_sam_low"],
                    "dpsnr": va["ref_psnr"] - va["base_ref_psnr"],
                    "dsam": va["ref_sam"] - va["base_ref_sam"],
                    "dsam_high": (
                        va["ref_sam_high"]
                        - va["base_ref_sam_high"]
                    ),
                    "update_ratio": va["update_ratio"],
                    "best": best,
                }
            )
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            _save_refiner(
                refiner,
                optimizer,
                epoch,
                best,
                last_path,
                args,
            )
        logger.write(row)


def test(args):
    if not args.refiner_checkpoint:
        raise ValueError("--stage test requires --refiner_checkpoint")
    set_seed(args.seed)
    device = get_device(args.device)
    sigma = float(
        _load_json(args.psf_json)["terminal_sigma_hr_pixels"]
    )
    radiometry = _radiometry(args.radiometry_json, device)
    _, _, test_loader, info = build_augsburg_real_loaders(
        args.cache_root,
        train_patch_size=args.train_patch_size,
        train_stride=args.train_stride,
        eval_patch_size=args.eval_patch_size,
        min_valid_fraction=args.min_valid_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    base_process = build_augsburg_real_process(
        effective_sigma=sigma,
        diffusion_steps=args.diffusion_steps,
    )
    geometry, diffusion, projector, p0, srf = _build_frozen(
        args, info, device, base_process
    )
    refiner = _build_refiner(args, info, device)
    epoch, best = _load_training_refiner(
        refiner, args.refiner_checkpoint, device
    )
    print(
        f"REFINER_LOAD epoch={epoch} best={best:.6f} "
        f"path={args.refiner_checkpoint}"
    )
    metrics = evaluate(
        refiner,
        diffusion,
        geometry,
        projector,
        test_loader,
        base_process=base_process,
        p0=p0,
        srf=srf,
        radiometry=radiometry,
        args=args,
        device=device,
    )
    _run_eval_print(metrics, "FINAL_AUGSBURG_REAL")


def main():
    args = parse_args()
    if args.stage == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
