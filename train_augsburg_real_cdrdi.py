"""Real-C: self-supervised CDRDI adaptation on real Sentinel-2 + MDAS EnMAP30.

No synthetic deformation and no flow/deformation ground truth are used here.
Checkpoint selection is based only on observable cross-sensor physical closure.
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
from cdrdi_geometry import deformation_regularizer, jacobian_determinant, spectral_project
from degradations.effective_gaussian import EffectiveGaussianDegradation
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver, image_gradients
from utils import CSVLogger, ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Augsburg-Real Real-C geometry adaptation")
    p.add_argument("--cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--psf_json", default="./data/calibration/AugsburgReal_effective_psf.json")
    p.add_argument("--init_checkpoint", required=True, help="Synthetic Augsburg Stage-C checkpoint")
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
    p.add_argument("--jac_margin", type=float, default=0.1)
    p.add_argument("--local_window", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--resume", default="")
    return p.parse_args()


def _load_sigma(path: str) -> float:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("scale_ratio", 0)) != 3:
        raise ValueError("Augsburg-Real PSF calibration must use scale_ratio=3")
    return float(payload["terminal_sigma_hr_pixels"])


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


def real_geometry_loss(
    outputs: Dict[str, object],
    target: torch.Tensor,
    mask: torch.Tensor,
    args,
):
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
    total = (
        args.lambda_norm * norm_loss
        + args.lambda_grad * grad_loss
        + args.lambda_raw * raw_loss
        + args.lambda_def * reg
        + args.lambda_jac * jac_penalty
    )
    return total, {
        "norm": norm_loss,
        "grad": grad_loss,
        "raw": raw_loss,
        "reg": reg,
        "jac_penalty": jac_penalty,
        "min_jac": jac.amin(),
    }


def _estimate_batch(model, batch, *, p0, srf, steps: int, device):
    lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
    hr_msi = batch["hr_msi"].to(device, non_blocking=True)
    mask_hr = batch["valid_mask"].to(device, non_blocking=True)
    target = spectral_project(lr_hsi, srf)
    mask = _lr_mask(mask_hr)
    outputs = model(target, hr_msi, p0, steps=steps)
    return target, mask, outputs


def train_one_epoch(model, loader, optimizer, *, p0, srf, args, device):
    model.train()
    sums = {k: 0.0 for k in ("loss", "norm", "grad", "raw", "reg", "jac")}
    count = 0
    for batch in loader:
        target, mask, outputs = _estimate_batch(
            model, batch, p0=p0, srf=srf, steps=args.train_steps, device=device
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
        count += n
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, *, p0, srf, args, device):
    model.eval()
    rows = []
    for batch in loader:
        target, mask, outputs = _estimate_batch(
            model, batch, p0=p0, srf=srf, steps=args.eval_steps, device=device
        )
        _, parts = real_geometry_loss(outputs, target, mask, args)
        initial = outputs["initial_prediction"]
        initial_n = _local_standardize(initial, args.local_window)
        target_n = _local_standardize(target, args.local_window)
        initial_norm = float(_charbonnier(target_n - initial_n, mask).item())
        rigid = outputs["final_rigid"]
        local = outputs["final_local_field"]
        rows.append(
            {
                "initial_norm": initial_norm,
                "norm": float(parts["norm"].item()),
                "grad": float(parts["grad"].item()),
                "raw": float(parts["raw"].item()),
                "min_jac": float(parts["min_jac"].item()),
                "dx_abs": float(rigid[:, 0].abs().mean().item()),
                "dy_abs": float(rigid[:, 1].abs().mean().item()),
                "theta_abs": float(rigid[:, 2].abs().mean().item()),
                "local_mean": float(torch.linalg.vector_norm(local, dim=1).mean().item()),
            }
        )
    if not rows:
        raise ValueError("empty validation loader")
    out = {k: mean(row[k] for row in rows) for k in rows[0]}
    out["closure_reduction"] = 1.0 - out["norm"] / max(out["initial_norm"], 1e-12)
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    sigma = _load_sigma(args.psf_json)

    train_loader, val_loader, _, info = build_augsburg_real_loaders(
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
            "theta_abs", "local_mean", "best_val_norm",
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
        "AUGSBURG_REAL_C supervision=observable_physical_closure_only "
        f"sigma={sigma:.6f} scale=3 stages=1,2,3 train_steps={args.train_steps} "
        f"eval_steps={args.eval_steps}"
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
            args=args,
            device=device,
        )
        print(
            f"epoch={epoch:03d}/{args.epochs} loss={tr['loss']:.8f} "
            f"norm={tr['norm']:.8f} grad={tr['grad']:.8f} raw={tr['raw']:.8f}"
        )
        if epoch % args.eval_interval != 0 and epoch != args.epochs:
            continue

        va = evaluate(model, val_loader, p0=p0, srf=srf, args=args, device=device)
        print(
            "REAL_C_VAL "
            f"INIT_NORM={va['initial_norm']:.8f} FINAL_NORM={va['norm']:.8f} "
            f"REDUCTION={100*va['closure_reduction']:.3f}% RAW={va['raw']:.8f} "
            f"MIN_JAC={va['min_jac']:.6f} DX={va['dx_abs']:.4f} "
            f"DY={va['dy_abs']:.4f} THETA={va['theta_abs']:.4f} "
            f"LOCAL={va['local_mean']:.4f}"
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
                "best_val_norm": min(best, va["norm"]),
            }
        )
        if va["norm"] < best:
            best = va["norm"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best,
                checkpoint,
                extra={
                    "stage": "AugsburgReal-C",
                    "observable_monitor": "local_normalized_closure",
                    "effective_sigma": sigma,
                    "scale_ratio": 3,
                    "stages": [1, 2, 3],
                    "train_steps": args.train_steps,
                    "eval_steps": args.eval_steps,
                    "flow_gt_used": False,
                },
            )
            print(f"SAVED_BEST {checkpoint} val_norm={best:.8f}")


if __name__ == "__main__":
    main()
