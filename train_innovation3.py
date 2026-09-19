"""Train/test Innovation 3 on top of the registered Innovation-1 Raw-MSI Direct baseline.

The first experiment intentionally freezes the Raw-Direct backbone and keeps
the original L1 + SAM objective unchanged.  Only the spectral-refinement branch
is optimized.  This isolates the structural effect of heterogeneity guidance.

Ablation variants:
    generic : spectral residual only
    hetero  : + MSI heterogeneity gate
    broad   : + C4:L broad-shape DCT projection
    full    : + spectral-direction tangent projection
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from config import TrainConfig
from data_loader import build_loaders
from innovation1 import batch_state_at, build_progressive_process
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models import (
    HeterogeneityGuidedSpectralPredictor,
    RawMSIDirectPredictor,
    load_legacy_raw_direct_checkpoint,
    ranked_msi_heterogeneity,
)
from utils import CSVLogger, ensure_dir, get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Innovation-3 heterogeneity-guided spectral refinement")
    p.add_argument("--stage", choices=["train", "test"], default="train")
    p.add_argument("--variant", choices=["generic", "hetero", "broad", "full"], default="full")
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)

    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)

    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--refine_hidden", type=int, default=64)

    p.add_argument(
        "--baseline_checkpoint",
        default="./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth",
    )
    p.add_argument("--baseline_checkpoint_type", choices=["legacy", "standard"], default="legacy")
    p.add_argument("--refiner_checkpoint", default="")
    p.add_argument("--resume", default="")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_sam", type=float, default=0.1)
    p.add_argument("--boundary_probability", type=float, default=0.2)
    p.add_argument("--boundary_radius", type=int, default=1)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument("--checkpoint_root", default="./checkpoints/innovation3")
    p.add_argument("--log_root", default="./logs")
    p.add_argument("--save_name", default="")
    return p.parse_args()


def _config(args) -> TrainConfig:
    return TrainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        test_size=args.test_size,
        scale_ratio=args.scale_ratio,
        srf_interp=args.srf_interp,
        degradation_mode="physical",
        diffusion_steps=args.diffusion_steps,
        lift_mode="normalized_adjoint",
        mtf_nyquist=args.mtf_nyquist,
        psf_truncate=args.psf_truncate,
        predictor="raw_direct",
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        spectral_hidden=args.spectral_hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )


def _build_baseline(args, info, device):
    model = RawMSIDirectPredictor(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_msi_bands"],
        total_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        residual_prediction=True,
        spectral_hidden=args.spectral_hidden,
    ).to(device)
    if args.baseline_checkpoint_type == "legacy":
        report = load_legacy_raw_direct_checkpoint(
            model,
            args.baseline_checkpoint,
            map_location=str(device),
        )
        print("BASELINE_LOAD legacy", report)
    else:
        epoch, best = load_checkpoint(
            model,
            args.baseline_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        print(f"BASELINE_LOAD standard epoch={epoch} best={best}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _build_model(args, info, device):
    baseline = _build_baseline(args, info, device)
    model = HeterogeneityGuidedSpectralPredictor(
        baseline,
        n_bands=info["n_bands"],
        total_steps=args.diffusion_steps,
        hidden_channels=args.refine_hidden,
        variant=args.variant,
        freeze_backbone=True,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"INNOVATION3 variant={args.variant} trainable={trainable/1e6:.4f}M "
        f"total_with_frozen_backbone={total/1e6:.4f}M low_dct_C4L=[4,{model.refiner.low_end-1}]"
    )
    return model


def _checkpoint_path(args) -> str:
    ensure_dir(args.checkpoint_root)
    name = args.save_name or f"{args.dataset}_innovation3_{args.variant}.pth"
    if not name.endswith(".pth"):
        name += ".pth"
    return os.path.join(args.checkpoint_root, name)


def _save_refiner_checkpoint(model, optimizer, epoch: int, best_high_sam: float, path: str, args):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": int(epoch),
            "best_high_sam": float(best_high_sam),
            "variant": model.variant,
            "refiner": model.refiner.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "extra": vars(args),
        },
        path,
    )


def _load_refiner_checkpoint(model, path: str, device, optimizer=None):
    if not path:
        raise ValueError("refiner checkpoint path is empty")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    checkpoint_variant = state.get("variant", model.variant)
    if checkpoint_variant != model.variant:
        raise ValueError(
            f"checkpoint variant={checkpoint_variant!r} but requested variant={model.variant!r}"
        )
    model.refiner.load_state_dict(state["refiner"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0)), float(state.get("best_high_sam", float("inf")))


def _pixel_sam_deg(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    dot = (pred.float() * target.float()).sum(dim=1)
    pn = torch.linalg.vector_norm(pred.float(), dim=1)
    tn = torch.linalg.vector_norm(target.float(), dim=1)
    cos = dot / (pn * tn).clamp_min(eps)
    return torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * (180.0 / math.pi)


def _region_sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    risk: torch.Tensor,
    fraction: float,
) -> Tuple[float, float]:
    sam = _pixel_sam_deg(pred, target)
    sam = sam[:, 1:-1, 1:-1].reshape(-1)
    r = risk[:, 1:-1, 1:-1].reshape(-1)
    valid = torch.isfinite(sam) & torch.isfinite(r)
    sam = sam[valid]
    r = r[valid]
    lo = torch.quantile(r, fraction)
    hi = torch.quantile(r, 1.0 - fraction)
    return float(sam[r >= hi].mean().item()), float(sam[r <= lo].mean().item())


@torch.no_grad()
def _reverse_baseline(model, process, gt, hr_msi):
    terminal_lr = process.terminal_observation(gt)
    x_t = process.terminal_state(terminal_lr, target_size=tuple(gt.shape[-2:]))
    for t in range(process.total_steps, 0, -1):
        step = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        pred = model(x_t, hr_msi, step)
        x_t = process.reverse_update(x_t, pred, t)
    return x_t


@torch.no_grad()
def _reverse_refined(model, process, gt, hr_msi):
    terminal_lr = process.terminal_observation(gt)
    x_t = process.terminal_state(terminal_lr, target_size=tuple(gt.shape[-2:]))
    last_details = None
    for t in range(process.total_steps, 0, -1):
        step = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        pred, details = model.forward_with_details(x_t, hr_msi, step)
        if t == 1:
            last_details = details
        x_t = process.reverse_update(x_t, pred, t)
    return x_t, last_details


def _mechanism_stats(model, details, fraction: float) -> Dict[str, float]:
    if details is None:
        return {}
    risk = details["risk"]
    update = details["pre_update"]
    base = details["base_x0"]
    energy = torch.linalg.vector_norm(update, dim=1)

    interior_r = risk[:, 1:-1, 1:-1].reshape(-1)
    interior_e = energy[:, 1:-1, 1:-1].reshape(-1)
    lo = torch.quantile(interior_r, fraction)
    hi = torch.quantile(interior_r, 1.0 - fraction)
    e_high = float(interior_e[interior_r >= hi].mean().item())
    e_low = float(interior_e[interior_r <= lo].mean().item())

    coeff = model.refiner.dct(update)
    band_energy = coeff.square().sum(dim=(0, 2, 3))
    total = band_energy.sum().clamp_min(1e-12)
    c4l = band_energy[4:model.refiner.low_end].sum() / total

    dot = (update * base).sum(dim=1).abs()
    denom = (
        torch.linalg.vector_norm(update, dim=1)
        * torch.linalg.vector_norm(base, dim=1)
    ).clamp_min(1e-12)
    nonzero = torch.linalg.vector_norm(update, dim=1) > 1e-10
    tangent_cos = float((dot[nonzero] / denom[nonzero]).mean().item()) if nonzero.any() else 0.0

    return {
        "UPDATE_HIGH": e_high,
        "UPDATE_LOW": e_low,
        "UPDATE_HIGH_LOW_RATIO": e_high / max(e_low, 1e-12),
        "UPDATE_C4L_FRACTION": float(c4l.item()),
        "UPDATE_BASE_ABS_COS": tangent_cos,
        "GATE_MEAN": float(details["gate"].mean().item()),
        "ETA": float(model.refiner.eta.item()),
        "GATE_SLOPE": float(model.refiner.gate_slope.item()),
        "GATE_BIAS": float(model.refiner.gate_bias.item()),
    }


@torch.no_grad()
def evaluate_pair(model, loader, process, device, scale_ratio: int, region_fraction: float):
    model.eval()
    baseline_meter = MetricAverager()
    refined_meter = MetricAverager()
    base_high = []
    base_low = []
    refined_high = []
    refined_low = []
    mechanism_rows = []

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        risk = ranked_msi_heterogeneity(hr_msi)

        base_pred = _reverse_baseline(model.backbone, process, gt, hr_msi)
        refined_pred, details = _reverse_refined(model, process, gt, hr_msi)

        baseline_meter.update(calc_metrics(base_pred, gt, scale_ratio))
        refined_meter.update(calc_metrics(refined_pred, gt, scale_ratio))

        bh, bl = _region_sam(base_pred, gt, risk, region_fraction)
        rh, rl = _region_sam(refined_pred, gt, risk, region_fraction)
        base_high.append(bh)
        base_low.append(bl)
        refined_high.append(rh)
        refined_low.append(rl)
        mechanism_rows.append(_mechanism_stats(model, details, region_fraction))

    baseline = baseline_meter.average()
    refined = refined_meter.average()
    baseline["SAM_HIGH"] = sum(base_high) / len(base_high)
    baseline["SAM_LOW"] = sum(base_low) / len(base_low)
    refined["SAM_HIGH"] = sum(refined_high) / len(refined_high)
    refined["SAM_LOW"] = sum(refined_low) / len(refined_low)

    mechanism = {}
    if mechanism_rows:
        for key in mechanism_rows[0]:
            mechanism[key] = sum(row[key] for row in mechanism_rows) / len(mechanism_rows)
    return baseline, refined, mechanism


def _print_eval(baseline, refined, mechanism):
    print(
        "A0_BASE "
        f"PSNR={baseline['PSNR']:.6f} SAM={baseline['SAM']:.6f} "
        f"SAM_HIGH={baseline['SAM_HIGH']:.6f} SAM_LOW={baseline['SAM_LOW']:.6f}"
    )
    print(
        "VARIANT "
        f"PSNR={refined['PSNR']:.6f} SAM={refined['SAM']:.6f} "
        f"SAM_HIGH={refined['SAM_HIGH']:.6f} SAM_LOW={refined['SAM_LOW']:.6f}"
    )
    print(
        "DELTA "
        f"dPSNR={refined['PSNR']-baseline['PSNR']:+.6f} "
        f"dSAM={refined['SAM']-baseline['SAM']:+.6f} "
        f"dSAM_HIGH={refined['SAM_HIGH']-baseline['SAM_HIGH']:+.6f} "
        f"dSAM_LOW={refined['SAM_LOW']-baseline['SAM_LOW']:+.6f}"
    )
    if mechanism:
        print(
            "MECHANISM "
            + " ".join(f"{key}={value:.6f}" for key, value in mechanism.items())
        )


def train(args):
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    train_loader, test_loader, info = build_loaders(cfg)
    process = build_progressive_process(cfg)
    model = _build_model(args, info, device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    best_high_sam = float("inf")
    if args.resume:
        loaded_epoch, best_high_sam = _load_refiner_checkpoint(
            model, args.resume, device, optimizer=optimizer
        )
        start_epoch = loaded_epoch + 1
        print(f"RESUME epoch={loaded_epoch} best_high_sam={best_high_sam:.6f}")

    sam_loss_fn = SAMLoss()
    best_path = _checkpoint_path(args)
    stem, ext = os.path.splitext(best_path)
    last_path = stem + "_last" + ext
    logger = CSVLogger(
        os.path.join(args.log_root, f"{args.dataset}_innovation3_{args.variant}.csv"),
        [
            "epoch", "loss", "l1", "sam",
            "PSNR", "SAM", "SAM_HIGH", "SAM_LOW",
            "dSAM_HIGH", "dSAM_LOW", "best_SAM_HIGH",
        ],
    )

    print(
        "TRAINING_POLICY backbone=frozen objective=baseline_L1_plus_SAM "
        "checkpoint_monitor=held-out_center_patch_SAM_HIGH"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = l1_sum = sam_sum = 0.0
        count = 0
        for batch in train_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            timesteps = process.sample_timesteps(
                gt.shape[0],
                boundary_probability=args.boundary_probability,
                boundary_radius=args.boundary_radius,
                device=device,
            )
            with torch.no_grad():
                x_t = batch_state_at(process, gt, timesteps)

            pred = model(x_t, hr_msi, timesteps)
            l1 = F.l1_loss(pred, gt)
            sam = sam_loss_fn(pred, gt)
            loss = args.lambda_l1 * l1 + args.lambda_sam * sam
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Innovation-3 loss")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                    error_if_nonfinite=True,
                )
            optimizer.step()

            batch_size = gt.shape[0]
            loss_sum += float(loss.item()) * batch_size
            l1_sum += float(l1.item()) * batch_size
            sam_sum += float(sam.item()) * batch_size
            count += batch_size

        row = {
            "epoch": epoch,
            "loss": loss_sum / max(count, 1),
            "l1": l1_sum / max(count, 1),
            "sam": sam_sum / max(count, 1),
            "PSNR": "",
            "SAM": "",
            "SAM_HIGH": "",
            "SAM_LOW": "",
            "dSAM_HIGH": "",
            "dSAM_LOW": "",
            "best_SAM_HIGH": best_high_sam,
        }
        print(
            f"epoch={epoch:03d} loss={row['loss']:.7f} "
            f"l1={row['l1']:.7f} sam={row['sam']:.7f}"
        )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            baseline, refined, mechanism = evaluate_pair(
                model,
                test_loader,
                process,
                device,
                args.scale_ratio,
                args.region_fraction,
            )
            _print_eval(baseline, refined, mechanism)
            row.update(
                {
                    "PSNR": refined["PSNR"],
                    "SAM": refined["SAM"],
                    "SAM_HIGH": refined["SAM_HIGH"],
                    "SAM_LOW": refined["SAM_LOW"],
                    "dSAM_HIGH": refined["SAM_HIGH"] - baseline["SAM_HIGH"],
                    "dSAM_LOW": refined["SAM_LOW"] - baseline["SAM_LOW"],
                }
            )
            if refined["SAM_HIGH"] < best_high_sam:
                best_high_sam = refined["SAM_HIGH"]
                _save_refiner_checkpoint(
                    model, optimizer, epoch, best_high_sam, best_path, args
                )
                print(f"SAVED_BEST {best_path} SAM_HIGH={best_high_sam:.6f}")
            row["best_SAM_HIGH"] = best_high_sam

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            _save_refiner_checkpoint(
                model, optimizer, epoch, best_high_sam, last_path, args
            )
        logger.write(row)


def test(args):
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    _, test_loader, info = build_loaders(cfg)
    process = build_progressive_process(cfg)
    model = _build_model(args, info, device)

    path = args.refiner_checkpoint or _checkpoint_path(args)
    epoch, best = _load_refiner_checkpoint(model, path, device)
    print(f"REFINER_LOAD path={path} epoch={epoch} best_SAM_HIGH={best:.6f}")
    baseline, refined, mechanism = evaluate_pair(
        model,
        test_loader,
        process,
        device,
        args.scale_ratio,
        args.region_fraction,
    )
    _print_eval(baseline, refined, mechanism)
    print(
        "FULL_METRICS "
        + " ".join(
            f"{key}={refined[key]:.6f}"
            for key in ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")
        )
    )


def main():
    args = parse_args()
    if args.stage == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
