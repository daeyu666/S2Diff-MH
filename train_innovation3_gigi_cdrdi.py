"""Innovation-3 terminal GIGI on top of the final CDRDI nonregistered pipeline.

This script freezes both completed modules:
  (1) CDRDI geometry solver;
  (2) Stage-2D estimated-geometry-aware diffusion model.

For each synthetic nonregistered HSI acquisition:
    hidden phi_GT -> Y_H = A_(T,phi_GT)(X)
    (Y_H, Y_M)    -> frozen CDRDI -> phi_hat
    phi_hat        -> frozen deformation-aware diffusion -> X_hat
    terminal residual under A_(T,phi_hat)
                   -> R_phy
    (X_hat, Y_M, R_phy, H_M)
                   -> terminal GIGI -> X_final

The observed LR-HSI is never inverse warped.  X_hat and X_final remain in the
reliable HR-MSI coordinate system.

The GIGI refiner is the same full-resolution terminal module used by
train_innovation3_gigi.py.  It can optionally initialize from the registered
GIGI checkpoint and then adapt only the terminal refiner to the nonregistered
CDRDI output distribution.
"""

from __future__ import annotations

import argparse
import math
import os
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from cdrdi_geometry import (
    SyntheticGeometry,
    sample_synthetic_geometry,
    sampling_coordinates,
    spectral_project,
)
from config import TrainConfig
from data_loader import build_loaders
from degradations.deformation_aware import DeformationAwareProgressiveDegradation
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from losses import SAMLoss
from main import build_model
from metrics import MetricAverager, calc_metrics
from models import (
    HeterogeneityGuidedSpectralRefiner,
    TerminalGIGISpectralRefiner,
    ranked_msi_heterogeneity,
)
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from train_cdrdi_diffusion_oracle import sample_training_geometry
from utils import CSVLogger, ensure_dir, get_device, load_checkpoint, set_seed


VARIANTS = ("conv", "gigi", "gigi_hetero", "gigi_phy", "full")
METRIC_KEYS = ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 terminal GIGI refinement on final CDRDI nonregistered fusion"
    )
    p.add_argument("--stage", choices=["train", "test"], default="train")
    p.add_argument("--variant", choices=VARIANTS, default="full")
    p.add_argument(
        "--dataset",
        choices=["PaviaU", "Houston13", "Chikusei"],
        default="PaviaU",
    )
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

    # Keep these identical to the final Innovation-2 protocol by default.
    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)
    p.add_argument("--identity_probability", type=float, default=0.10)
    p.add_argument("--min_strength", type=float, default=0.20)

    p.add_argument("--geometry_steps", type=int, default=9)
    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument(
        "--geometry_checkpoint",
        default="./checkpoints/cdrdi_stage1/PaviaU_recursive_k6_finalonly_300ep_lr1e4.pth",
    )

    p.add_argument("--diffusion_base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--diffusion_checkpoint",
        default="./checkpoints/cdrdi_stage2/PaviaU_estimated_deform_diffusion_k9_stage2d_A.pth",
    )

    p.add_argument("--refine_hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument(
        "--disable_tangent",
        action="store_true",
        help="Disable tangent projection of the learned terminal update.",
    )
    p.add_argument(
        "--init_refiner_checkpoint",
        default="./checkpoints/innovation3_gigi/PaviaU_innovation3_gigi_full.pth",
        help="Optional registered-GIGI checkpoint used only to initialize the refiner.",
    )
    p.add_argument("--refiner_checkpoint", default="")
    p.add_argument("--resume", default="")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_ang", type=float, default=0.1)
    p.add_argument("--lambda_phy", type=float, default=0.1)
    p.add_argument("--lambda_msi", type=float, default=0.1)

    p.add_argument("--eval_cases", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument(
        "--monitor",
        choices=["warp_sam_high", "warp_sam", "warp_psnr"],
        default="warp_sam_high",
    )
    p.add_argument(
        "--checkpoint_root",
        default="./checkpoints/innovation3_gigi_cdrdi",
    )
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
        base_channels=args.diffusion_base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        spectral_hidden=args.spectral_hidden,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _process_from_geometry(base_process, geometry: SyntheticGeometry):
    rigid = torch.stack([geometry.dx, geometry.dy, geometry.theta_deg], dim=1)
    return DeformationAwareProgressiveDegradation(
        base_process,
        rigid=rigid,
        local_field=geometry.local_field,
    )


def _process_from_estimate(base_process, rigid: torch.Tensor, local: torch.Tensor):
    return DeformationAwareProgressiveDegradation(
        base_process,
        rigid=rigid.detach(),
        local_field=local.detach(),
    )


def _estimate_geometry(
    geometry_model,
    *,
    y_h: torch.Tensor,
    hr_msi: torch.Tensor,
    p0,
    srf: torch.Tensor,
    steps: int,
):
    z_h = spectral_project(y_h, srf)
    out = geometry_model(z_h, hr_msi, p0, steps=steps)
    return out["final_rigid"].detach(), out["final_local_field"].detach()


def _geometry_epe(
    h: int,
    w: int,
    gt_phi: SyntheticGeometry,
    pred_rigid: torch.Tensor,
    pred_local: torch.Tensor,
) -> float:
    gt_x, gt_y = sampling_coordinates(
        h, w, gt_phi.dx, gt_phi.dy, gt_phi.theta_deg, gt_phi.local_field
    )
    pred_x, pred_y = sampling_coordinates(
        h,
        w,
        pred_rigid[:, 0],
        pred_rigid[:, 1],
        pred_rigid[:, 2],
        pred_local,
    )
    epe = torch.sqrt((pred_x - gt_x).pow(2) + (pred_y - gt_y).pow(2))
    return float(epe.mean().item())


def _build_frozen_models(args, info, device):
    geometry_model = LearnedPhysicalResidualSolver(
        info["n_msi_bands"],
        base_channels=args.geometry_base_channels,
        control_grid=args.control_grid,
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px,
    ).to(device)
    load_checkpoint(
        geometry_model,
        args.geometry_checkpoint,
        map_location=str(device),
        load_optimizer=False,
    )
    geometry_model.eval()
    for p in geometry_model.parameters():
        p.requires_grad_(False)

    cfg = _config(args)
    diffusion_model = build_model(cfg, info, device)
    load_checkpoint(
        diffusion_model,
        args.diffusion_checkpoint,
        map_location=str(device),
        load_optimizer=False,
    )
    diffusion_model.eval()
    for p in diffusion_model.parameters():
        p.requires_grad_(False)

    return geometry_model, diffusion_model


def _build_projector(args, info, device):
    projector = HeterogeneityGuidedSpectralRefiner(
        n_bands=info["n_bands"],
        total_steps=args.diffusion_steps,
        hidden_channels=1,
        variant="full",
    ).to(device)
    projector.eval()
    for p in projector.parameters():
        p.requires_grad_(False)
    return projector


def _build_refiner(args, info, device):
    model = TerminalGIGISpectralRefiner(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_msi_bands"],
        hidden_channels=args.refine_hidden,
        heads=args.heads,
        variant=args.variant,
        tangent_output=not args.disable_tangent,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"REFINER variant={args.variant} hidden={args.refine_hidden} heads={args.heads} "
        f"tangent={not args.disable_tangent} trainable={trainable/1e6:.4f}M"
    )
    return model


def _load_refiner_weights(model, path: str, device):
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    checkpoint_variant = state.get("variant", model.variant)
    if checkpoint_variant != model.variant:
        raise ValueError(
            f"init refiner variant={checkpoint_variant!r}, requested={model.variant!r}"
        )
    payload = state.get("model")
    if payload is None:
        raise KeyError(f"{path} does not contain key 'model'")
    model.load_state_dict(payload, strict=True)
    print(f"INIT_REFINER_LOAD {path}")


def _checkpoint_path(args):
    ensure_dir(args.checkpoint_root)
    name = (
        args.save_name
        or f"{args.dataset}_innovation3_gigi_cdrdi_{args.variant}_k{args.geometry_steps}"
    )
    if not name.endswith(".pth"):
        name += ".pth"
    return os.path.join(args.checkpoint_root, name)


def _save_checkpoint(model, optimizer, epoch, best_value, path, args):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_value),
            "monitor": args.monitor,
            "variant": model.variant,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "extra": vars(args),
        },
        path,
    )


def _load_training_checkpoint(model, path, device, optimizer=None):
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


def _hsi_to_msi(x: torch.Tensor, srf: torch.Tensor) -> torch.Tensor:
    return torch.einsum("mc,bchw->bmhw", srf, x)


@torch.no_grad()
def _physical_residual(
    process,
    projector,
    observed_terminal: torch.Tensor,
    base_pred: torch.Tensor,
) -> torch.Tensor:
    # Under nonregistration process is A_(T,phi_hat), so this is the direct
    # deformation-aware counterpart of U_T[Y_H-D_T(X_hat)].
    predicted_terminal = process.terminal_observation(base_pred)
    native_residual = observed_terminal - predicted_terminal
    lifted = process.terminal_state(
        native_residual,
        target_size=tuple(base_pred.shape[-2:]),
    )
    broad = projector.project_broadshape(lifted)
    return projector.project_tangent(
        broad,
        base_pred,
        preserve_broadshape=True,
    )


def _pixel_sam_deg(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    p = pred.float()
    t = target.float()
    dot = (p * t).sum(dim=1)
    pn = torch.linalg.vector_norm(p, dim=1)
    tn = torch.linalg.vector_norm(t, dim=1)
    cosine = dot / (pn * tn).clamp_min(eps)
    return torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * (
        180.0 / math.pi
    )


def _region_sam(pred, target, heterogeneity, fraction: float) -> Tuple[float, float]:
    sam = _pixel_sam_deg(pred, target)[:, 1:-1, 1:-1].reshape(-1)
    h = heterogeneity[:, 1:-1, 1:-1].reshape(-1)
    valid = torch.isfinite(sam) & torch.isfinite(h)
    sam = sam[valid]
    h = h[valid]
    lo = torch.quantile(h, fraction)
    hi = torch.quantile(h, 1.0 - fraction)
    return (
        float(sam[h >= hi].mean().item()),
        float(sam[h <= lo].mean().item()),
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _mechanism_stats(details, fraction: float) -> Dict[str, float]:
    update = details["update"]
    heterogeneity = details["heterogeneity"]
    r_phy = details["physical_input"]

    energy = torch.linalg.vector_norm(update.float(), dim=1)
    e = energy[:, 1:-1, 1:-1].reshape(-1)
    h = heterogeneity[:, 1:-1, 1:-1].reshape(-1)
    lo = torch.quantile(h, fraction)
    hi = torch.quantile(h, 1.0 - fraction)
    high = float(e[h >= hi].mean().item())
    low = float(e[h <= lo].mean().item())

    uf = update.float().permute(0, 2, 3, 1).reshape(-1, update.shape[1])
    rf = r_phy.float().permute(0, 2, 3, 1).reshape(-1, r_phy.shape[1])
    denom = torch.linalg.vector_norm(uf, dim=1) * torch.linalg.vector_norm(rf, dim=1)
    valid = denom > 1e-10
    update_phy_cos = 0.0
    if valid.any():
        update_phy_cos = float(
            (((uf[valid] * rf[valid]).sum(dim=1)) / denom[valid]).mean().item()
        )

    stats = {
        "UPDATE_HIGH": high,
        "UPDATE_LOW": low,
        "UPDATE_HIGH_LOW_RATIO": high / max(low, 1e-12),
        "UPDATE_PHY_COS": update_phy_cos,
        "UPDATE_ABS_MEAN": float(update.abs().mean().item()),
    }
    attention = details.get("attention")
    if attention is not None and attention.shape[-1] > 1:
        p = attention.float().clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=-1) / math.log(attention.shape[-1])
        stats["ATTN_ENTROPY"] = float(entropy.mean().item())
    return stats


@torch.no_grad()
def _registered_pair(
    diffusion_model,
    refiner,
    projector,
    base_process,
    gt,
    hr_msi,
):
    y_h = base_process.terminal_observation(gt)
    base = reconstruct_from_terminal_lr(
        diffusion_model,
        base_process,
        y_h,
        target_size=tuple(gt.shape[-2:]),
        hr_msi=hr_msi,
    )
    r_phy = _physical_residual(base_process, projector, y_h, base)
    heterogeneity = ranked_msi_heterogeneity(hr_msi)
    refined, details = refiner(
        base,
        hr_msi,
        r_phy,
        heterogeneity,
        return_details=True,
    )
    return base, refined, heterogeneity, details


@torch.no_grad()
def _warp_pair(
    diffusion_model,
    geometry_model,
    refiner,
    projector,
    base_process,
    p0,
    srf,
    gt,
    hr_msi,
    gt_phi,
    geometry_steps: int,
):
    true_process = _process_from_geometry(base_process, gt_phi)
    y_h = true_process.terminal_observation(gt)
    pred_rigid, pred_local = _estimate_geometry(
        geometry_model,
        y_h=y_h,
        hr_msi=hr_msi,
        p0=p0,
        srf=srf,
        steps=geometry_steps,
    )
    estimated_process = _process_from_estimate(base_process, pred_rigid, pred_local)
    base = reconstruct_from_terminal_lr(
        diffusion_model,
        estimated_process,
        y_h,
        target_size=tuple(gt.shape[-2:]),
        hr_msi=hr_msi,
    )
    r_phy = _physical_residual(estimated_process, projector, y_h, base)
    heterogeneity = ranked_msi_heterogeneity(hr_msi)
    refined, details = refiner(
        base,
        hr_msi,
        r_phy,
        heterogeneity,
        return_details=True,
    )
    epe = _geometry_epe(
        gt.shape[-2],
        gt.shape[-1],
        gt_phi,
        pred_rigid,
        pred_local,
    )
    return base, refined, heterogeneity, details, epe


@torch.no_grad()
def evaluate(
    refiner,
    diffusion_model,
    geometry_model,
    projector,
    test_loader,
    *,
    base_process,
    p0,
    srf,
    args,
    device,
):
    refiner.eval()
    batch = next(iter(test_loader))
    gt = batch["gt"].to(device, non_blocking=True)
    hr_msi = batch["hr_msi"].to(device, non_blocking=True)
    if gt.shape[0] != 1:
        raise ValueError("evaluation expects the standard single test patch")

    meters = {
        "registered_base": MetricAverager(),
        "registered_refined": MetricAverager(),
        "warp_base": MetricAverager(),
        "warp_refined": MetricAverager(),
    }
    region: Dict[str, List[float]] = {
        "registered_base_high": [],
        "registered_base_low": [],
        "registered_refined_high": [],
        "registered_refined_low": [],
        "warp_base_high": [],
        "warp_base_low": [],
        "warp_refined_high": [],
        "warp_refined_low": [],
    }
    mechanism_rows: List[Dict[str, float]] = []
    epes: List[float] = []

    reg_base, reg_refined, reg_h, reg_details = _registered_pair(
        diffusion_model,
        refiner,
        projector,
        base_process,
        gt,
        hr_msi,
    )
    reg_base_metrics = calc_metrics(reg_base, gt, args.scale_ratio)
    reg_ref_metrics = calc_metrics(reg_refined, gt, args.scale_ratio)
    rb_h, rb_l = _region_sam(reg_base, gt, reg_h, args.region_fraction)
    rr_h, rr_l = _region_sam(reg_refined, gt, reg_h, args.region_fraction)

    generator = _make_generator(device, args.seed + 70000)
    h, w = gt.shape[-2:]
    for _ in range(int(args.eval_cases)):
        gt_phi = sample_synthetic_geometry(
            h,
            w,
            device=device,
            dtype=gt.dtype,
            generator=generator,
            max_translation=args.max_translation,
            max_rotation_deg=args.max_rotation_deg,
            max_local_px=args.max_local_px,
            control_grid=args.control_grid,
            min_jacobian=args.min_jacobian,
        )
        warp_base, warp_refined, warp_h, details, epe = _warp_pair(
            diffusion_model,
            geometry_model,
            refiner,
            projector,
            base_process,
            p0,
            srf,
            gt,
            hr_msi,
            gt_phi,
            args.geometry_steps,
        )
        epes.append(epe)

        meters["registered_base"].update(reg_base_metrics)
        meters["registered_refined"].update(reg_ref_metrics)
        meters["warp_base"].update(calc_metrics(warp_base, gt, args.scale_ratio))
        meters["warp_refined"].update(calc_metrics(warp_refined, gt, args.scale_ratio))

        region["registered_base_high"].append(rb_h)
        region["registered_base_low"].append(rb_l)
        region["registered_refined_high"].append(rr_h)
        region["registered_refined_low"].append(rr_l)
        wb_h, wb_l = _region_sam(warp_base, gt, warp_h, args.region_fraction)
        wr_h, wr_l = _region_sam(
            warp_refined, gt, warp_h, args.region_fraction
        )
        region["warp_base_high"].append(wb_h)
        region["warp_base_low"].append(wb_l)
        region["warp_refined_high"].append(wr_h)
        region["warp_refined_low"].append(wr_l)
        mechanism_rows.append(_mechanism_stats(details, args.region_fraction))

    out = {name: meter.average() for name, meter in meters.items()}
    out["registered_base"]["SAM_HIGH"] = _mean(region["registered_base_high"])
    out["registered_base"]["SAM_LOW"] = _mean(region["registered_base_low"])
    out["registered_refined"]["SAM_HIGH"] = _mean(
        region["registered_refined_high"]
    )
    out["registered_refined"]["SAM_LOW"] = _mean(
        region["registered_refined_low"]
    )
    out["warp_base"]["SAM_HIGH"] = _mean(region["warp_base_high"])
    out["warp_base"]["SAM_LOW"] = _mean(region["warp_base_low"])
    out["warp_refined"]["SAM_HIGH"] = _mean(region["warp_refined_high"])
    out["warp_refined"]["SAM_LOW"] = _mean(region["warp_refined_low"])

    mechanism: Dict[str, float] = {}
    if mechanism_rows:
        for key in mechanism_rows[0]:
            mechanism[key] = _mean(
                row[key] for row in mechanism_rows if key in row
            )
    return out, mechanism, mean(epes)


def _short(m: Dict[str, float]) -> str:
    return (
        f"PSNR={m['PSNR']:.6f} SAM={m['SAM']:.6f} "
        f"SAM_HIGH={m['SAM_HIGH']:.6f} SAM_LOW={m['SAM_LOW']:.6f} "
        f"SSIM={m['SSIM']:.6f} ERGAS={m['ERGAS']:.6f} "
        f"CC={m['CC']:.6f} RMSE={m['RMSE']:.6f}"
    )


def _print_eval(metrics, mechanism, geom_epe):
    rb = metrics["registered_base"]
    rr = metrics["registered_refined"]
    wb = metrics["warp_base"]
    wr = metrics["warp_refined"]

    print(f"A0_REGISTERED {_short(rb)}")
    print(f"GIGI_REGISTERED {_short(rr)}")
    print(
        "DELTA_REGISTERED "
        f"dPSNR={rr['PSNR']-rb['PSNR']:+.6f} "
        f"dSAM={rr['SAM']-rb['SAM']:+.6f} "
        f"dSAM_HIGH={rr['SAM_HIGH']-rb['SAM_HIGH']:+.6f} "
        f"dSAM_LOW={rr['SAM_LOW']-rb['SAM_LOW']:+.6f}"
    )
    print(f"A0_WARP {_short(wb)}")
    print(f"GIGI_WARP {_short(wr)}")
    print(
        "DELTA_WARP "
        f"dPSNR={wr['PSNR']-wb['PSNR']:+.6f} "
        f"dSAM={wr['SAM']-wb['SAM']:+.6f} "
        f"dSAM_HIGH={wr['SAM_HIGH']-wb['SAM_HIGH']:+.6f} "
        f"dSAM_LOW={wr['SAM_LOW']-wb['SAM_LOW']:+.6f}"
    )
    print(f"GEOMETRY AVG_EPE_HR={geom_epe:.6f}px")
    if mechanism:
        print(
            "MECHANISM_WARP "
            + " ".join(f"{k}={v:.6f}" for k, v in mechanism.items())
        )


def _monitor_value(metrics, mode: str) -> float:
    wr = metrics["warp_refined"]
    if mode == "warp_sam_high":
        return float(wr["SAM_HIGH"])
    if mode == "warp_sam":
        return float(wr["SAM"])
    if mode == "warp_psnr":
        return float(wr["PSNR"])
    raise ValueError(mode)


def _is_better(value: float, best: float, mode: str) -> bool:
    if mode == "warp_psnr":
        return value > best
    return value < best


def train(args):
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    if args.geometry_steps < 1:
        raise ValueError("--geometry_steps must be >=1")
    if args.eval_cases < 1:
        raise ValueError("--eval_cases must be >=1")

    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    train_loader, test_loader, info = build_loaders(cfg)
    base_process = build_progressive_process(cfg)
    p0 = base_process.operator
    srf = torch.as_tensor(
        info["srf_weights"], device=device, dtype=torch.float32
    )

    geometry_model, diffusion_model = _build_frozen_models(args, info, device)
    projector = _build_projector(args, info, device)
    refiner = _build_refiner(args, info, device)

    if args.resume:
        init_from_registered = False
    else:
        init_from_registered = bool(args.init_refiner_checkpoint)
        if init_from_registered:
            _load_refiner_weights(
                refiner,
                args.init_refiner_checkpoint,
                device,
            )

    optimizer = torch.optim.Adam(
        refiner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    best_value = (
        float("-inf") if args.monitor == "warp_psnr" else float("inf")
    )
    if args.resume:
        loaded_epoch, best_value = _load_training_checkpoint(
            refiner,
            args.resume,
            device,
            optimizer=optimizer,
        )
        start_epoch = loaded_epoch + 1
        print(
            f"RESUME epoch={loaded_epoch} monitor={args.monitor} best={best_value:.6f}"
        )

    best_path = _checkpoint_path(args)
    stem, ext = os.path.splitext(best_path)
    last_path = stem + "_last" + ext
    ensure_dir(args.log_root)
    logger = CSVLogger(
        os.path.join(
            args.log_root,
            f"{args.dataset}_innovation3_gigi_cdrdi_{args.variant}.csv",
        ),
        [
            "epoch",
            "loss",
            "l1",
            "ang",
            "phy",
            "msi",
            "registered_psnr",
            "registered_sam",
            "warp_psnr",
            "warp_sam",
            "warp_sam_high",
            "warp_sam_low",
            "dwarp_psnr",
            "dwarp_sam",
            "dwarp_sam_high",
            "best_monitor",
        ],
    )

    sam_fn = SAMLoss()
    train_generator = _make_generator(device, args.seed + 93000)
    print(
        "TRAINING_POLICY geometry=frozen diffusion=frozen terminal_GIGI_only "
        f"geometry_steps={args.geometry_steps} init_registered_GIGI={init_from_registered}"
    )
    print(
        "NONREGISTERED_PROTOCOL "
        f"translation<={args.max_translation}px rotation<={args.max_rotation_deg}deg "
        f"local<={args.max_local_px}px control={args.control_grid}x{args.control_grid} "
        f"identity_probability={args.identity_probability}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        refiner.train()
        loss_sum = l1_sum = ang_sum = phy_sum = msi_sum = 0.0
        count = 0

        for batch in train_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            b, _, h, w = gt.shape

            with torch.no_grad():
                gt_phi = sample_training_geometry(
                    b,
                    h,
                    w,
                    device=device,
                    dtype=gt.dtype,
                    generator=train_generator,
                    args=args,
                )
                true_process = _process_from_geometry(base_process, gt_phi)
                y_h = true_process.terminal_observation(gt)
                pred_rigid, pred_local = _estimate_geometry(
                    geometry_model,
                    y_h=y_h,
                    hr_msi=hr_msi,
                    p0=p0,
                    srf=srf,
                    steps=args.geometry_steps,
                )
                estimated_process = _process_from_estimate(
                    base_process,
                    pred_rigid,
                    pred_local,
                )
                base = reconstruct_from_terminal_lr(
                    diffusion_model,
                    estimated_process,
                    y_h,
                    target_size=(h, w),
                    hr_msi=hr_msi,
                )
                r_phy = _physical_residual(
                    estimated_process,
                    projector,
                    y_h,
                    base,
                )
                heterogeneity = ranked_msi_heterogeneity(hr_msi)

            refined = refiner(base, hr_msi, r_phy, heterogeneity)
            l1 = F.l1_loss(refined, gt)
            ang = sam_fn(refined, gt)
            phy = F.l1_loss(
                estimated_process.terminal_observation(refined),
                y_h,
            )
            msi = F.l1_loss(_hsi_to_msi(refined, srf), hr_msi)
            loss = (
                args.lambda_l1 * l1
                + args.lambda_ang * ang
                + args.lambda_phy * phy
                + args.lambda_msi * msi
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite GIGI-CDRDI loss")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    refiner.parameters(),
                    args.grad_clip,
                    error_if_nonfinite=True,
                )
            optimizer.step()

            loss_sum += float(loss.item()) * b
            l1_sum += float(l1.item()) * b
            ang_sum += float(ang.item()) * b
            phy_sum += float(phy.item()) * b
            msi_sum += float(msi.item()) * b
            count += b

        row = {
            "epoch": epoch,
            "loss": loss_sum / max(count, 1),
            "l1": l1_sum / max(count, 1),
            "ang": ang_sum / max(count, 1),
            "phy": phy_sum / max(count, 1),
            "msi": msi_sum / max(count, 1),
            "registered_psnr": "",
            "registered_sam": "",
            "warp_psnr": "",
            "warp_sam": "",
            "warp_sam_high": "",
            "warp_sam_low": "",
            "dwarp_psnr": "",
            "dwarp_sam": "",
            "dwarp_sam_high": "",
            "best_monitor": best_value,
        }
        print(
            f"epoch={epoch:03d} loss={row['loss']:.7f} "
            f"l1={row['l1']:.7f} ang={row['ang']:.7f} "
            f"phy={row['phy']:.7f} msi={row['msi']:.7f}"
        )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            metrics, mechanism, geom_epe = evaluate(
                refiner,
                diffusion_model,
                geometry_model,
                projector,
                test_loader,
                base_process=base_process,
                p0=p0,
                srf=srf,
                args=args,
                device=device,
            )
            _print_eval(metrics, mechanism, geom_epe)
            wb = metrics["warp_base"]
            wr = metrics["warp_refined"]
            rr = metrics["registered_refined"]
            row.update(
                {
                    "registered_psnr": rr["PSNR"],
                    "registered_sam": rr["SAM"],
                    "warp_psnr": wr["PSNR"],
                    "warp_sam": wr["SAM"],
                    "warp_sam_high": wr["SAM_HIGH"],
                    "warp_sam_low": wr["SAM_LOW"],
                    "dwarp_psnr": wr["PSNR"] - wb["PSNR"],
                    "dwarp_sam": wr["SAM"] - wb["SAM"],
                    "dwarp_sam_high": wr["SAM_HIGH"] - wb["SAM_HIGH"],
                }
            )

            value = _monitor_value(metrics, args.monitor)
            if _is_better(value, best_value, args.monitor):
                best_value = value
                _save_checkpoint(
                    refiner,
                    optimizer,
                    epoch,
                    best_value,
                    best_path,
                    args,
                )
                print(
                    f"SAVED_BEST {best_path} monitor={args.monitor} value={best_value:.6f}"
                )
            row["best_monitor"] = best_value

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            _save_checkpoint(
                refiner,
                optimizer,
                epoch,
                best_value,
                last_path,
                args,
            )
        logger.write(row)


def test(args):
    if args.eval_cases < 1:
        raise ValueError("--eval_cases must be >=1")
    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    _, test_loader, info = build_loaders(cfg)
    base_process = build_progressive_process(cfg)
    p0 = base_process.operator
    srf = torch.as_tensor(
        info["srf_weights"], device=device, dtype=torch.float32
    )

    geometry_model, diffusion_model = _build_frozen_models(args, info, device)
    projector = _build_projector(args, info, device)
    refiner = _build_refiner(args, info, device)

    path = args.refiner_checkpoint or _checkpoint_path(args)
    epoch, best = _load_training_checkpoint(refiner, path, device)
    print(
        f"REFINER_LOAD path={path} epoch={epoch} monitor={args.monitor} best={best:.6f}"
    )

    metrics, mechanism, geom_epe = evaluate(
        refiner,
        diffusion_model,
        geometry_model,
        projector,
        test_loader,
        base_process=base_process,
        p0=p0,
        srf=srf,
        args=args,
        device=device,
    )
    _print_eval(metrics, mechanism, geom_epe)

    print(
        "FINAL_REGISTERED "
        + " ".join(
            f"{k}={metrics['registered_refined'][k]:.6f}" for k in METRIC_KEYS
        )
    )
    print(
        "FINAL_WARP "
        + " ".join(
            f"{k}={metrics['warp_refined'][k]:.6f}" for k in METRIC_KEYS
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
