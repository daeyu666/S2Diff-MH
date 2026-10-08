"""Augsburg-Real D2 validation diagnostics (no training, no test leakage).

Compares:
- bicubic upsampling of observed EnMAP30, no predicted geometry;
- normalized physical-adjoint baseline, identity geometry;
- estimated-geometry physical-adjoint baseline;
- roundtrip of EnMAP10 through normalized warp-adjoint and forward warp;
- Real-D2 prediction in its native MSI frame and mapped EnMAP reference frame;
- MSI-to-HSI reference spectral correspondence before/after estimated warp.

Metrics are aggregated from summed squared errors and summed pixel SAM;
validation edge masks are honored and padded pixels never enter metrics.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from augsburg_real import build_augsburg_real_loaders
from augsburg_real_process import build_augsburg_real_process
from cdrdi_geometry import forward_warp, spectral_project
from innovation1 import reconstruct_from_terminal_lr
from main import build_model
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from train_augsburg_real_diffusion import (
    _apply_radiometry,
    _config,
    _estimated_process,
    _estimate_geometry,
    _load_json,
    _lr_mask,
    _masked_l1,
    _masked_metric_sums,
    _metrics_from_sums,
    _radiometry,
    _warp_adjoint_normalized,
)
from utils import get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose Augsburg-Real D2 validation gap")
    p.add_argument("--cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--psf_json", default="./data/calibration/AugsburgReal_effective_psf.json")
    p.add_argument("--radiometry_json", default="./data/calibration/AugsburgReal_radiometry.json")
    p.add_argument("--geometry_checkpoint", required=True)
    p.add_argument("--diffusion_checkpoint", required=True)
    p.add_argument("--eval_patch_size", type=int, default=192)
    p.add_argument("--train_patch_size", type=int, default=96)
    p.add_argument("--train_stride", type=int, default=48)
    p.add_argument("--min_valid_fraction", type=float, default=0.80)
    p.add_argument("--geometry_steps", type=int, default=9)
    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--check_identity_diffusion", action="store_true")
    p.add_argument("--max_tiles", type=int, default=0, help="0 means full validation")
    return p.parse_args()


class MetricCounter:
    def __init__(self):
        self.sse = 0.0
        self.n = 0
        self.sam_sum = 0.0
        self.sam_n = 0

    def update(self, pred, target, mask):
        sse, n, sam_sum, sam_n = _masked_metric_sums(pred, target, mask)
        self.sse += sse
        self.n += n
        self.sam_sum += sam_sum
        self.sam_n += sam_n

    def result(self):
        return _metrics_from_sums(self.sse, self.n, self.sam_sum, self.sam_n)


def _masked_pearson(x, y, mask):
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.expand(-1, x.shape[1], -1, -1)
    xv = x[mask].double()
    yv = y[mask].double()
    if xv.numel() < 2:
        return float("nan")
    xc = xv - xv.mean()
    yc = yv - yv.mean()
    den = torch.sqrt((xc * xc).sum() * (yc * yc).sum())
    return float(((xc * yc).sum() / den.clamp_min(1e-12)).item())


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    sigma = float(_load_json(args.psf_json)["terminal_sigma_hr_pixels"])
    radiometry = _radiometry(args.radiometry_json, device)
    _, loader, _, info = build_augsburg_real_loaders(
        args.cache_root,
        train_patch_size=args.train_patch_size,
        train_stride=args.train_stride,
        eval_patch_size=args.eval_patch_size,
        min_valid_fraction=args.min_valid_fraction,
        batch_size=1,
        num_workers=0,
    )
    base_process = build_augsburg_real_process(
        effective_sigma=sigma, diffusion_steps=args.diffusion_steps
    )
    p0 = base_process.operator.to(device)
    srf = torch.as_tensor(info["srf_weights"], dtype=torch.float32, device=device)
    geometry = LearnedPhysicalResidualSolver(
        4,
        base_channels=args.geometry_base_channels,
        control_grid=args.control_grid,
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px,
    ).to(device)
    load_checkpoint(geometry, args.geometry_checkpoint, map_location=str(device), load_optimizer=False)
    geometry.eval()
    diffusion = build_model(_config(args), info, device)
    load_checkpoint(diffusion, args.diffusion_checkpoint, map_location=str(device), load_optimizer=False)
    diffusion.eval()

    keys = (
        "LR_BICUBIC",
        "LR_NORMALIZED_ADJOINT_IDENTITY",
        "LR_NORMALIZED_ADJOINT_GEOMETRY",
        "GEOMETRY_REFERENCE_ROUNDTRIP",
        "D2_NATIVE_NO_WARP",
        "D2_GEOMETRY_TO_REFERENCE",
    )
    if args.check_identity_diffusion:
        keys += ("D2_IDENTITY_NO_GEOMETRY",)
    counters = {key: MetricCounter() for key in keys}

    sum_corr_before = 0.0
    sum_corr_after = 0.0
    sum_spatial_count = 0.0
    sum_proj_before = 0.0
    sum_proj_after = 0.0
    tiles = 0

    for batch in loader:
        if args.max_tiles and tiles >= args.max_tiles:
            break
        gt = batch["gt"].to(device)
        y_h = batch["lr_hsi"].to(device)
        y_m = _apply_radiometry(batch["hr_msi"].to(device), radiometry)
        mask = batch["valid_mask"].to(device) > 0.5
        h, w = gt.shape[-2:]
        rigid, local = _estimate_geometry(
            geometry, y_h, y_m, p0=p0, srf=srf, steps=args.geometry_steps
        )
        process = _estimated_process(base_process, rigid, local)

        bicubic = F.interpolate(
            y_h, size=(h, w), mode="bicubic", align_corners=False
        )
        adj_id = base_process.terminal_state(y_h, target_size=(h, w))
        adj_geo_msi = process.terminal_state(y_h, target_size=(h, w))
        adj_geo_ref = forward_warp(
            adj_geo_msi, rigid[:, 0], rigid[:, 1], rigid[:, 2], local
        )

        target_msi = _warp_adjoint_normalized(gt, rigid, local)
        target_roundtrip = forward_warp(
            target_msi, rigid[:, 0], rigid[:, 1], rigid[:, 2], local
        )

        pred_msi = reconstruct_from_terminal_lr(
            diffusion,
            process,
            y_h,
            target_size=(h, w),
            hr_msi=y_m,
        )
        pred_ref = forward_warp(
            pred_msi, rigid[:, 0], rigid[:, 1], rigid[:, 2], local
        )
        to_check = {
            "LR_BICUBIC": bicubic,
            "LR_NORMALIZED_ADJOINT_IDENTITY": adj_id,
            "LR_NORMALIZED_ADJOINT_GEOMETRY": adj_geo_ref,
            "GEOMETRY_REFERENCE_ROUNDTRIP": target_roundtrip,
            "D2_NATIVE_NO_WARP": pred_msi,
            "D2_GEOMETRY_TO_REFERENCE": pred_ref,
        }
        if args.check_identity_diffusion:
            to_check["D2_IDENTITY_NO_GEOMETRY"] = reconstruct_from_terminal_lr(
                diffusion,
                base_process,
                y_h,
                target_size=(h, w),
                hr_msi=y_m,
            )

        for key, result in to_check.items():
            counters[key].update(result, gt, mask)

        target_msi4 = spectral_project(gt, srf)
        projected_real_before = y_m
        projected_real_after = forward_warp(
            y_m, rigid[:, 0], rigid[:, 1], rigid[:, 2], local
        )
        corr_before = _masked_pearson(
            projected_real_before, target_msi4, mask
        )
        corr_after = _masked_pearson(
            projected_real_after, target_msi4, mask
        )
        l1_before = float(
            _masked_l1(projected_real_before, target_msi4, mask).item()
        )
        l1_after = float(
            _masked_l1(projected_real_after, target_msi4, mask).item()
        )
        weight = float(mask[:, 0].sum().item())
        sum_spatial_count += weight
        sum_corr_before += corr_before * weight
        sum_corr_after += corr_after * weight
        sum_proj_before += l1_before * weight
        sum_proj_after += l1_after * weight

        print(
            f"VAL_TILE={tiles + 1} HR={h}x{w} "
            f"LR={tuple(y_h.shape[-2:])} "
            f"BICUBIC_PSNR={counters['LR_BICUBIC'].result()[0]:.4f} "
            f"D2_REF_PSNR={counters['D2_GEOMETRY_TO_REFERENCE'].result()[0]:.4f} "
            f"CORR_S2_BEFORE={corr_before:.5f} "
            f"CORR_S2_AFTER={corr_after:.5f}"
        )
        tiles += 1

    print(f"DIAGNOSTIC_SUMMARY validation_tiles={tiles} sigma={sigma:.6f}")
    for key, counter in counters.items():
        psnr, sam = counter.result()
        print(f"{key}: REF_PSNR={psnr:.6f} REF_SAM={sam:.6f}")
    if sum_spatial_count:
        print(
            "S2_TO_ENMAP10_REFERENCE "
            f"CORR_BEFORE={sum_corr_before/sum_spatial_count:.6f} "
            f"CORR_AFTER={sum_corr_after/sum_spatial_count:.6f} "
            f"L1_BEFORE={sum_proj_before/sum_spatial_count:.8f} "
            f"L1_AFTER={sum_proj_after/sum_spatial_count:.8f}"
        )
    print(
        "NOTE: GEOMETRY_REFERENCE_ROUNDTRIP measures the estimated "
        "warp-adjoint/reference mapping distortion; it is a diagnostic, "
        "not an upper bound on any optimally trained reconstruction."
    )


if __name__ == "__main__":
    main()
