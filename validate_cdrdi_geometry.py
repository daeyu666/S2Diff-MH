"""Stage-0 CDRDI experiment: can fixed P0/R0 identify phi without flow GT?"""

from __future__ import annotations

import argparse
from statistics import mean

import torch

from cdrdi_geometry import (
    forward_warp,
    geometry_metrics,
    multiscale_closure_loss,
    sample_synthetic_geometry,
    solve_geometry_from_closure,
    spectral_project,
)
from config import TrainConfig
from data_loader import build_loaders
from degradations.physical import PhysicalDegradation
from utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Validate deformation-only physical closure")
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=3)

    p.add_argument("--max_translation", type=float, default=4.0, help="HR-pixel rigid translation bound")
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=2.0, help="HR-pixel local deformation amplitude")
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)

    p.add_argument("--lambda_def", type=float, default=1e-3)
    p.add_argument("--coarse_iterations", type=int, default=300)
    p.add_argument("--middle_iterations", type=int, default=400)
    p.add_argument("--fine_iterations", type=int, default=600)

    p.add_argument("--pass_epe_hr", type=float, default=0.5)
    p.add_argument("--pass_p95_hr", type=float, default=1.0)
    p.add_argument("--pass_closure_reduction", type=float, default=0.90)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cases < 1:
        raise ValueError("--cases must be >= 1")
    set_seed(args.seed)
    device = get_device(args.device)

    cfg = TrainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        test_size=args.test_size,
        scale_ratio=args.scale_ratio,
        srf_interp=args.srf_interp,
        batch_size=1,
        num_workers=0,
    )
    _, test_loader, info = build_loaders(cfg)
    batch = next(iter(test_loader))
    gt_hr_hsi = batch["gt"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=gt_hr_hsi.dtype)

    p0 = PhysicalDegradation(
        scale_ratio=args.scale_ratio,
        mtf_nyquist=args.mtf_nyquist,
        truncate=args.psf_truncate,
    ).to(device)

    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed + 1000)

    all_metrics = []
    for case_idx in range(args.cases):
        phi_gt = sample_synthetic_geometry(
            gt_hr_hsi.shape[-2],
            gt_hr_hsi.shape[-1],
            device=device,
            dtype=gt_hr_hsi.dtype,
            generator=generator,
            max_translation=args.max_translation,
            max_rotation_deg=args.max_rotation_deg,
            max_local_px=args.max_local_px,
            control_grid=args.control_grid,
            min_jacobian=args.min_jacobian,
        )

        with torch.no_grad():
            y_h = p0.degrade(
                forward_warp(
                    gt_hr_hsi,
                    phi_gt.dx,
                    phi_gt.dy,
                    phi_gt.theta_deg,
                    phi_gt.local_field,
                )
            )
            z_h = spectral_project(y_h, srf)

            z_m_gt = p0.degrade(
                forward_warp(
                    hr_msi,
                    phi_gt.dx,
                    phi_gt.dy,
                    phi_gt.theta_deg,
                    phi_gt.local_field,
                )
            )
            gt_closure_mae = float((z_h - z_m_gt).abs().mean().item())
            gt_closure_pyramid = float(multiscale_closure_loss(z_h, z_m_gt, (4, 2, 1)).item())

        # phi_gt is deliberately not passed into the solver. It is used only for evaluation below.
        result = solve_geometry_from_closure(
            hr_msi=hr_msi,
            target_lr_msi=z_h,
            spatial_operator=p0,
            max_translation=args.max_translation,
            max_rotation_deg=args.max_rotation_deg,
            max_local_px=args.max_local_px,
            lambda_def=args.lambda_def,
            coarse_iterations=args.coarse_iterations,
            middle_iterations=args.middle_iterations,
            fine_iterations=args.fine_iterations,
        )
        metrics = geometry_metrics(result, phi_gt, scale_ratio=args.scale_ratio)
        metrics["gt_closure_mae"] = gt_closure_mae
        metrics["gt_closure_pyramid"] = gt_closure_pyramid
        all_metrics.append(metrics)

        print("=" * 96)
        print(f"case={case_idx + 1}/{args.cases} dataset={args.dataset} HR={tuple(gt_hr_hsi.shape[-2:])}")
        print(
            "GT rigid:  "
            f"dx={metrics['dx_gt']:+.4f}px dy={metrics['dy_gt']:+.4f}px "
            f"theta={metrics['theta_gt']:+.4f}deg"
        )
        print(
            "EST rigid: "
            f"dx={metrics['dx_est']:+.4f}px dy={metrics['dy_est']:+.4f}px "
            f"theta={metrics['theta_est']:+.4f}deg"
        )
        print(
            "phi error:  "
            f"EPE_HR_mean={metrics['epe_hr_mean']:.6f}px "
            f"EPE_HR_p95={metrics['epe_hr_p95']:.6f}px "
            f"EPE_LR_mean={metrics['epe_lr_mean']:.6f}px"
        )
        print(
            "closure:    "
            f"initial={metrics['closure_initial']:.8f} "
            f"final={metrics['closure_final']:.8f} "
            f"reduction={100.0 * metrics['closure_reduction']:.3f}% "
            f"GT_commute_MAE={metrics['gt_closure_mae']:.3e}"
        )
        print(
            "Jacobian:   "
            f"GT_min={metrics['gt_min_jacobian']:.6f} "
            f"EST_min={metrics['pred_min_jacobian']:.6f}"
        )
        print("stage closure:", " ".join(f"{k}={v:.8f}" for k, v in result.stage_losses.items()))

    avg_epe = mean(m["epe_hr_mean"] for m in all_metrics)
    avg_p95 = mean(m["epe_hr_p95"] for m in all_metrics)
    avg_reduction = mean(m["closure_reduction"] for m in all_metrics)
    avg_commute = mean(m["gt_closure_mae"] for m in all_metrics)
    min_pred_jac = min(m["pred_min_jacobian"] for m in all_metrics)
    passed = (
        avg_epe <= args.pass_epe_hr
        and avg_p95 <= args.pass_p95_hr
        and avg_reduction >= args.pass_closure_reduction
        and min_pred_jac > 0.0
    )

    print("=" * 96)
    print(
        f"SUMMARY cases={args.cases} AVG_EPE_HR={avg_epe:.6f}px "
        f"AVG_P95_HR={avg_p95:.6f}px AVG_CLOSURE_REDUCTION={100.0 * avg_reduction:.3f}% "
        f"AVG_GT_COMMUTE_MAE={avg_commute:.3e} MIN_PRED_JAC={min_pred_jac:.6f}"
    )
    print("CDRDI_GEOMETRY_SANITY=" + ("PASS" if passed else "FAIL"))
    print(
        "PASS is only an identifiability sanity check. The solver is direct differentiable optimization, "
        "not the final learned recursive deformation network."
    )


if __name__ == "__main__":
    main()
