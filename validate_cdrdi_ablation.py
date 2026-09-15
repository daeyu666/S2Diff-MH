"""Ablations for Stage-0 CDRDI geometry identifiability.

All variants use the same fixed P0/R0 observation model and the same synthetic
rigid + B-spline deformation generator. Flow GT is never passed to the solver;
it is used only after optimization for EPE/Jacobian evaluation.

Variants:
  rigid_only   : 1300 steps, rigid parameters only, multiscale closure.
  direct_fine  : 300 rigid + 1000 direct 5x5 local steps, no coarse local stage.
  single_scale : full rigid->coarse->fine recursion, but every stage uses Q1 only.
  full         : 300/400/600 recursive stages with Q4 -> (Q4,Q2) -> (Q4,Q2,Q1).
"""

from __future__ import annotations

import argparse
from statistics import mean
from typing import Dict, Sequence

import torch

from cdrdi_geometry import (
    GeometrySolveResult,
    RecursiveGeometryState,
    _stage_optimize,
    forward_warp,
    geometry_metrics,
    multiscale_closure_loss,
    sample_synthetic_geometry,
    sampling_coordinates,
    spectral_project,
)
from config import TrainConfig
from data_loader import build_loaders
from degradations.physical import PhysicalDegradation
from utils import get_device, set_seed


VARIANTS = ("rigid_only", "direct_fine", "single_scale", "full")


def parse_args():
    p = argparse.ArgumentParser(description="CDRDI recursive/multiscale geometry ablations")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)

    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)
    p.add_argument("--lambda_def", type=float, default=1e-3)

    # Full CDRDI uses 300+400+600=1300 updates. The direct-fine and rigid-only
    # baselines are budget-matched to the same total count.
    p.add_argument("--rigid_steps", type=int, default=300)
    p.add_argument("--coarse_steps", type=int, default=400)
    p.add_argument("--fine_steps", type=int, default=600)
    p.add_argument("--total_steps", type=int, default=1300)
    return p.parse_args()


def _opt(
    state: RecursiveGeometryState,
    *,
    hr_msi: torch.Tensor,
    target: torch.Tensor,
    p0,
    factors: Sequence[int],
    params,
    steps: int,
    lr: float,
    lambda_def: float,
    use_coarse: bool,
    use_fine: bool,
) -> float:
    return _stage_optimize(
        state,
        hr_msi=hr_msi,
        target_lr_msi=target,
        spatial_operator=p0,
        factors=factors,
        parameters=params,
        iterations=steps,
        lr=lr,
        lambda_def=lambda_def,
        use_coarse=use_coarse,
        use_fine=use_fine,
    )


def solve_variant(args, hr_msi: torch.Tensor, target: torch.Tensor, p0) -> GeometrySolveResult:
    state = RecursiveGeometryState(
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px,
    ).to(device=hr_msi.device, dtype=hr_msi.dtype)

    h, w = hr_msi.shape[-2:]
    zero = torch.zeros(1, device=hr_msi.device, dtype=hr_msi.dtype)
    zero_local = torch.zeros((1, 2, h, w), device=hr_msi.device, dtype=hr_msi.dtype)
    with torch.no_grad():
        initial_pred = p0.degrade(forward_warp(hr_msi, zero, zero, zero, zero_local))
        initial = float(multiscale_closure_loss(target, initial_pred, (4, 2, 1)).item())

    losses: Dict[str, float] = {}
    variant = args.variant

    if variant == "rigid_only":
        losses["rigid_only"] = _opt(
            state,
            hr_msi=hr_msi,
            target=target,
            p0=p0,
            factors=(4, 2, 1),
            params=(state.rigid,),
            steps=args.total_steps,
            lr=0.10,
            lambda_def=0.0,
            use_coarse=False,
            use_fine=False,
        )
        use_coarse, use_fine = False, False

    elif variant == "direct_fine":
        losses["rigid"] = _opt(
            state,
            hr_msi=hr_msi,
            target=target,
            p0=p0,
            factors=(4,),
            params=(state.rigid,),
            steps=args.rigid_steps,
            lr=0.10,
            lambda_def=0.0,
            use_coarse=False,
            use_fine=False,
        )
        remaining = max(0, args.total_steps - args.rigid_steps)
        losses["direct_fine"] = _opt(
            state,
            hr_msi=hr_msi,
            target=target,
            p0=p0,
            factors=(4, 2, 1),
            params=(state.control_fine,),
            steps=remaining,
            lr=0.03,
            lambda_def=args.lambda_def,
            use_coarse=False,
            use_fine=True,
        )
        use_coarse, use_fine = False, True

    elif variant == "single_scale":
        losses["rigid"] = _opt(
            state,
            hr_msi=hr_msi,
            target=target,
            p0=p0,
            factors=(1,),
            params=(state.rigid,),
            steps=args.rigid_steps,
            lr=0.10,
            lambda_def=0.0,
            use_coarse=False,
            use_fine=False,
        )
        losses["local_coarse"] = _opt(
            state,
            hr_msi=hr_msi,
            target=target,
            p0=p0,
            factors=(1,),
            params=(state.control_coarse,),
            steps=args.coarse_steps,
            lr=0.05,
            lambda_def=args.lambda_def,
            use_coarse=True,
            use_fine=False,
        )
        losses["local_fine"] = _opt(
            state,
            hr_msi=hr_msi,
            target=target,
            p0=p0,
            factors=(1,),
            params=(state.control_coarse, state.control_fine),
            steps=args.fine_steps,
            lr=0.03,
            lambda_def=args.lambda_def,
            use_coarse=True,
            use_fine=True,
        )
        use_coarse, use_fine = True, True

    elif variant == "full":
        losses["rigid"] = _opt(
            state,
            hr_msi=hr_msi,
            target=target,
            p0=p0,
            factors=(4,),
            params=(state.rigid,),
            steps=args.rigid_steps,
            lr=0.10,
            lambda_def=0.0,
            use_coarse=False,
            use_fine=False,
        )
        losses["local_coarse"] = _opt(
            state,
            hr_msi=hr_msi,
            target=target,
            p0=p0,
            factors=(4, 2),
            params=(state.control_coarse,),
            steps=args.coarse_steps,
            lr=0.05,
            lambda_def=args.lambda_def,
            use_coarse=True,
            use_fine=False,
        )
        losses["local_fine"] = _opt(
            state,
            hr_msi=hr_msi,
            target=target,
            p0=p0,
            factors=(4, 2, 1),
            params=(state.control_coarse, state.control_fine),
            steps=args.fine_steps,
            lr=0.03,
            lambda_def=args.lambda_def,
            use_coarse=True,
            use_fine=True,
        )
        use_coarse, use_fine = True, True

    else:
        raise ValueError(variant)

    with torch.no_grad():
        local = state.local_field((h, w), use_coarse=use_coarse, use_fine=use_fine)
        dx, dy, theta = state.rigid_tensors()
        pred = p0.degrade(forward_warp(hr_msi, dx, dy, theta, local))
        final = float(multiscale_closure_loss(target, pred, (4, 2, 1)).item())
        sx, sy = sampling_coordinates(h, w, dx, dy, theta, local)

    return GeometrySolveResult(
        dx=float(dx.item()),
        dy=float(dy.item()),
        theta_deg=float(theta.item()),
        closure_initial=initial,
        closure_final=final,
        local_field=local.detach(),
        sampling_x=sx.detach(),
        sampling_y=sy.detach(),
        stage_losses=losses,
    )


def main():
    args = parse_args()
    if args.cases < 1:
        raise ValueError("--cases must be >=1")
    if args.variant in ("single_scale", "full"):
        if args.rigid_steps + args.coarse_steps + args.fine_steps != args.total_steps:
            raise ValueError("recursive step counts must sum to --total_steps for a fair ablation")

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

    rows = []
    for idx in range(args.cases):
        gt_phi = sample_synthetic_geometry(
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
                forward_warp(gt_hr_hsi, gt_phi.dx, gt_phi.dy, gt_phi.theta_deg, gt_phi.local_field)
            )
            z_h = spectral_project(y_h, srf)

        result = solve_variant(args, hr_msi, z_h, p0)
        metrics = geometry_metrics(result, gt_phi, scale_ratio=args.scale_ratio)
        rows.append(metrics)
        print(
            f"case={idx+1:02d}/{args.cases} variant={args.variant} "
            f"EPE={metrics['epe_hr_mean']:.6f}px P95={metrics['epe_hr_p95']:.6f}px "
            f"closure_red={100.0*metrics['closure_reduction']:.3f}% "
            f"min_jac={metrics['pred_min_jacobian']:.6f}"
        )

    avg_epe = mean(v["epe_hr_mean"] for v in rows)
    avg_p95 = mean(v["epe_hr_p95"] for v in rows)
    avg_red = mean(v["closure_reduction"] for v in rows)
    min_jac = min(v["pred_min_jacobian"] for v in rows)
    avg_final = mean(v["closure_final"] for v in rows)
    print("=" * 96)
    print(
        f"ABLATION_SUMMARY variant={args.variant} cases={args.cases} max_local_px={args.max_local_px:.3f} "
        f"AVG_EPE_HR={avg_epe:.6f}px AVG_P95_HR={avg_p95:.6f}px "
        f"AVG_FINAL_CLOSURE={avg_final:.8f} AVG_CLOSURE_REDUCTION={100.0*avg_red:.3f}% "
        f"MIN_PRED_JAC={min_jac:.6f} total_steps={args.total_steps}"
    )


if __name__ == "__main__":
    main()
