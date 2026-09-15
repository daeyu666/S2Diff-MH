"""Stage-2A algebraic closure diagnostic for deformation-aware diffusion.

Before adapting the Raw-MSI predictor, verify that the deformation-aware
progressive process itself is closed.  We synthesize a terminal HSI observation
with a known geometry phi, lift it with the normalized adjoint, and then run the
deterministic reverse update while supplying the exact clean HR-HSI X as x0_hat.

If the operator implementation is correct, every reverse step must telescope:

    x_(t-1) = x_t + A~_(t-1,phi)(X) - A~_(t,phi)(X)
            = A~_(t-1,phi)(X),

and the final t=0 state must equal X to numerical precision.  No network is
involved in this diagnostic.
"""

from __future__ import annotations

import argparse
from statistics import mean

import torch

from cdrdi_geometry import sample_synthetic_geometry
from config import TrainConfig
from data_loader import build_loaders
from degradations.deformation_aware import DeformationAwareProgressiveDegradation
from innovation1 import build_progressive_process
from metrics import calc_psnr
from utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="CDRDI Stage-2 oracle reverse algebraic closure")
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=3)
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)
    p.add_argument("--pass_max_abs", type=float, default=2e-5)
    return p.parse_args()


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def main():
    args = parse_args()
    if args.cases < 1:
        raise ValueError("--cases must be >=1")

    set_seed(args.seed)
    device = get_device(args.device)
    cfg = TrainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        test_size=args.test_size,
        scale_ratio=args.scale_ratio,
        srf_interp=args.srf_interp,
        degradation_mode="physical",
        diffusion_steps=args.diffusion_steps,
        lift_mode="normalized_adjoint",
        mtf_nyquist=args.mtf_nyquist,
        psf_truncate=args.psf_truncate,
        batch_size=1,
        num_workers=0,
    )
    _, test_loader, _ = build_loaders(cfg)
    gt = next(iter(test_loader))["gt"].to(device)
    if gt.shape[0] != 1:
        raise ValueError("oracle closure diagnostic expects the standard single test patch")

    base = build_progressive_process(cfg)
    generator = _make_generator(device, args.seed + 81000)
    h, w = gt.shape[-2:]

    final_max_abs = []
    final_mean_abs = []
    max_step_abs = []
    terminal_lift_abs = []

    for case_idx in range(args.cases):
        phi = sample_synthetic_geometry(
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
        rigid = torch.stack([phi.dx, phi.dy, phi.theta_deg], dim=1)
        process = DeformationAwareProgressiveDegradation(
            base,
            rigid=rigid,
            local_field=phi.local_field,
        )

        with torch.no_grad():
            y_t = process.terminal_observation(gt)
            x_t = process.terminal_state(y_t, target_size=(h, w))
            exact_terminal = process.state_at(gt, process.total_steps)
            terminal_err = float((x_t - exact_terminal).abs().max().item())

            worst = terminal_err
            for t in range(process.total_steps, 0, -1):
                expected_t = process.state_at(gt, t)
                state_err = float((x_t - expected_t).abs().max().item())
                worst = max(worst, state_err)
                x_t = process.reverse_update(x_t, gt, t)
                expected_prev = process.state_at(gt, t - 1)
                step_err = float((x_t - expected_prev).abs().max().item())
                worst = max(worst, step_err)

            err = (x_t - gt).abs()
            fmax = float(err.max().item())
            fmean = float(err.mean().item())
            psnr = calc_psnr(x_t, gt)

        terminal_lift_abs.append(terminal_err)
        max_step_abs.append(worst)
        final_max_abs.append(fmax)
        final_mean_abs.append(fmean)
        print(
            f"case={case_idx+1:02d}/{args.cases} "
            f"TERMINAL_LIFT_MAX={terminal_err:.3e} "
            f"MAX_TRAJECTORY_ABS={worst:.3e} "
            f"FINAL_MAX_ABS={fmax:.3e} FINAL_MEAN_ABS={fmean:.3e} "
            f"FINAL_PSNR={psnr:.4f}"
        )

    worst_final = max(final_max_abs)
    worst_trajectory = max(max_step_abs)
    passed = worst_final <= float(args.pass_max_abs) and worst_trajectory <= float(args.pass_max_abs)
    print("=" * 104)
    print(
        f"ORACLE_CLOSURE_SUMMARY cases={args.cases} "
        f"AVG_TERMINAL_LIFT_MAX={mean(terminal_lift_abs):.3e} "
        f"MAX_TRAJECTORY_ABS={worst_trajectory:.3e} "
        f"MAX_FINAL_ABS={worst_final:.3e} "
        f"AVG_FINAL_MEAN_ABS={mean(final_mean_abs):.3e} "
        f"PASS_THRESHOLD={args.pass_max_abs:.3e}"
    )
    print("CDRDI_DIFFUSION_ORACLE_CLOSURE=" + ("PASS" if passed else "FAIL"))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
