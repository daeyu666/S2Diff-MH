"""Stage-2 diagnostic: couple learned CDRDI geometry to Innovation-1 diffusion.

This script deliberately does not retrain the HSI super-resolution predictor.
It asks a narrower question first: if the acquisition geometry is inserted into
Innovation-1 through the forward operator A_t=D_t o W_phi and its normalized
adjoint, can the existing registered Raw-Direct checkpoint recover performance
under synthetic misregistration?

Four paths are compared on exactly the same HR-HSI/MSI test patch:

  registered : no geometric deformation, original Innovation-1 process;
  naive      : deformed LR-HSI, but the original process ignores geometry;
  oracle     : deformed LR-HSI + GT phi in the deformation-aware process;
  estimated  : deformed LR-HSI + learned CDRDI phi in that same process.

No LR-HSI inverse warp is used in either oracle or estimated paths.
"""

from __future__ import annotations

import argparse
from statistics import mean

import torch

from cdrdi_geometry import (
    forward_warp,
    sample_synthetic_geometry,
    sampling_coordinates,
    spectral_project,
)
from config import TrainConfig
from data_loader import build_loaders
from degradations.deformation_aware import DeformationAwareProgressiveDegradation
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from main import build_model
from metrics import MetricAverager, calc_metrics
from models import load_legacy_raw_direct_checkpoint
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from utils import get_device, load_checkpoint, set_seed


VARIANTS = ("registered", "naive", "oracle", "estimated")


def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 CDRDI x physical-diffusion coupling diagnostic")
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

    p.add_argument("--geometry_steps", type=int, default=4)
    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument(
        "--geometry_checkpoint",
        default="./checkpoints/cdrdi_stage1/PaviaU_recursive_k4_local4_seed10.pth",
    )

    p.add_argument("--diffusion_base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--diffusion_checkpoint", default="")
    p.add_argument(
        "--legacy_raw_direct_checkpoint",
        default="./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth",
    )
    return p.parse_args()


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _geometry_epe(
    height: int,
    width: int,
    gt_phi,
    pred_rigid: torch.Tensor,
    pred_local: torch.Tensor,
) -> float:
    gt_x, gt_y = sampling_coordinates(
        height,
        width,
        gt_phi.dx,
        gt_phi.dy,
        gt_phi.theta_deg,
        gt_phi.local_field,
    )
    pred_x, pred_y = sampling_coordinates(
        height,
        width,
        pred_rigid[:, 0],
        pred_rigid[:, 1],
        pred_rigid[:, 2],
        pred_local,
    )
    epe = torch.sqrt((pred_x - gt_x).pow(2) + (pred_y - gt_y).pow(2))
    return float(epe.mean().item())


def _format_short(metrics):
    return (
        f"PSNR={metrics['PSNR']:.4f} SAM={metrics['SAM']:.4f} "
        f"RMSE={metrics['RMSE']:.6f} SSIM={metrics['SSIM']:.6f}"
    )


def main():
    args = parse_args()
    if args.cases < 1:
        raise ValueError("--cases must be >=1")
    if args.geometry_steps < 1:
        raise ValueError("--geometry_steps must be >=1")

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
        predictor="raw_direct",
        base_channels=args.diffusion_base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        spectral_hidden=args.spectral_hidden,
        batch_size=1,
        num_workers=0,
    )
    _, test_loader, info = build_loaders(cfg)
    batch = next(iter(test_loader))
    gt = batch["gt"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    if gt.shape[0] != 1:
        raise ValueError("Stage-2 diagnostic expects the standard single test patch")
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=gt.dtype)

    base_process = build_progressive_process(cfg)
    p0 = base_process.operator

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

    diffusion_model = build_model(cfg, info, device)
    if args.diffusion_checkpoint:
        load_checkpoint(
            diffusion_model,
            args.diffusion_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        print("Loaded diffusion checkpoint:", args.diffusion_checkpoint)
    else:
        report = load_legacy_raw_direct_checkpoint(
            diffusion_model,
            args.legacy_raw_direct_checkpoint,
            map_location=str(device),
        )
        print("Loaded legacy Raw-Direct checkpoint:", report)
    diffusion_model.eval()

    meters = {name: MetricAverager() for name in VARIANTS}
    geometry_epes = []
    generator = _make_generator(device, args.seed + 70000)
    h, w = gt.shape[-2:]

    # Registered reference is independent of the synthetic deformation case.
    with torch.no_grad():
        registered_lr = base_process.terminal_observation(gt)
        registered_pred = reconstruct_from_terminal_lr(
            diffusion_model,
            base_process,
            registered_lr,
            target_size=(h, w),
            hr_msi=hr_msi,
        )
        registered_metrics = calc_metrics(registered_pred, gt, args.scale_ratio)

    for case_idx in range(args.cases):
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
        gt_rigid = torch.stack(
            [gt_phi.dx, gt_phi.dy, gt_phi.theta_deg], dim=1
        )
        oracle_process = DeformationAwareProgressiveDegradation(
            base_process,
            rigid=gt_rigid,
            local_field=gt_phi.local_field,
        )

        with torch.no_grad():
            # Real HSI observation is synthesized only in the acquisition frame.
            y_h = oracle_process.terminal_observation(gt)
            z_h = spectral_project(y_h, srf)
            geometry_out = geometry_model(
                z_h,
                hr_msi,
                p0,
                steps=args.geometry_steps,
            )
            pred_rigid = geometry_out["final_rigid"].detach()
            pred_local = geometry_out["final_local_field"].detach()
            geom_epe = _geometry_epe(h, w, gt_phi, pred_rigid, pred_local)
            geometry_epes.append(geom_epe)

            estimated_process = DeformationAwareProgressiveDegradation(
                base_process,
                rigid=pred_rigid,
                local_field=pred_local,
            )

            naive_pred = reconstruct_from_terminal_lr(
                diffusion_model,
                base_process,
                y_h,
                target_size=(h, w),
                hr_msi=hr_msi,
            )
            oracle_pred = reconstruct_from_terminal_lr(
                diffusion_model,
                oracle_process,
                y_h,
                target_size=(h, w),
                hr_msi=hr_msi,
            )
            estimated_pred = reconstruct_from_terminal_lr(
                diffusion_model,
                estimated_process,
                y_h,
                target_size=(h, w),
                hr_msi=hr_msi,
            )

        case_metrics = {
            "registered": registered_metrics,
            "naive": calc_metrics(naive_pred, gt, args.scale_ratio),
            "oracle": calc_metrics(oracle_pred, gt, args.scale_ratio),
            "estimated": calc_metrics(estimated_pred, gt, args.scale_ratio),
        }
        for name, metrics in case_metrics.items():
            meters[name].update(metrics)

        print("=" * 104)
        print(
            f"case={case_idx+1:02d}/{args.cases} GEOM_EPE_HR={geom_epe:.6f}px "
            f"GT_rigid=({float(gt_phi.dx.item()):+.3f},{float(gt_phi.dy.item()):+.3f},"
            f"{float(gt_phi.theta_deg.item()):+.3f}deg) "
            f"EST_rigid=({float(pred_rigid[0,0].item()):+.3f},"
            f"{float(pred_rigid[0,1].item()):+.3f},"
            f"{float(pred_rigid[0,2].item()):+.3f}deg)"
        )
        for name in VARIANTS:
            print(f"{name:>10s}: {_format_short(case_metrics[name])}")

    averages = {name: meters[name].average() for name in VARIANTS}
    print("=" * 104)
    print(
        f"STAGE2_SUMMARY cases={args.cases} AVG_GEOM_EPE_HR={mean(geometry_epes):.6f}px "
        f"geometry_steps={args.geometry_steps}"
    )
    for name in VARIANTS:
        print(f"STAGE2_{name.upper():>10s} {_format_short(averages[name])}")

    reg_psnr = averages["registered"]["PSNR"]
    naive_psnr = averages["naive"]["PSNR"]
    oracle_psnr = averages["oracle"]["PSNR"]
    est_psnr = averages["estimated"]["PSNR"]
    print(
        "STAGE2_GAPS "
        f"NAIVE_DROP={reg_psnr-naive_psnr:+.4f}dB "
        f"ORACLE_RECOVERY={oracle_psnr-naive_psnr:+.4f}dB "
        f"ORACLE_TO_REGISTERED={oracle_psnr-reg_psnr:+.4f}dB "
        f"EST_TO_ORACLE={est_psnr-oracle_psnr:+.4f}dB "
        f"EST_RECOVERY={est_psnr-naive_psnr:+.4f}dB"
    )
    print(
        "Interpretation: if ORACLE approaches REGISTERED, the acquisition-operator coupling works with the "
        "existing diffusion checkpoint. If ORACLE stays far below REGISTERED, the next step is diffusion "
        "fine-tuning on A~_(t,phi) states rather than changing the geometry solver. The EST-vs-ORACLE gap "
        "isolates the remaining geometry-estimation bottleneck."
    )


if __name__ == "__main__":
    main()
