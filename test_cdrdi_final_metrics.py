"""Final paper-style test for Innovation-2 CDRDI.

Reports the complete HSI fusion metric set used by this repository:
PSNR, SSIM, ERGAS, SAM, CC, RMSE, together with the training and testing
conditions that produced them.

The test keeps the four Stage-2 paths on exactly the same deterministic
synthetic deformation cases:

    registered : registered physical-diffusion reference
    naive      : deformed LR-HSI, geometry ignored
    oracle     : deformed LR-HSI with GT geometry in A_(t,phi)
    estimated  : deformed LR-HSI with frozen CDRDI estimated geometry

No LR-HSI inverse warp is used.  Metrics are computed with metrics.py exactly
as in the rest of the repository and are averaged over the requested cases.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Any, Dict

import torch

from cdrdi_geometry import sample_synthetic_geometry, sampling_coordinates, spectral_project
from config import TrainConfig
from data_loader import build_loaders
from degradations.deformation_aware import DeformationAwareProgressiveDegradation
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from main import build_model
from metrics import MetricAverager, calc_metrics
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from utils import ensure_dir, get_device, load_checkpoint, set_seed


VARIANTS = ("registered", "naive", "oracle", "estimated")
METRIC_ORDER = ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")


def parse_args():
    p = argparse.ArgumentParser(description="Final CDRDI paper-style full-metric test")
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)
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

    p.add_argument("--geometry_steps", type=int, default=9, help="CDRDI recursive steps used at test time")
    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument(
        "--geometry_checkpoint",
        default="./checkpoints/cdrdi_stage1/PaviaU_recursive_k6_finalonly_300ep_lr1e4.pth",
    )
    # Stage-1 checkpoint predates full args-in-extra logging, so keep the known
    # final-run training conditions explicit and overridable for the report.
    p.add_argument("--geometry_train_epochs", type=int, default=300)
    p.add_argument("--geometry_train_batch_size", type=int, default=4)

    p.add_argument("--diffusion_base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--diffusion_checkpoint",
        default="./checkpoints/cdrdi_stage2/PaviaU_estimated_deform_diffusion_k9_stage2d_A.pth",
    )

    p.add_argument("--print_cases", action="store_true", help="also print every case's full metric set")
    p.add_argument("--output_json", default="./results/cdrdi_final_test_PaviaU_seed10_cases10.json")
    return p.parse_args()


def _load_checkpoint_record(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        return {"epoch": None, "best_metric": None, "extra": {}, "optimizer_lr": None}
    optimizer_lr = None
    optimizer = state.get("optimizer")
    if isinstance(optimizer, dict):
        groups = optimizer.get("param_groups", [])
        if groups:
            optimizer_lr = groups[0].get("lr")
    extra = state.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    return {
        "epoch": state.get("epoch"),
        "best_metric": state.get("best_metric"),
        "extra": extra,
        "optimizer_lr": optimizer_lr,
    }


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _geometry_epe(height, width, gt_phi, pred_rigid, pred_local) -> float:
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


def _format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"PSNR={metrics['PSNR']:.4f} "
        f"SSIM={metrics['SSIM']:.6f} "
        f"ERGAS={metrics['ERGAS']:.4f} "
        f"SAM={metrics['SAM']:.4f} "
        f"CC={metrics['CC']:.6f} "
        f"RMSE={metrics['RMSE']:.6f}"
    )


def _clean_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _checkpoint_training_conditions(args, geometry_record, diffusion_record):
    g_extra = geometry_record["extra"]
    d_extra = diffusion_record["extra"]
    geometry = {
        "checkpoint": args.geometry_checkpoint,
        "dataset": g_extra.get("dataset", args.dataset),
        "variant": g_extra.get("variant", "recursive"),
        "train_recursive_steps": g_extra.get("steps", 6),
        "loss_mode": g_extra.get("loss_mode", "final_only"),
        "configured_epochs": args.geometry_train_epochs,
        "selected_best_epoch": geometry_record["epoch"],
        "selected_by": "minimum observable physical closure (no flow-GT selection)",
        "best_observable_closure": geometry_record["best_metric"],
        "optimizer_lr_at_selected_checkpoint": geometry_record["optimizer_lr"],
        "batch_size": args.geometry_train_batch_size,
        "max_translation_hr_px": g_extra.get("max_translation", args.max_translation),
        "max_rotation_deg": g_extra.get("max_rotation_deg", args.max_rotation_deg),
        "max_local_hr_px": g_extra.get("max_local_px", args.max_local_px),
        "control_grid": g_extra.get("control_grid", args.control_grid),
        "supervision": "fixed-P0/R0 physical closure; synthetic GT geometry only generates observations",
    }
    diffusion = {
        "checkpoint": args.diffusion_checkpoint,
        "dataset": d_extra.get("dataset", args.dataset),
        "stage": "Stage-2D estimated-phi-aware Raw-Direct adaptation",
        "configured_epochs": d_extra.get("epochs"),
        "selected_best_epoch": diffusion_record["epoch"],
        "selected_by": "maximum ESTIMATED PSNR",
        "best_estimated_psnr_recorded": diffusion_record["best_metric"],
        "optimizer_lr_at_selected_checkpoint": diffusion_record["optimizer_lr"],
        "geometry_steps_during_adaptation": d_extra.get("geometry_steps"),
        "geometry_checkpoint_during_adaptation": d_extra.get("geometry_checkpoint"),
        "batch_size": d_extra.get("batch_size"),
        "lambda_l1": d_extra.get("lambda_l1", 1.0),
        "lambda_sam": d_extra.get("lambda_sam", 0.1),
        "init_checkpoint": d_extra.get("init_checkpoint"),
        "trajectory": "observation-anchored estimated-phi deformation-aware trajectory",
        "geometry_solver": "frozen during diffusion adaptation",
    }
    return geometry, diffusion


def _print_condition_block(title: str, data: Dict[str, Any]):
    print(f"{title}_BEGIN")
    for key, value in data.items():
        print(f"  {key}={_clean_value(value)}")
    print(f"{title}_END")


def main():
    args = parse_args()
    if args.cases < 1:
        raise ValueError("--cases must be >=1")
    if args.geometry_steps < 1:
        raise ValueError("--geometry_steps must be >=1")

    set_seed(args.seed)
    device = get_device(args.device)

    geometry_record = _load_checkpoint_record(args.geometry_checkpoint)
    diffusion_record = _load_checkpoint_record(args.diffusion_checkpoint)
    geometry_train, diffusion_train = _checkpoint_training_conditions(
        args, geometry_record, diffusion_record
    )

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
        raise ValueError("final CDRDI test expects the standard single test patch")
    h, w = gt.shape[-2:]
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
    for parameter in geometry_model.parameters():
        parameter.requires_grad_(False)

    diffusion_model = build_model(cfg, info, device)
    load_checkpoint(
        diffusion_model,
        args.diffusion_checkpoint,
        map_location=str(device),
        load_optimizer=False,
    )
    diffusion_model.eval()

    test_conditions = {
        "dataset": args.dataset,
        "test_patch": f"{h}x{w}",
        "metric_region": "full-frame",
        "cases": args.cases,
        "seed": args.seed,
        "synthetic_case_generator_seed": args.seed + 70000,
        "scale_ratio": args.scale_ratio,
        "physical_degradation": "shared calibrated PSF/MTF + detector integration + sampling",
        "mtf_nyquist": args.mtf_nyquist,
        "psf_truncate": args.psf_truncate,
        "srf_interp": args.srf_interp,
        "diffusion_steps": args.diffusion_steps,
        "lift_mode": "normalized_adjoint",
        "test_geometry_steps": args.geometry_steps,
        "max_translation_hr_px": args.max_translation,
        "max_rotation_deg": args.max_rotation_deg,
        "max_local_hr_px": args.max_local_px,
        "local_deformation": f"{args.control_grid}x{args.control_grid} sparse control points -> cubic B-spline dense field",
        "min_synthetic_jacobian": args.min_jacobian,
        "msi_coordinate_system": "reliable HR reference coordinate system",
        "hsi_observation": "warp HR-HSI first, then physical PSF/integration/downsampling",
        "inverse_warp_lr_hsi": False,
        "metric_implementation": "repository metrics.py; per-case metrics averaged over cases",
    }

    print("=" * 120)
    print("CDRDI_FINAL_TEST_PROTOCOL")
    _print_condition_block("GEOMETRY_TRAIN_CONDITIONS", geometry_train)
    _print_condition_block("DIFFUSION_TRAIN_CONDITIONS", diffusion_train)
    _print_condition_block("TEST_CONDITIONS", test_conditions)
    print("=" * 120)

    meters = {name: MetricAverager() for name in VARIANTS}
    geometry_epes = []
    per_case = []
    generator = _make_generator(device, args.seed + 70000)

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
        gt_rigid = torch.stack([gt_phi.dx, gt_phi.dy, gt_phi.theta_deg], dim=1)
        oracle_process = DeformationAwareProgressiveDegradation(
            base_process,
            rigid=gt_rigid,
            local_field=gt_phi.local_field,
        )

        with torch.no_grad():
            y_h = oracle_process.terminal_observation(gt)
            z_h = spectral_project(y_h, srf)
            geometry_out = geometry_model(z_h, hr_msi, p0, steps=args.geometry_steps)
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
                diffusion_model, base_process, y_h, target_size=(h, w), hr_msi=hr_msi
            )
            oracle_pred = reconstruct_from_terminal_lr(
                diffusion_model, oracle_process, y_h, target_size=(h, w), hr_msi=hr_msi
            )
            estimated_pred = reconstruct_from_terminal_lr(
                diffusion_model, estimated_process, y_h, target_size=(h, w), hr_msi=hr_msi
            )

        case_metrics = {
            "registered": registered_metrics,
            "naive": calc_metrics(naive_pred, gt, args.scale_ratio),
            "oracle": calc_metrics(oracle_pred, gt, args.scale_ratio),
            "estimated": calc_metrics(estimated_pred, gt, args.scale_ratio),
        }
        for name in VARIANTS:
            meters[name].update(case_metrics[name])

        record = {
            "case": case_idx + 1,
            "geometry_epe_hr_px": geom_epe,
            "gt_rigid": {
                "dx": float(gt_phi.dx.item()),
                "dy": float(gt_phi.dy.item()),
                "theta_deg": float(gt_phi.theta_deg.item()),
            },
            "estimated_rigid": {
                "dx": float(pred_rigid[0, 0].item()),
                "dy": float(pred_rigid[0, 1].item()),
                "theta_deg": float(pred_rigid[0, 2].item()),
            },
            "metrics": case_metrics,
        }
        per_case.append(record)
        if args.print_cases:
            print("-" * 120)
            print(f"CASE={case_idx+1:02d} GEOM_EPE_HR={geom_epe:.6f}px")
            for name in VARIANTS:
                print(f"CASE_{name.upper():>10s} {_format_metrics(case_metrics[name])}")

    averages = {name: meters[name].average() for name in VARIANTS}
    avg_epe = mean(geometry_epes)

    print("=" * 120)
    print(
        f"FINAL_TEST_SUMMARY dataset={args.dataset} cases={args.cases} seed={args.seed} "
        f"AVG_GEOM_EPE_HR={avg_epe:.6f}px geometry_steps={args.geometry_steps} metric_region=full-frame"
    )
    for name in VARIANTS:
        print(f"FINAL_{name.upper():>10s} {_format_metrics(averages[name])}")

    reg = averages["registered"]["PSNR"]
    naive = averages["naive"]["PSNR"]
    oracle = averages["oracle"]["PSNR"]
    estimated = averages["estimated"]["PSNR"]
    print(
        "FINAL_PSNR_GAPS "
        f"NAIVE_DROP={reg-naive:+.4f}dB "
        f"ORACLE_TO_REGISTERED={oracle-reg:+.4f}dB "
        f"EST_TO_ORACLE={estimated-oracle:+.4f}dB "
        f"EST_TO_REGISTERED={estimated-reg:+.4f}dB "
        f"EST_RECOVERY={estimated-naive:+.4f}dB"
    )

    output = {
        "geometry_training_conditions": geometry_train,
        "diffusion_training_conditions": diffusion_train,
        "test_conditions": test_conditions,
        "average_geometry_epe_hr_px": avg_epe,
        "average_metrics": averages,
        "per_case": per_case,
    }
    if args.output_json:
        ensure_dir(os.path.dirname(args.output_json))
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
        print(f"FINAL_TEST_JSON={args.output_json}")


if __name__ == "__main__":
    main()
