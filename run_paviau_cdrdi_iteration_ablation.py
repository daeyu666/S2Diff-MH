"""PaviaU unregistered CDRDI one-shot-vs-recursive ablation.

One-shot is an independently trained single-update geometry estimator.
K=3/K=6/K=9 are inference-depth diagnostics of the separately trained
recursive CDRDI. All conditions receive identical synthetic unregistered
observations and share the same frozen Stage-2D diffusion checkpoint.
HPGC/GIGI is disabled throughout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch
import torch.nn.functional as F

import train_innovation3_gigi_cdrdi as pipeline
from cdrdi_geometry import (
    forward_warp,
    jacobian_determinant,
    sample_synthetic_geometry,
    sampling_coordinates,
    spectral_project,
)
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from utils import get_device, load_checkpoint, set_seed


METRICS = ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")


def parse_args():
    p = argparse.ArgumentParser(
        description="PaviaU unregistered CDRDI one-step/recursive ablation"
    )
    p.add_argument("--dataset", choices=("PaviaU",), default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument(
        "--geometry_checkpoint",
        default="./checkpoints/PaviaU/PaviaU_recursive_k6_finalonly_500ep_lr1e4.pth",
    )
    p.add_argument(
        "--diffusion_checkpoint",
        default="./checkpoints/PaviaU/PaviaU_estimated_deform_diffusion_k9_stage2d_A.pth",
    )
    p.add_argument(
        "--one_shot_geometry_checkpoint",
        default="./checkpoints/PaviaU/PaviaU_one_shot_independent_finalonly_500ep_lr1e4.pth",
    )
    p.add_argument("--output_dir", default="./results/ablation_cdrdi_iteration/PaviaU")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 evaluates the complete test split; positive values are smoke tests")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 3, 6, 9])
    p.add_argument("--recursive_steps", type=int, default=9,
                   help="Formal current-method recursive depth; default matches Stage-2D k9")

    # PaviaU benchmark and physical degradation settings.
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=("pchip", "linear"), default="pchip")
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)

    # Geometry checkpoint architecture and deformation distribution.
    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)

    # Stage-2D checkpoint architecture. These must match its saved weights.
    p.add_argument("--diffusion_base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--overwrite_output", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _read_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(
        str(tuple(value.shape)).encode("utf-8") + value.numpy().tobytes()
    ).hexdigest()


def _validate_checkpoints(args):
    geometry_path = Path(args.geometry_checkpoint)
    one_shot_path = Path(args.one_shot_geometry_checkpoint)
    diffusion_path = Path(args.diffusion_checkpoint)
    for path in (geometry_path, one_shot_path, diffusion_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    geometry_state = _read_checkpoint(geometry_path)
    one_shot_state = _read_checkpoint(one_shot_path)
    diffusion_state = _read_checkpoint(diffusion_path)
    if not all(isinstance(x, dict) for x in (geometry_state, one_shot_state, diffusion_state)):
        raise TypeError("All checkpoints must be PyTorch checkpoint dictionaries")

    geometry_extra = geometry_state.get("extra", {})
    one_shot_extra = one_shot_state.get("extra", {})
    diffusion_extra = diffusion_state.get("extra", {})
    if not isinstance(geometry_extra, dict):
        geometry_extra = {}
    if not isinstance(one_shot_extra, dict):
        one_shot_extra = {}
    if not isinstance(diffusion_extra, dict):
        diffusion_extra = {}

    if geometry_extra.get("dataset") not in (None, "PaviaU"):
        raise ValueError("Geometry checkpoint is not a PaviaU checkpoint")
    if geometry_extra.get("variant") not in (None, "recursive"):
        raise ValueError("Expected the migrated recursive CDRDI checkpoint")
    if geometry_extra.get("loss_mode") not in (None, "final_only"):
        raise ValueError("Expected the final_only recursive CDRDI checkpoint")
    if geometry_extra.get("steps") not in (None, 6):
        raise ValueError(
            f"Expected a K=6-trained recursive geometry checkpoint, got steps={geometry_extra.get('steps')}"
        )
    if one_shot_extra.get("dataset") not in (None, "PaviaU"):
        raise ValueError("One-shot geometry checkpoint is not a PaviaU checkpoint")
    if one_shot_extra.get("variant") != "one_shot":
        raise ValueError("One-shot checkpoint must be independently trained with --variant one_shot")
    if one_shot_extra.get("steps") not in (None, 1):
        raise ValueError("One-shot checkpoint must contain exactly one trained update")
    if one_shot_extra.get("loss_mode") not in (None, "final_only"):
        raise ValueError("One-shot checkpoint must use final_only closure supervision")
    for key, expected in (
        ("max_translation", args.max_translation),
        ("max_rotation_deg", args.max_rotation_deg),
        ("max_local_px", args.max_local_px),
        ("control_grid", args.control_grid),
    ):
        actual = one_shot_extra.get(key)
        if actual is not None and not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"One-shot geometry metadata mismatch: {key}={actual}, expected {expected}")

    if diffusion_extra.get("dataset") not in (None, "PaviaU"):
        raise ValueError("Stage-2D diffusion checkpoint is not a PaviaU checkpoint")
    expected_diffusion = {
        "geometry_steps": 9,
        "diffusion_steps": args.diffusion_steps,
        "base_channels": args.diffusion_base_channels,
        "time_dim": args.time_dim,
        "spectral_hidden": args.spectral_hidden,
    }
    for key, expected in expected_diffusion.items():
        actual = diffusion_extra.get(key)
        if actual is not None and int(actual) != int(expected):
            raise ValueError(
                f"Stage-2D checkpoint metadata mismatch: {key}={actual}, expected {expected}"
            )

    for key, expected in (
        ("max_translation", args.max_translation),
        ("max_rotation_deg", args.max_rotation_deg),
        ("max_local_px", args.max_local_px),
        ("control_grid", args.control_grid),
    ):
        actual = geometry_extra.get(key)
        if actual is not None and not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-8):
            raise ValueError(
                f"Geometry checkpoint metadata mismatch: {key}={actual}, expected {expected}"
            )

    return {
        "geometry": {
            "path": str(geometry_path.resolve()),
            "sha256": _sha256_file(geometry_path),
            "epoch": geometry_state.get("epoch"),
            "best_metric": geometry_state.get("best_metric"),
            "extra": geometry_extra,
        },
        "one_shot_geometry": {
            "path": str(one_shot_path.resolve()),
            "sha256": _sha256_file(one_shot_path),
            "epoch": one_shot_state.get("epoch"),
            "best_metric": one_shot_state.get("best_metric"),
            "extra": one_shot_extra,
        },
        "diffusion": {
            "path": str(diffusion_path.resolve()),
            "sha256": _sha256_file(diffusion_path),
            "epoch": diffusion_state.get("epoch"),
            "best_metric": diffusion_state.get("best_metric"),
            "extra": diffusion_extra,
        },
    }


def _geometry_metrics(
    gt_phi,
    pred_rigid: torch.Tensor,
    pred_local: torch.Tensor,
    *,
    height: int,
    width: int,
):
    gt_x, gt_y = sampling_coordinates(
        height, width, gt_phi.dx, gt_phi.dy, gt_phi.theta_deg, gt_phi.local_field
    )
    pred_x, pred_y = sampling_coordinates(
        height, width,
        pred_rigid[:, 0], pred_rigid[:, 1], pred_rigid[:, 2], pred_local
    )
    epe = torch.sqrt((gt_x - pred_x).square() + (gt_y - pred_y).square())
    return {
        "EPE_mean_px": float(epe.mean().item()),
        "EPE_p95_px": float(torch.quantile(epe.flatten(), 0.95).item()),
        "min_jacobian": float(jacobian_determinant(pred_local).amin().item()),
        "dx_abs_error_px": float((pred_rigid[:, 0] - gt_phi.dx).abs().mean().item()),
        "dy_abs_error_px": float((pred_rigid[:, 1] - gt_phi.dy).abs().mean().item()),
        "rotation_abs_error_deg": float(
            (pred_rigid[:, 2] - gt_phi.theta_deg).abs().mean().item()
        ),
    }


def _identity_geometry(gt: torch.Tensor):
    batch, _, height, width = gt.shape
    rigid = gt.new_zeros((batch, 3))
    local = gt.new_zeros((batch, 2, height, width))
    return rigid, local


def _closure_mse(p0, srf, observed, hr_msi, rigid, local):
    target = spectral_project(observed, srf)
    warped_msi = forward_warp(
        hr_msi, rigid[:, 0], rigid[:, 1], rigid[:, 2], local
    )
    predicted = p0.degrade(warped_msi)
    return float(F.mse_loss(predicted, target).item())


def _mean_metrics(rows: List[Dict[str, float]]):
    if not rows:
        raise ValueError("Cannot aggregate an empty metric list")
    return {key: mean(float(row[key]) for row in rows) for key in rows[0]}


def _write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run(args):
    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint_info = _validate_checkpoints(args)

    cfg = pipeline._config(args)
    _, test_loader, info = pipeline.build_loaders(cfg)
    if args.batch_size != 1:
        raise ValueError("Evaluation requires batch_size=1 for patch-macro averaging")

    base_process = pipeline.build_progressive_process(cfg)
    p0 = base_process.operator
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=torch.float32)

    geometry_model, diffusion_model = pipeline._build_frozen_models(args, info, device)
    geometry_model.eval().requires_grad_(False)
    diffusion_model.eval().requires_grad_(False)

    one_shot_model = LearnedPhysicalResidualSolver(
        info["n_msi_bands"],
        base_channels=args.geometry_base_channels,
        control_grid=args.control_grid,
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px,
    ).to(device)
    load_checkpoint(
        one_shot_model,
        args.one_shot_geometry_checkpoint,
        map_location=str(device),
        load_optimizer=False,
        strict=True,
    )
    one_shot_model.eval().requires_grad_(False)

    depths = sorted(set(int(x) for x in args.depths))
    if any(x < 2 for x in depths):
        raise ValueError("--depths are recursive depth diagnostics; values must be >=2")
    if args.recursive_steps not in depths:
        raise ValueError("--depths must include --recursive_steps")
    if args.cases < 1:
        raise ValueError("--cases must be >=1")

    variants = ["w/o CDRDI", "One-shot"] + [f"CDRDI_K{k}" for k in depths]
    patch_summaries: List[Dict[str, object]] = []
    case_path = Path(args.output_dir) / "cases.jsonl"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("PAVIAU CDRDI ITERATION ABLATION")
    print("Registration: unregistered synthetic acquisitions")
    print("HPGC/GIGI: disabled")
    print("w/o CDRDI: same Stage-2D weights + identity physical operator")
    print("K=1: one update using the same K=6-trained recursive weights")
    print(f"Recursive main condition: K={args.recursive_steps}")
    print("Depth diagnostics:", depths)
    print("Geometry checkpoint:", args.geometry_checkpoint)
    print("Diffusion checkpoint:", args.diffusion_checkpoint)
    print("=" * 96)

    max_samples = args.max_samples if args.max_samples > 0 else None
    with case_path.open("w", encoding="utf-8") as case_file:
        for sample_index, batch in enumerate(test_loader):
            if max_samples is not None and sample_index >= max_samples:
                break
            gt = batch["gt"].to(device=device, dtype=torch.float32, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device=device, dtype=torch.float32, non_blocking=True)
            if gt.shape[0] != 1:
                raise ValueError("Test loader must yield one patch at a time")
            height, width = gt.shape[-2:]
            generator = pipeline._make_generator(device, args.seed + 70000)

            sample_values: Dict[str, List[Dict[str, float]]] = {
                name: [] for name in variants
            }
            for case_index in range(args.cases):
                gt_phi = sample_synthetic_geometry(
                    height, width,
                    device=device, dtype=gt.dtype, generator=generator,
                    max_translation=args.max_translation,
                    max_rotation_deg=args.max_rotation_deg,
                    max_local_px=args.max_local_px,
                    control_grid=args.control_grid,
                    min_jacobian=args.min_jacobian,
                )
                true_process = pipeline._process_from_geometry(base_process, gt_phi)
                observed = true_process.terminal_observation(gt)
                observed_hash = _sha256_tensor(observed)

                # No CDRDI: retain the same unregistered observation and D2
                # weights, but use the unwarped physical degradation operator.
                identity_rigid, identity_local = _identity_geometry(gt)
                no_cdrdi_pred = pipeline.reconstruct_from_terminal_lr(
                    diffusion_model, base_process, observed,
                    target_size=(height, width), hr_msi=hr_msi,
                )
                no_cdrdi_metrics = pipeline.calc_metrics(
                    no_cdrdi_pred, gt, args.scale_ratio
                )
                no_cdrdi_geo = _geometry_metrics(
                    gt_phi, identity_rigid, identity_local,
                    height=height, width=width,
                )
                no_cdrdi_geo["closure_mse"] = _closure_mse(
                    p0, srf, observed, hr_msi, identity_rigid, identity_local
                )
                sample_values["w/o CDRDI"].append(
                    {**no_cdrdi_metrics, **no_cdrdi_geo}
                )
                case_file.write(json.dumps({
                    "sample": sample_index + 1,
                    "case": case_index + 1,
                    "variant": "w/o CDRDI",
                    "geometry_steps": 0,
                    "observed_sha256": observed_hash,
                    "metrics": no_cdrdi_metrics,
                    "geometry": no_cdrdi_geo,
                }, allow_nan=False) + "\n")

                # Independently trained one-shot geometry predictor: exactly one update.
                one_rigid, one_local = pipeline._estimate_geometry(
                    one_shot_model,
                    y_h=observed,
                    hr_msi=hr_msi,
                    p0=p0,
                    srf=srf,
                    steps=1,
                )
                one_process = pipeline._process_from_estimate(
                    base_process, one_rigid, one_local
                )
                one_prediction = pipeline.reconstruct_from_terminal_lr(
                    diffusion_model, one_process, observed,
                    target_size=(height, width), hr_msi=hr_msi,
                )
                one_metrics = pipeline.calc_metrics(
                    one_prediction, gt, args.scale_ratio
                )
                one_geo = _geometry_metrics(
                    gt_phi, one_rigid, one_local,
                    height=height, width=width,
                )
                one_geo["closure_mse"] = _closure_mse(
                    p0, srf, observed, hr_msi, one_rigid, one_local
                )
                sample_values["One-shot"].append({**one_metrics, **one_geo})
                case_file.write(json.dumps({
                    "sample": sample_index + 1,
                    "case": case_index + 1,
                    "variant": "One-shot",
                    "geometry_steps": 1,
                    "observed_sha256": observed_hash,
                    "metrics": one_metrics,
                    "geometry": one_geo,
                    "estimated_rigid": one_rigid[0].detach().cpu().tolist(),
                }, allow_nan=False) + "\n")

                for depth in depths:
                    pred_rigid, pred_local = pipeline._estimate_geometry(
                        geometry_model,
                        y_h=observed,
                        hr_msi=hr_msi,
                        p0=p0,
                        srf=srf,
                        steps=depth,
                    )
                    estimated_process = pipeline._process_from_estimate(
                        base_process, pred_rigid, pred_local
                    )
                    prediction = pipeline.reconstruct_from_terminal_lr(
                        diffusion_model, estimated_process, observed,
                        target_size=(height, width), hr_msi=hr_msi,
                    )
                    metrics = pipeline.calc_metrics(
                        prediction, gt, args.scale_ratio
                    )
                    geometry = _geometry_metrics(
                        gt_phi, pred_rigid, pred_local,
                        height=height, width=width,
                    )
                    geometry["closure_mse"] = _closure_mse(
                        p0, srf, observed, hr_msi, pred_rigid, pred_local
                    )
                    name = f"CDRDI_K{depth}"
                    sample_values[name].append({**metrics, **geometry})
                    case_file.write(json.dumps({
                        "sample": sample_index + 1,
                        "case": case_index + 1,
                        "variant": name,
                        "geometry_steps": depth,
                        "observed_sha256": observed_hash,
                        "metrics": metrics,
                        "geometry": geometry,
                        "estimated_rigid": pred_rigid[0].detach().cpu().tolist(),
                    }, allow_nan=False) + "\n")

                case_file.flush()

            for name in variants:
                values = sample_values[name]
                patch_summaries.append({
                    "sample": sample_index + 1,
                    "variant": name,
                    "cases": args.cases,
                    **_mean_metrics(values),
                })
            print(
                f"TEST_PATCH {sample_index + 1} cases={args.cases} "
                f"observed={observed_hash[:12]}",
                flush=True,
            )

    if not patch_summaries:
        raise RuntimeError("No test patches were evaluated")

    summary_rows: List[Dict[str, object]] = []
    recursive_name = f"CDRDI_K{args.recursive_steps}"
    aggregated = {}
    for name in variants:
        rows = [row for row in patch_summaries if row["variant"] == name]
        aggregated[name] = _mean_metrics([
            {key: float(row[key]) for key in METRICS + (
                "EPE_mean_px", "EPE_p95_px", "min_jacobian",
                "dx_abs_error_px", "dy_abs_error_px",
                "rotation_abs_error_deg", "closure_mse",
            )}
            for row in rows
        ])
    recursive_reference = aggregated[recursive_name]
    for name in variants:
        numeric = aggregated[name]
        summary_rows.append({
            "Variant": (
                "w/o CDRDI" if name == "w/o CDRDI"
                else ("One-shot" if name == "One-shot"
                      else (f"Recursive K={args.recursive_steps}" if name == recursive_name
                            else name.replace("CDRDI_", "Depth diagnostic ")))
            ),
            **numeric,
            "PSNR_delta_vs_recursive_db": numeric["PSNR"] - float(recursive_reference["PSNR"]),
            "SAM_reduction_vs_recursive_deg": float(recursive_reference["SAM"]) - numeric["SAM"],
        })

    _write_csv(output / "summary.csv", summary_rows)
    _write_csv(output / "per_patch.csv", patch_summaries)
    manifest = {
        "protocol_id": "PaviaU-CDRDI-iteration-ablation-v1",
        "dataset": "PaviaU",
        "registration": "unregistered",
        "hpgc_refiner": False,
        "same_diffusion_checkpoint_for_all_conditions": True,
        "no_cdrdi_definition": "identity geometry with the same frozen Stage-2D diffusion weights",
        "one_shot_definition": "independently trained CDRDI checkpoint with variant=one_shot and steps=1; evaluated with the same frozen Stage-2D diffusion checkpoint as recursive conditions",
        "recursive_definition": f"K={args.recursive_steps} inference using the K=6-trained recursive CDRDI weights",
        "depth_diagnostics": depths,
        "same_observation_per_case": True,
        "geometry_sampler": "sample_synthetic_geometry; seed reset to seed+70000 per test patch",
        "aggregation": "mean over cases within each patch, then equal-weight mean over test patches",
        "args": vars(args),
        "checkpoints": checkpoint_info,
        "summary": summary_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== PAVIAU CDRDI ITERATION ABLATION SUMMARY ===")
    for row in summary_rows:
        print(
            f"{row['Variant']:<24} "
            + " ".join(f"{key}={float(row[key]):.6f}" for key in (
                "PSNR", "SAM", "SSIM", "ERGAS",
                "EPE_mean_px", "EPE_p95_px", "closure_mse",
            ))
        )
    print("CSV:", output / "summary.csv")
    print("Per-patch CSV:", output / "per_patch.csv")
    print("Case details:", case_path)
    print("Manifest:", output / "manifest.json")


def main():
    args = parse_args()
    if args.recursive_steps < 2:
        raise ValueError("--recursive_steps must be >=2")
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        for value in (args.geometry_checkpoint, args.diffusion_checkpoint):
            if not Path(value).is_file():
                raise FileNotFoundError(value)
        print("DRY_RUN_OK: checkpoint paths exist; no models or data were loaded.")
        return
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory is nonempty: {output}. Choose a new directory "
            "or pass --overwrite_output to replace its result files."
        )
    run(args)


if __name__ == "__main__":
    main()
