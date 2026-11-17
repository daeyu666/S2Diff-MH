"""PaviaU CDRDI: explicit registration vs deformation-aware fusion ablation.

2x2 paired design:
  estimated geometry x explicit LR-HSI registration
  oracle geometry    x explicit LR-HSI registration
  estimated geometry x deformation-aware physical diffusion
  oracle geometry    x deformation-aware physical diffusion

The explicit-registration arm inverse-resamples the observed LR-HSI using the
estimated/oracle HR geometry converted to the LR grid, then uses the registered
Raw-MSI Direct checkpoint. The deformation-aware arm keeps the LR observation
unaltered and embeds the geometry in the progressive physical acquisition
operator, using the Stage-2D estimated-geometry-adapted diffusion checkpoint.

All four conditions share each synthetic observation, HR-MSI, GT, case schedule,
and a common valid evaluation mask. HPGC/GIGI is disabled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

import train_innovation3_gigi as registered_pipeline
import train_innovation3_gigi_cdrdi as pipeline
from cdrdi_geometry import sample_synthetic_geometry
from utils import set_seed


METRICS = ("PSNR", "SAM", "SSIM", "ERGAS", "CC", "RMSE")
VARIANTS = (
    "Estimated + Explicit Registration",
    "Oracle + Explicit Registration",
    "Estimated + Deformation-aware",
    "Oracle + Deformation-aware",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="PaviaU CDRDI estimated/oracle x registration/deformation-aware ablation"
    )
    p.add_argument("--dataset", choices=("PaviaU",), default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--geometry_checkpoint",
                   default="./checkpoints/PaviaU/PaviaU_recursive_k6_finalonly_500ep_lr1e4.pth")
    p.add_argument("--diffusion_checkpoint",
                   default="./checkpoints/PaviaU/PaviaU_estimated_deform_diffusion_k9_stage2d_A.pth")
    p.add_argument("--registered_checkpoint",
                   default="./checkpoints/PaviaU/PaviaU_innovation1_physical_raw_direct.pth")
    p.add_argument("--registered_full_checkpoint",
                   default="./checkpoints/PaviaU/PaviaU_innovation3_gigi_full.pth",
                   help="Registered Full HPGC checkpoint used only to recover the exact registered-baseline loading/config metadata.")
    p.add_argument("--output_dir",
                   default="./results/ablation_cdrdi_registration/PaviaU")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 evaluates the complete test split; positive values are smoke tests.")
    p.add_argument("--geometry_steps", type=int, default=9)

    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=("pchip", "linear"), default="pchip")
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)

    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)

    p.add_argument("--diffusion_base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--inverse_iterations", type=int, default=40)
    p.add_argument("--inverse_tolerance", type=float, default=1e-4)
    p.add_argument("--inverse_valid_margin_lr", type=int, default=3)
    p.add_argument("--inverse_residual_tolerance", type=float, default=0.10)
    p.add_argument("--overwrite_output", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _read_record(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(
        str(tuple(value.shape)).encode("utf-8") + value.numpy().tobytes()
    ).hexdigest()


def _validate_checkpoints(args):
    paths = {
        "geometry": Path(args.geometry_checkpoint),
        "estimated_diffusion": Path(args.diffusion_checkpoint),
        "registered_diffusion": Path(args.registered_checkpoint),
        "registered_full_metadata": Path(args.registered_full_checkpoint),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint missing: {path}")

    geo = _read_record(paths["geometry"])
    est_d2 = _read_record(paths["estimated_diffusion"])
    reg_d2 = _read_record(paths["registered_diffusion"])
    reg_full = _read_record(paths["registered_full_metadata"])
    if not all(isinstance(x, dict) for x in (geo, est_d2, reg_d2, reg_full)):
        raise TypeError("Expected PyTorch checkpoint dictionaries")

    gx = geo.get("extra", {}) or {}
    dx = est_d2.get("extra", {}) or {}
    rx = reg_full.get("extra", {}) or {}
    if not all(isinstance(x, dict) for x in (gx, dx, rx)):
        raise TypeError("Checkpoint extra metadata must be dictionaries")

    if gx.get("dataset") not in (None, "PaviaU"):
        raise ValueError("CDRDI checkpoint is not a PaviaU checkpoint")
    if gx.get("variant") not in (None, "recursive"):
        raise ValueError("Expected the trained recursive CDRDI checkpoint")
    if gx.get("steps") not in (None, 6):
        raise ValueError(f"Expected K=6-trained CDRDI weights, got {gx.get('steps')}")

    if dx.get("dataset") not in (None, "PaviaU"):
        raise ValueError("Estimated Stage-2D checkpoint is not PaviaU")
    expected_d2 = {
        "geometry_steps": args.geometry_steps,
        "diffusion_steps": args.diffusion_steps,
        "base_channels": args.diffusion_base_channels,
        "time_dim": args.time_dim,
        "spectral_hidden": args.spectral_hidden,
    }
    for key, expected in expected_d2.items():
        actual = dx.get(key)
        if actual is not None and int(actual) != int(expected):
            raise ValueError(
                f"Estimated Stage-2D metadata mismatch: {key}={actual}, expected {expected}"
            )
    trained_geometry_path = dx.get("geometry_checkpoint")
    if trained_geometry_path:
        if Path(str(trained_geometry_path)).name != paths["geometry"].name:
            raise ValueError(
                "The supplied CDRDI checkpoint differs from the geometry checkpoint "
                "recorded by Stage-2D: "
                f"{Path(str(trained_geometry_path)).name} vs {paths['geometry'].name}"
            )

    if rx.get("dataset") not in (None, "PaviaU"):
        raise ValueError("Registered Full HPGC metadata checkpoint is not PaviaU")
    if rx.get("variant") not in (None, "full"):
        raise ValueError("registered_full_checkpoint must be the registered Full HPGC checkpoint")
    recorded_baseline = rx.get("baseline_checkpoint")
    if recorded_baseline and Path(str(recorded_baseline)).name != paths["registered_diffusion"].name:
        raise ValueError(
            "Registered baseline path does not match the baseline used by registered Full HPGC: "
            f"{Path(str(recorded_baseline)).name} vs {paths['registered_diffusion'].name}"
        )

    baseline_type = rx.get("baseline_checkpoint_type", "legacy")
    if baseline_type not in ("legacy", "standard"):
        raise ValueError(f"Unsupported registered checkpoint type: {baseline_type}")
    for key, expected in (
        ("diffusion_steps", args.diffusion_steps),
        ("base_channels", args.diffusion_base_channels),
        ("time_dim", args.time_dim),
        ("spectral_hidden", args.spectral_hidden),
    ):
        actual = rx.get(key)
        if actual is not None and int(actual) != int(expected):
            raise ValueError(
                f"Registered/estimated predictor architecture mismatch: {key}={actual}, expected {expected}"
            )

    for key, expected in (
        ("max_translation", args.max_translation),
        ("max_rotation_deg", args.max_rotation_deg),
        ("max_local_px", args.max_local_px),
        ("control_grid", args.control_grid),
    ):
        actual = gx.get(key)
        if actual is not None and not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"CDRDI geometry metadata mismatch: {key}={actual}, expected {expected}")

    return {
        "geometry": {"path": str(paths["geometry"].resolve()), "sha256": _sha256(paths["geometry"]),
                     "epoch": geo.get("epoch"), "best_metric": geo.get("best_metric"), "extra": gx},
        "estimated_diffusion": {"path": str(paths["estimated_diffusion"].resolve()),
                                "sha256": _sha256(paths["estimated_diffusion"]),
                                "epoch": est_d2.get("epoch"), "best_metric": est_d2.get("best_metric"),
                                "extra": dx},
        "registered_diffusion": {"path": str(paths["registered_diffusion"].resolve()),
                                 "sha256": _sha256(paths["registered_diffusion"])},
        "registered_full_metadata": {"path": str(paths["registered_full_metadata"].resolve()),
                                     "sha256": _sha256(paths["registered_full_metadata"]),
                                     "extra": rx},
        "registered_baseline_type": baseline_type,
    }


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name} was requested but CUDA is not available")
        index = int(name.split(":", 1)[1])
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {name}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
        return torch.device(name)
    return torch.device(name)


def _make_generator(device: torch.device, seed: int):
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _grid_from_xy(x: torch.Tensor, y: torch.Tensor, height: int, width: int):
    gx = 2.0 * x / max(width - 1, 1) - 1.0
    gy = 2.0 * y / max(height - 1, 1) - 1.0
    return torch.stack((gx, gy), dim=-1)


def inverse_register_lr(
    observed_lr: torch.Tensor,
    rigid_hr: torch.Tensor,
    local_hr: torch.Tensor,
    *,
    scale_ratio: int,
    iterations: int,
    tolerance: float,
    valid_margin_lr: int,
    residual_tolerance: float,
):
    """Resample observed LR-HSI into the reference MSI grid using phi^{-1}.

    Geometry is converted from HR pixels to LR pixels. The inverse map solves
    p + u(p) = R^{-1}(q-c-d)+c by fixed-point iteration. This is explicit
    post-acquisition registration; it is intentionally distinct from the
    deformation-aware acquisition operator used by the proposed method.
    """
    if observed_lr.ndim != 4 or rigid_hr.ndim != 2 or rigid_hr.shape[1] != 3:
        raise ValueError("Invalid observed LR-HSI or rigid geometry shape")
    b, _, h, w = observed_lr.shape
    # Area-averaged LR pixel i is centered at HR coordinate
    # scale_ratio*i + (scale_ratio-1)/2. align_corners=False uses exactly
    # this half-pixel center mapping when downscaling by an integer factor.
    local_lr = F.interpolate(
        local_hr, size=(h, w), mode="bilinear", align_corners=False
    ) / float(scale_ratio)
    dx = rigid_hr[:, 0] / float(scale_ratio)
    dy = rigid_hr[:, 1] / float(scale_ratio)
    angle = rigid_hr[:, 2] * (math.pi / 180.0)
    cos_a = torch.cos(angle).view(b, 1, 1)
    sin_a = torch.sin(angle).view(b, 1, 1)

    yy, xx = torch.meshgrid(
        torch.arange(h, device=observed_lr.device, dtype=observed_lr.dtype),
        torch.arange(w, device=observed_lr.device, dtype=observed_lr.dtype),
        indexing="ij",
    )
    qx = xx.unsqueeze(0).expand(b, -1, -1)
    qy = yy.unsqueeze(0).expand(b, -1, -1)
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0

    qx0 = qx - cx - dx.view(b, 1, 1)
    qy0 = qy - cy - dy.view(b, 1, 1)
    rx = cos_a * qx0 + sin_a * qy0 + cx
    ry = -sin_a * qx0 + cos_a * qy0 + cy

    px, py = rx.clone(), ry.clone()
    last_delta = float("inf")
    iterations_used = 0
    for iteration in range(int(iterations)):
        iterations_used = iteration + 1
        grid = _grid_from_xy(px, py, h, w)
        sampled_local = F.grid_sample(
            local_lr, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        next_px = rx - sampled_local[:, 0]
        next_py = ry - sampled_local[:, 1]
        last_delta = float(torch.maximum(
            (next_px - px).abs().amax(), (next_py - py).abs().amax()
        ).item())
        px, py = next_px, next_py
        if last_delta <= tolerance:
            break

    inverse_grid = _grid_from_xy(px, py, h, w)
    sampled_local = F.grid_sample(
        local_lr, inverse_grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    ux = px + sampled_local[:, 0] - cx
    uy = py + sampled_local[:, 1] - cy
    phi_x = cos_a * ux - sin_a * uy + cx + dx.view(b, 1, 1)
    phi_y = sin_a * ux + cos_a * uy + cy + dy.view(b, 1, 1)
    inverse_residual = torch.sqrt((phi_x - qx).square() + (phi_y - qy).square())

    margin = int(valid_margin_lr)
    valid = (
        (px >= margin) & (px <= (w - 1 - margin)) &
        (py >= margin) & (py <= (h - 1 - margin)) &
        (inverse_residual <= float(residual_tolerance))
    )
    registered_lr = F.grid_sample(
        observed_lr, inverse_grid, mode="bilinear",
        padding_mode="border", align_corners=True
    )
    diagnostics = {
        "inverse_iterations_used": int(iterations_used),
        "inverse_fixed_point_last_delta_px": last_delta,
        "inverse_residual_mean_lr_px": float(inverse_residual.mean().item()),
        "inverse_residual_p95_lr_px": float(torch.quantile(inverse_residual.flatten(), 0.95).item()),
        "valid_lr_fraction": float(valid.float().mean().item()),
    }
    return registered_lr, valid.unsqueeze(1), diagnostics


def _masked_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: int):
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("metric mask must have shape Bx1xHxW")
    valid = mask[:, 0].bool()
    if int(valid.sum().item()) < 64:
        raise RuntimeError(f"Too few common valid pixels: {int(valid.sum().item())}")
    p = pred.permute(0, 2, 3, 1)[valid]
    t = target.permute(0, 2, 3, 1)[valid]

    error = p - t
    rmse = torch.sqrt(torch.mean(error.square())).item()
    psnr = 100.0 if rmse <= 1e-12 else 20.0 * math.log10(1.0 / rmse)

    pn = torch.linalg.vector_norm(p, dim=1)
    tn = torch.linalg.vector_norm(t, dim=1)
    spectral_valid = (pn > 1e-12) & (tn > 1e-12)
    if spectral_valid.any():
        cosine = (
            (p[spectral_valid] * t[spectral_valid]).sum(dim=1) /
            (pn[spectral_valid] * tn[spectral_valid]).clamp_min(1e-12)
        ).clamp(-1.0, 1.0)
        sam = torch.acos(cosine).mean().item() * (180.0 / math.pi)
    else:
        sam = 0.0

    p_band, t_band = p.transpose(0, 1), t.transpose(0, 1)
    rmse_band = torch.sqrt(torch.mean((p_band - t_band).square(), dim=1) + 1e-8)
    mean_target = t_band.mean(dim=1)
    ergas = (100.0 / float(scale)) * torch.sqrt(
        torch.mean((rmse_band / (mean_target + 1e-8)).square())
    ).item()

    pc = p_band - p_band.mean(dim=1, keepdim=True)
    tc = t_band - t_band.mean(dim=1, keepdim=True)
    cc_den = torch.sqrt((pc.square().sum(dim=1) * tc.square().sum(dim=1)).clamp_min(1e-16))
    cc = ((pc * tc).sum(dim=1) / cc_den).mean().item()

    pf, tf = p.reshape(-1), t.reshape(-1)
    mu_p, mu_t = pf.mean(), tf.mean()
    var_p, var_t = pf.var(unbiased=False), tf.var(unbiased=False)
    cov = ((pf - mu_p) * (tf - mu_t)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = (
        (2 * mu_p * mu_t + c1) * (2 * cov + c2) /
        ((mu_p.square() + mu_t.square() + c1) * (var_p + var_t + c2) + 1e-8)
    ).item()
    return {"PSNR": float(psnr), "SAM": float(sam), "SSIM": float(ssim),
            "ERGAS": float(ergas), "CC": float(cc), "RMSE": float(rmse)}


def _mean_metrics(rows: List[Dict[str, float]]):
    if not rows:
        raise ValueError("Cannot average an empty metric list")
    return {key: mean(float(row[key]) for row in rows) for key in rows[0]}


def _write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_registered_model(args, info, device, checkpoint_info):
    extra = checkpoint_info["registered_full_metadata"]["extra"]
    baseline_args = SimpleNamespace(
        diffusion_steps=args.diffusion_steps,
        base_channels=args.diffusion_base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        spectral_hidden=args.spectral_hidden,
        baseline_checkpoint_type=checkpoint_info["registered_baseline_type"],
        baseline_checkpoint=args.registered_checkpoint,
    )
    model = registered_pipeline._build_baseline(baseline_args, info, device)
    return model


@torch.no_grad()
def run(args):
    set_seed(args.seed)
    device = _resolve_device(args.device)
    checkpoint_info = _validate_checkpoints(args)

    cfg = pipeline._config(args)
    _, test_loader, info = pipeline.build_loaders(cfg)
    base_process = pipeline.build_progressive_process(cfg)
    p0 = base_process.operator
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=torch.float32)

    geometry_model, estimated_diffusion_model = pipeline._build_frozen_models(
        args, info, device
    )
    geometry_model.eval().requires_grad_(False)
    estimated_diffusion_model.eval().requires_grad_(False)
    registered_diffusion_model = _build_registered_model(
        args, info, device, checkpoint_info
    )
    registered_diffusion_model.eval().requires_grad_(False)

    print("=" * 100)
    print("PAVIAU CDRDI REGISTRATION / DEFORMATION-AWARE ABLATION")
    print("Unregistered synthetic observations; HPGC/GIGI disabled")
    print("Explicit registration: inverse-resample LR-HSI, then registered Raw-MSI Direct fusion")
    print("Deformation-aware: preserve LR-HSI and use A_{t,phi}=D_t o W_phi")
    print("Geometry source: CDRDI K=9 estimate or synthetic oracle")
    print("Registered checkpoint:", args.registered_checkpoint)
    print("Estimated-geometry D2 checkpoint:", args.diffusion_checkpoint)
    print("CDRDI checkpoint:", args.geometry_checkpoint)
    print("=" * 100)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    case_path = output / "cases.jsonl"
    patch_rows: List[Dict[str, object]] = []
    case_rows: List[Dict[str, object]] = []
    max_samples = args.max_samples if args.max_samples > 0 else None

    with case_path.open("w", encoding="utf-8") as case_file:
        for sample_index, batch in enumerate(test_loader):
            if max_samples is not None and sample_index >= max_samples:
                break
            gt = batch["gt"].to(device=device, dtype=torch.float32, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device=device, dtype=torch.float32, non_blocking=True)
            if gt.shape[0] != 1:
                raise ValueError("Evaluation requires batch_size=1 and one test patch per loader item")
            height, width = gt.shape[-2:]
            generator = _make_generator(device, args.seed + 70000)
            per_patch = {name: [] for name in VARIANTS}

            for case_index in range(args.cases):
                gt_phi = sample_synthetic_geometry(
                    height, width, device=device, dtype=gt.dtype, generator=generator,
                    max_translation=args.max_translation,
                    max_rotation_deg=args.max_rotation_deg,
                    max_local_px=args.max_local_px,
                    control_grid=args.control_grid,
                    min_jacobian=args.min_jacobian,
                )
                true_process = pipeline._process_from_geometry(base_process, gt_phi)
                observed = true_process.terminal_observation(gt)
                observed_hash = _tensor_sha256(observed)

                pred_rigid, pred_local = pipeline._estimate_geometry(
                    geometry_model, y_h=observed, hr_msi=hr_msi,
                    p0=p0, srf=srf, steps=args.geometry_steps,
                )

                est_registered_lr, est_valid, est_inv_diag = inverse_register_lr(
                    observed, pred_rigid, pred_local,
                    scale_ratio=args.scale_ratio,
                    iterations=args.inverse_iterations,
                    tolerance=args.inverse_tolerance,
                    valid_margin_lr=args.inverse_valid_margin_lr,
                    residual_tolerance=args.inverse_residual_tolerance,
                )
                oracle_rigid = torch.stack(
                    (gt_phi.dx, gt_phi.dy, gt_phi.theta_deg), dim=1
                )
                oracle_registered_lr, oracle_valid, oracle_inv_diag = inverse_register_lr(
                    observed, oracle_rigid, gt_phi.local_field,
                    scale_ratio=args.scale_ratio,
                    iterations=args.inverse_iterations,
                    tolerance=args.inverse_tolerance,
                    valid_margin_lr=args.inverse_valid_margin_lr,
                    residual_tolerance=args.inverse_residual_tolerance,
                )

                # The same valid pixels are used for every condition in this case.
                common_valid_lr = est_valid & oracle_valid
                common_valid_hr = F.interpolate(
                    common_valid_lr.float(), size=(height, width), mode="nearest"
                ) > 0.5
                valid_fraction = float(common_valid_hr.float().mean().item())

                # A1/B1: explicit post-acquisition LR-HSI registration, then registered fusion.
                est_reg_pred = registered_pipeline.reconstruct_from_terminal_lr(
                    registered_diffusion_model, base_process, est_registered_lr,
                    target_size=(height, width), hr_msi=hr_msi,
                )
                oracle_reg_pred = registered_pipeline.reconstruct_from_terminal_lr(
                    registered_diffusion_model, base_process, oracle_registered_lr,
                    target_size=(height, width), hr_msi=hr_msi,
                )

                # A2/B2: no explicit warp of the observed LR-HSI; geometry is
                # embedded in the progressive physical acquisition operator.
                est_process = pipeline._process_from_estimate(
                    base_process, pred_rigid, pred_local
                )
                oracle_process = pipeline._process_from_geometry(base_process, gt_phi)
                est_dwa_pred = pipeline.reconstruct_from_terminal_lr(
                    estimated_diffusion_model, est_process, observed,
                    target_size=(height, width), hr_msi=hr_msi,
                )
                oracle_dwa_pred = pipeline.reconstruct_from_terminal_lr(
                    estimated_diffusion_model, oracle_process, observed,
                    target_size=(height, width), hr_msi=hr_msi,
                )

                predictions = {
                    VARIANTS[0]: est_reg_pred,
                    VARIANTS[1]: oracle_reg_pred,
                    VARIANTS[2]: est_dwa_pred,
                    VARIANTS[3]: oracle_dwa_pred,
                }
                geometry_epe = pipeline._geometry_epe(
                    height, width, gt_phi, pred_rigid, pred_local
                )
                case_metrics = {}
                for name, prediction in predictions.items():
                    metrics = _masked_metrics(prediction, gt, common_valid_hr, args.scale_ratio)
                    case_metrics[name] = metrics
                    per_patch[name].append(metrics)
                    row = {
                        "sample": sample_index + 1,
                        "case": case_index + 1,
                        "variant": name,
                        **metrics,
                        "valid_hr_fraction": valid_fraction,
                        "observed_sha256": observed_hash,
                    }
                    case_rows.append(row)
                    case_file.write(json.dumps({
                        **row,
                        "geometry_epe_mean_hr_px": geometry_epe,
                        "estimated_inverse_registration": est_inv_diag,
                        "oracle_inverse_registration": oracle_inv_diag,
                        "estimated_rigid": pred_rigid[0].detach().cpu().tolist(),
                        "oracle_rigid": oracle_rigid[0].detach().cpu().tolist(),
                    }, ensure_ascii=False, allow_nan=False) + "\n")

                print(
                    f"sample={sample_index+1:03d} case={case_index+1:02d} "
                    f"valid={valid_fraction:.3f} EPE={geometry_epe:.4f}px "
                    f"EST_REG_PSNR={case_metrics[VARIANTS[0]]['PSNR']:.4f} "
                    f"ORACLE_REG_PSNR={case_metrics[VARIANTS[1]]['PSNR']:.4f} "
                    f"EST_DWA_PSNR={case_metrics[VARIANTS[2]]['PSNR']:.4f} "
                    f"ORACLE_DWA_PSNR={case_metrics[VARIANTS[3]]['PSNR']:.4f}",
                    flush=True,
                )

            for name in VARIANTS:
                patch_rows.append({
                    "sample": sample_index + 1,
                    "variant": name,
                    **_mean_metrics(per_patch[name]),
                })

    if not patch_rows:
        raise RuntimeError("No PaviaU test patches were evaluated")

    summary = []
    for name in VARIANTS:
        rows = [r for r in patch_rows if r["variant"] == name]
        summary.append({"Variant": name, **_mean_metrics([
            {key: float(r[key]) for key in METRICS} for r in rows
        ])})

    _write_csv(output / "summary.csv", [
        {"Variant": r["Variant"], "PSNR": r["PSNR"], "SAM": r["SAM"]}
        for r in summary
    ])
    _write_csv(output / "summary_extended.csv", summary)
    _write_csv(output / "per_patch.csv", patch_rows)
    _write_csv(output / "per_case.csv", case_rows)

    manifest = {
        "protocol_id": "PaviaU-CDRDI-registration-vs-deformation-aware-v1",
        "dataset": "PaviaU",
        "scale_ratio": args.scale_ratio,
        "registration": "synthetic unregistered acquisitions",
        "hpgc_refiner": False,
        "factorial_design": {
            "geometry_source": ["CDRDI-estimated", "synthetic-oracle"],
            "fusion_strategy": ["explicit-LR-HSI-registration", "deformation-aware-physical-operator"],
        },
        "explicit_registration": {
            "method": "inverse-resample observed LR-HSI with phi^{-1} on LR grid",
            "geometry_scaling": "HR displacement fields sampled at area-averaged LR pixel centers with align_corners=False, then divided by scale_ratio; translations divided by scale_ratio; rotation unchanged",
            "inverse_solver": "fixed-point inversion of p + u(p) = R^{-1}(q-c-d)+c",
            "iterations": args.inverse_iterations,
            "valid_margin_lr": args.inverse_valid_margin_lr,
            "note": "Post-acquisition resampling is approximate because blur/downsampling precede explicit LR-grid registration.",
            "diffusion_checkpoint": args.registered_checkpoint,
        },
        "deformation_aware": {
            "method": "A_{t,phi}=D_t o W_phi with no explicit LR-HSI warp",
            "diffusion_checkpoint": args.diffusion_checkpoint,
        },
        "same_synthetic_observation_per_case": True,
        "same_valid_evaluation_mask_for_all_four_conditions": True,
        "valid_mask": "intersection of estimated/oracle inverse-map support, excluding configured LR boundary margin",
        "case_schedule": "sample_synthetic_geometry with seed reset to seed+70000 per test patch",
        "aggregation": "mean over cases within each patch, then equal-weight mean over test patches",
        "args": vars(args),
        "checkpoints": checkpoint_info,
        "summary": summary,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== PAVIAU CDRDI REGISTRATION ABLATION SUMMARY ===")
    for row in summary:
        print(
            f"{row['Variant']:<38} "
            f"PSNR={row['PSNR']:.6f} SAM={row['SAM']:.6f} "
            f"SSIM={row['SSIM']:.6f} ERGAS={row['ERGAS']:.6f}"
        )
    print("CSV:", output / "summary.csv")
    print("Extended CSV:", output / "summary_extended.csv")
    print("Per-patch CSV:", output / "per_patch.csv")
    print("Per-case CSV:", output / "per_case.csv")
    print("Case details:", case_path)
    print("Manifest:", output / "manifest.json")


def main():
    args = parse_args()
    if args.cases < 1 or args.geometry_steps < 1:
        raise ValueError("--cases and --geometry_steps must be >=1")
    if args.inverse_iterations < 1 or args.inverse_valid_margin_lr < 0:
        raise ValueError("Invalid inverse registration settings")
    if args.batch_size != 1:
        raise ValueError("Evaluation requires --batch_size 1")
    if args.dry_run:
        info = _validate_checkpoints(args)
        print(json.dumps({
            "dataset": args.dataset,
            "geometry_checkpoint": info["geometry"]["path"],
            "estimated_diffusion_checkpoint": info["estimated_diffusion"]["path"],
            "registered_checkpoint": info["registered_diffusion"]["path"],
            "registered_baseline_type": info["registered_baseline_type"],
            "output_dir": args.output_dir,
        }, ensure_ascii=False, indent=2))
        print("DRY_RUN_OK: checkpoints and metadata validated; data/models not loaded.")
        return

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory is nonempty: {output}. Choose a new output directory "
            "or pass --overwrite_output to replace its result files."
        )
    run(args)


if __name__ == "__main__":
    main()
