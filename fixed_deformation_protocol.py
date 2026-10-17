"""Identical fixed-strength final-test protocol distributed in both repositories.

Rigid motion is fixed, not sampled or alpha-scaled. The random local field can
be either the original cubic B-spline field or a Gaussian-smoothed elastic
field. Geometry uses a private CPU generator so model/GPU RNG cannot change it.
Predictions receive observations only, never GT geometry or HSI.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from tqdm import tqdm


PROTOCOL_ID = "fixed-rigid-random-local-v2"
METRIC_KEYS = ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")


@dataclass(frozen=True)
class Strength:
    level: int
    translation_hr_px: float
    rotation_deg: float
    local_max_hr_px: float


STRENGTHS = (
    Strength(1, 0.5, 0.25, 0.5),
    Strength(2, 1.0, 0.5, 1.0),
    Strength(3, 2.0, 1.0, 2.0),
    Strength(4, 3.0, 1.5, 3.0),
    Strength(5, 4.0, 2.0, 4.0),
    Strength(6, 6.0, 3.0, 4.0),
)


@dataclass
class FixedGeometry:
    dx: torch.Tensor
    dy: torch.Tensor
    theta_deg: torch.Tensor
    control: torch.Tensor
    local_field: torch.Tensor


def add_suite_args(parser, *, include_cases=True):
    if include_cases:
        parser.add_argument("--cases", type=int, default=10,
                            help="Random local fields per test patch per strength (default: 10)")
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--levels", type=int, nargs="+", choices=range(1, 7),
                        default=list(range(1, 7)))
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--local_deformation", choices=("bspline", "elastic"), default="bspline",
        help="Local deformation family. Default preserves the original B-spline protocol.",
    )
    parser.add_argument(
        "--elastic_sigma_hr_px", type=float, default=8.0,
        help="Gaussian smoothing sigma in HR pixels for elastic local deformation.",
    )
    parser.add_argument(
        "--elastic_truncate", type=float, default=3.0,
        help="Gaussian kernel radius in sigma units for elastic local deformation.",
    )


def tensor_sha256(tensor):
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def _basis(n_ctrl, n_out):
    u = torch.linspace(0.0, float(n_ctrl - 1), n_out, dtype=torch.float32)
    base = u.floor().long()
    t = u - base
    weights = torch.stack([
        (1-t).pow(3)/6, (3*t.pow(3)-6*t.pow(2)+4)/6,
        (-3*t.pow(3)+3*t.pow(2)+3*t+1)/6, t.pow(3)/6,
    ], dim=1)
    indices = torch.stack([base-1, base, base+1, base+2], dim=1).clamp(0, n_ctrl-1)
    return torch.zeros(n_out, n_ctrl).scatter_add_(1, indices, weights)


def _dense_field(control, height, width):
    return torch.einsum("hi,bcij,wj->bchw", _basis(control.shape[-2], height),
                        control, _basis(control.shape[-1], width))


def _gaussian_kernel1d(sigma, truncate=3.0):
    sigma = float(sigma)
    truncate = float(truncate)
    if sigma <= 0 or truncate <= 0:
        raise ValueError("elastic sigma and truncate must be positive")
    radius = max(1, int(math.ceil(truncate * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    return kernel / kernel.sum().clamp_min(1e-12)


def _elastic_field(noise, sigma, truncate=3.0):
    """Gaussian-smoothed dense random field, with global translation removed."""
    if noise.ndim != 4 or noise.shape[1] != 2:
        raise ValueError("elastic noise must have shape Bx2xHxW")
    kernel = _gaussian_kernel1d(sigma, truncate).to(device=noise.device, dtype=noise.dtype)
    radius = kernel.numel() // 2
    # reflect padding requires the padding radius to be smaller than the image size.
    max_radius = min(noise.shape[-2], noise.shape[-1]) - 1
    if max_radius < 1:
        raise ValueError("elastic deformation needs spatial dimensions >=2")
    if radius > max_radius:
        radius = max_radius
        coords = torch.arange(-radius, radius + 1, device=noise.device, dtype=noise.dtype)
        kernel = torch.exp(-0.5 * (coords / float(sigma)).square())
        kernel = kernel / kernel.sum().clamp_min(1e-12)
    channels = noise.shape[1]
    weight_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    weight_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    field = F.conv2d(F.pad(noise, (radius, radius, 0, 0), mode="reflect"),
                     weight_x, groups=channels)
    field = F.conv2d(F.pad(field, (0, 0, radius, radius), mode="reflect"),
                     weight_y, groups=channels)
    return field - field.mean(dim=(-2, -1), keepdim=True)


def local_field_statistics(field):
    magnitude = torch.linalg.vector_norm(field, dim=1)
    gx = field[:, :, :-1, 1:] - field[:, :, :-1, :-1]
    gy = field[:, :, 1:, :-1] - field[:, :, :-1, :-1]
    grad = torch.sqrt((gx.square() + gy.square()).sum(dim=1).clamp_min(0.0))
    return {
        "local_mean_hr_px": float(magnitude.mean().item()),
        "local_rms_hr_px": float(torch.sqrt(magnitude.square().mean()).item()),
        "local_p95_hr_px": float(torch.quantile(magnitude.flatten(), 0.95).item()),
        "local_max_hr_px": float(magnitude.amax().item()),
        "local_grad_p95": float(torch.quantile(grad.flatten(), 0.95).item()),
    }


def local_jacobian(field):
    vx, vy = field[:, 0], field[:, 1]
    a = vx[:, :-1, 1:] - vx[:, :-1, :-1]
    b = vx[:, 1:, :-1] - vx[:, :-1, :-1]
    c = vy[:, :-1, 1:] - vy[:, :-1, :-1]
    d = vy[:, 1:, :-1] - vy[:, :-1, :-1]
    return (1+a)*(1+d) - b*c


def sample_fixed_geometry(height, width, strength, *, sample_index, case_index,
                          seed=10, control_grid=5, min_jacobian=0.5,
                          local_deformation="bspline", elastic_sigma_hr_px=8.0,
                          elastic_truncate=3.0):
    """CPU/float32 synthesis; the seed is independent of level/model/device.

    Paired levels start with the same candidate field and amplitude fraction.
    Rejection is allowed only for local folding; rigid motion is never altered.
    Levels 5 and 6 have identical local fields, isolating rigid extrapolation.
    """
    if min(height, width) < 2 or control_grid < 2:
        raise ValueError("Spatial dimensions and control_grid must be >=2")
    if not 0 < min_jacobian <= 1:
        raise ValueError("min_jacobian must be in (0,1]")
    if local_deformation not in ("bspline", "elastic"):
        raise ValueError("local_deformation must be 'bspline' or 'elastic'")
    if elastic_sigma_hr_px <= 0 or elastic_truncate <= 0:
        raise ValueError("elastic sigma/truncate must be positive")
    local_seed = (int(seed) + 70000 + int(sample_index)*1000003
                  + int(case_index)*9176) % (2**63 - 1)
    gen = torch.Generator(device="cpu").manual_seed(local_seed)
    for attempt in range(64):
        if local_deformation == "bspline":
            control = torch.randn(1, 2, control_grid, control_grid, generator=gen)
            control -= control.mean(dim=(-2, -1), keepdim=True)
            dense = _dense_field(control, height, width)
            amplitude = torch.rand(1, generator=gen) * strength.local_max_hr_px
            control *= amplitude / torch.linalg.vector_norm(dense, dim=1).amax().clamp_min(1e-8)
            dense = _dense_field(control, height, width)
        else:
            # PSRF-style elastic family: dense random displacement followed by
            # Gaussian spatial smoothing. This is not claimed to reproduce the
            # unpublished Elastic1000 generator exactly.
            control = torch.randn(1, 2, height, width, generator=gen)
            control -= control.mean(dim=(-2, -1), keepdim=True)
            dense = _elastic_field(control, elastic_sigma_hr_px, elastic_truncate)
            amplitude = torch.rand(1, generator=gen) * strength.local_max_hr_px
            dense *= amplitude / torch.linalg.vector_norm(dense, dim=1).amax().clamp_min(1e-8)
        jacobian = local_jacobian(dense).amin().item()
        if jacobian >= min_jacobian:
            geometry = FixedGeometry(
                dx=torch.tensor([strength.translation_hr_px], dtype=torch.float32),
                dy=torch.tensor([strength.translation_hr_px], dtype=torch.float32),
                theta_deg=torch.tensor([strength.rotation_deg], dtype=torch.float32),
                control=control, local_field=dense,
            )
            stats = local_field_statistics(dense)
            record = {
                "geometry_seed": local_seed, "local_sampling_attempts": attempt + 1,
                "local_deformation": local_deformation,
                "elastic_sigma_hr_px": float(elastic_sigma_hr_px) if local_deformation == "elastic" else None,
                "dx_hr_px": float(geometry.dx.item()),
                "dy_hr_px": float(geometry.dy.item()),
                "rotation_deg": float(geometry.theta_deg.item()),
                **stats,
                "min_local_jacobian": jacobian,
                "local_field_sha256": tensor_sha256(dense),
            }
            return geometry, record
    raise RuntimeError("Failed to sample a non-folding local field in 64 attempts")


def warp_hsi(image, geometry):
    """Same local-then-rigid border sampler as existing final tests."""
    h, w = image.shape[-2:]
    local = geometry.local_field.to(device=image.device, dtype=image.dtype)
    yy, xx = torch.meshgrid(torch.arange(h, device=image.device, dtype=image.dtype),
                            torch.arange(w, device=image.device, dtype=image.dtype), indexing="ij")
    xx = xx.unsqueeze(0) + local[:, 0]
    yy = yy.unsqueeze(0) + local[:, 1]
    cx, cy = (w-1)/2, (h-1)/2
    angle = geometry.theta_deg.to(image) * (math.pi/180)
    ca, sa = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    sx = ca*(xx-cx) - sa*(yy-cy) + cx + geometry.dx.to(image).view(-1, 1, 1)
    sy = sa*(xx-cx) + ca*(yy-cy) + cy + geometry.dy.to(image).view(-1, 1, 1)
    grid = torch.stack([2*sx/max(w-1, 1)-1, 2*sy/max(h-1, 1)-1], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


def calc_metrics(pred, target, scale_ratio):
    """Preserve existing S2 final-test formulas, including global SSIM.

    PSNR/RMSE/SAM use clamp01; CC/ERGAS/SSIM use the raw predictions. Keeping
    this function identical across repositories avoids their root metric drift.
    """
    pred, target = pred.detach().float(), target.detach().float()
    p, t = pred.clamp(0, 1), target.clamp(0, 1)
    rmse = math.sqrt(max(F.mse_loss(p, t).item(), 1e-12))
    pn, tn = torch.linalg.vector_norm(p, dim=1), torch.linalg.vector_norm(t, dim=1)
    valid = (pn > 1e-12) & (tn > 1e-12)
    sam = 0.0
    if valid.any():
        cos = ((p*t).sum(dim=1)[valid] / (pn[valid]*tn[valid]).clamp_min(1e-12)).clamp(-1, 1)
        sam = (cos.acos()*(180/math.pi)).mean().item()
    pf, tf = pred.reshape(pred.shape[0], pred.shape[1], -1), target.reshape(target.shape[0], target.shape[1], -1)
    pc, tc = pf-pf.mean(dim=2, keepdim=True), tf-tf.mean(dim=2, keepdim=True)
    cc = ((pc*tc).sum(dim=2) / ((pc.square().sum(dim=2)*tc.square().sum(dim=2)+1e-8).sqrt()+1e-8)).mean().item()
    band_rmse = ((pred-target).square().mean(dim=(0, 2, 3))+1e-8).sqrt()
    ergas = (100/scale_ratio * ((band_rmse/(target.mean(dim=(0, 2, 3))+1e-8)).square().mean()).sqrt()).item()
    mx, my = pred.mean(), target.mean()
    vx, vy = pred.var(unbiased=False), target.var(unbiased=False)
    cov = ((pred-mx)*(target-my)).mean()
    ssim = (((2*mx*my+0.01**2)*(2*cov+0.03**2)) /
            ((mx.square()+my.square()+0.01**2)*(vx+vy+0.03**2)+1e-8)).item()
    return {"PSNR": 100.0 if rmse <= 1e-12 else 20*math.log10(1/rmse),
            "SSIM": ssim, "ERGAS": ergas, "SAM": sam, "CC": cc, "RMSE": rmse}


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def run_fixed_suite(test_set, predict, degrade, *, device, output_dir, metadata,
                    cases=10, seed=10, eval_seed=1234, levels=None,
                    control_grid=5, min_jacobian=0.5, scale_ratio=4,
                    local_deformation="bspline", elastic_sigma_hr_px=8.0,
                    elastic_truncate=3.0):
    """Each level traverses every patch; average cases, then patches equally.

    ``predict(lr_hsi, hr_msi)`` sees neither GT nor synthesis geometry. Every
    case gets a private reconstruction RNG seed, paired across strengths.
    """
    if cases < 1 or len(test_set) < 1:
        raise ValueError("cases and test patch count must be >=1")
    levels = list(range(1, 7)) if levels is None else list(levels)
    if not levels or len(set(levels)) != len(levels) or any(x not in range(1, 7) for x in levels):
        raise ValueError("levels must be distinct integers from 1 to 6")
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = {
        "protocol_id": PROTOCOL_ID, "torch_version": str(torch.__version__),
        "seed": seed, "eval_seed": eval_seed, "geometry_device": "cpu/float32",
        "rigid_definition": "dx=dy=+translation_hr_px; theta=+rotation_deg; no alpha",
        "local_deformation": local_deformation,
        "local_amplitude_proposal": "U(0,level.local_max_hr_px), conditioned on min local Jacobian",
        "elastic_sigma_hr_px": float(elastic_sigma_hr_px) if local_deformation == "elastic" else None,
        "elastic_truncate": float(elastic_truncate) if local_deformation == "elastic" else None,
        "case_schedule": "private seed per patch/case; paired across levels and models",
        "warp_observation": "Y_H=P0(W_phi(X)); HR-MSI and GT stay registered",
        "control_grid": control_grid, "min_local_jacobian": min_jacobian,
        "scale_ratio": scale_ratio, "cases_per_patch": cases,
        "test_patch_count": len(test_set), "selected_levels": levels,
        "aggregation": "mean of per-patch case means; full patch, no border crop",
        "metrics_protocol": "existing S2 final formulas; raw RMSE; simplified global SSIM",
        "model": metadata,
    }
    summary = {"test_conditions": conditions, "results": []}
    reference_inputs = None
    for level_id in levels:
        strength = STRENGTHS[level_id - 1]
        per_case, per_patch, inputs = [], [], []
        progress = tqdm(range(len(test_set)), desc=f"fixed level {level_id}/6", dynamic_ncols=True)
        for sample_index in progress:
            item = test_set[sample_index]
            gt = item["gt"].unsqueeze(0).to(device=device, dtype=torch.float32)
            hr_msi = item["hr_msi"].unsqueeze(0).to(device=device, dtype=torch.float32)
            fingerprint = {"sample": sample_index + 1, "shape": list(gt.shape),
                           "gt_sha256": tensor_sha256(gt), "msi_sha256": tensor_sha256(hr_msi)}
            inputs.append(fingerprint)
            if reference_inputs is not None and fingerprint != reference_inputs[sample_index]:
                raise RuntimeError("Test inputs/order changed between strength levels")
            patch_rows = []
            for case_index in range(cases):
                geometry, record = sample_fixed_geometry(
                    *gt.shape[-2:], strength, sample_index=sample_index,
                    case_index=case_index, seed=seed, control_grid=control_grid,
                    min_jacobian=min_jacobian, local_deformation=local_deformation,
                    elastic_sigma_hr_px=elastic_sigma_hr_px,
                    elastic_truncate=elastic_truncate,
                )
                lr_hsi = degrade(warp_hsi(gt, geometry))
                reconstruction_seed = (eval_seed + sample_index*1000003 + case_index*9176) % (2**63-1)
                cuda_devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.random.default_generator.manual_seed(reconstruction_seed)
                    if cuda_devices:
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(reconstruction_seed)
                    prediction = predict(lr_hsi, hr_msi)
                if prediction.shape != gt.shape or not torch.isfinite(prediction).all():
                    raise ValueError("Prediction must be finite and have the GT HSI shape")
                metrics = calc_metrics(prediction, gt, scale_ratio)
                patch_rows.append(metrics)
                per_case.append({"sample": sample_index+1, "case": case_index+1,
                                 "reconstruction_seed": reconstruction_seed, **record, "metrics": metrics})
            per_patch.append({key: mean(row[key] for row in patch_rows) for key in METRIC_KEYS})
        reference_inputs = inputs
        averaged = {key: mean(row[key] for row in per_patch) for key in METRIC_KEYS}
        local_values = [row["local_max_hr_px"] for row in per_case]
        local_p95_values = [row["local_p95_hr_px"] for row in per_case]
        local_mean_values = [row["local_mean_hr_px"] for row in per_case]
        local_grad_values = [row["local_grad_p95"] for row in per_case]
        result = {**asdict(strength), "local_deformation": local_deformation,
                  "test_patch_count": len(test_set),
                  "evaluated_cases": len(per_case), "average_metrics": averaged,
                  "actual_local_max_hr_px": {"min": min(local_values), "mean": mean(local_values), "max": max(local_values)},
                  "actual_local_p95_hr_px": {"min": min(local_p95_values), "mean": mean(local_p95_values), "max": max(local_p95_values)},
                  "actual_local_mean_hr_px": {"min": min(local_mean_values), "mean": mean(local_mean_values), "max": max(local_mean_values)},
                  "actual_local_grad_p95": {"min": min(local_grad_values), "mean": mean(local_grad_values), "max": max(local_grad_values)},
                  "detail_json": f"level_{level_id}.json"}
        _write_json(output_dir / result["detail_json"], {
            "test_conditions": conditions, "result": result,
            "test_inputs": inputs, "per_patch_metrics": per_patch, "per_sample_case": per_case,
        })
        summary["results"].append(result)
        _write_json(output_dir / "summary.json", summary)
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["level", "translation_hr_px", "rotation_deg", "local_max_hr_px",
                      "test_patch_count", "evaluated_cases", *METRIC_KEYS]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in summary["results"]:
                writer.writerow({**{key: row[key] for key in fields if key not in METRIC_KEYS}, **row["average_metrics"]})
        print(f"FIXED_LEVEL={level_id} dx=dy={strength.translation_hr_px:g}px "
              f"rotation={strength.rotation_deg:g}deg local={local_deformation}<={strength.local_max_hr_px:g}px "
              + " ".join(f"{key}={averaged[key]:.6f}" for key in METRIC_KEYS), flush=True)
    print(f"FIXED_TEST_SUMMARY={output_dir / 'summary.csv'}", flush=True)
    return summary
