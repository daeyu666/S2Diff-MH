"""Oracle-capacity and correction-alignment diagnostic for Innovation 3.

This diagnostic answers the question raised by the first Full experiment:

    the MSI heterogeneity gate locates the hard region correctly,
    but does the allowed C4:L+tangent correction space contain the true
    spectral-angle descent direction, and does the learned correction align
    with that direction?

For each selected reverse-diffusion timestep, using the actual refined reverse
trajectory, it computes

    d* = v - <u,v> u

where u is the unit-normalized Raw-Direct base prediction and v is unit GT HSI.

Two nested oracle projections are then evaluated:

    P_S(d*)                 : C4:L broad-shape subspace
    P_{S cap T_u}(d*)       : C4:L intersect current spectral tangent space

The principal statistics are

    CAPTURE_C4L = ||P_S(d*)||^2 / ||d*||^2
    ORACLE_CAPTURE = ||P_{S cap T_u}(d*)||^2 / ||d*||^2
    ALIGN = cos(delta_net, P_{S cap T_u}(d*))

where delta_net is the learned pre-normalization correction actually produced
by the trained Full refiner.  All statistics are reported overall and in the
top/bottom MSI-heterogeneity quartiles.

GT is used only for this post-hoc diagnosis.  It is not an inference input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from config import TrainConfig
from data_loader import build_loaders
from innovation1 import build_progressive_process
from metrics import calc_metrics
from models import (
    HeterogeneityGuidedSpectralPredictor,
    RawMSIDirectPredictor,
    load_legacy_raw_direct_checkpoint,
    ranked_msi_heterogeneity,
)
from utils import ensure_dir, get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 oracle correction capacity and learned-alignment diagnostic"
    )
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)

    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--timesteps", default="1,3,6,9,12")

    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--refine_hidden", type=int, default=64)
    p.add_argument("--variant", choices=["generic", "hetero", "broad", "full"], default="full")

    p.add_argument(
        "--baseline_checkpoint",
        default="./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth",
    )
    p.add_argument(
        "--baseline_checkpoint_type",
        choices=["legacy", "standard"],
        default="legacy",
    )
    p.add_argument(
        "--refiner_checkpoint",
        default="./checkpoints/innovation3/PaviaU_innovation3_full_smoke.pth",
    )

    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument(
        "--output_json",
        default="./results/innovation3_oracle_capacity_PaviaU.json",
    )
    return p.parse_args()


def _parse_timesteps(text: str, total_steps: int) -> Tuple[int, ...]:
    values = tuple(sorted({int(v.strip()) for v in text.split(",") if v.strip()}))
    if not values:
        raise ValueError("--timesteps is empty")
    if any(v < 1 or v > total_steps for v in values):
        raise ValueError(f"timesteps must lie in [1,{total_steps}]")
    return values


def _config(args) -> TrainConfig:
    return TrainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        patch_size=64,
        stride=32,
        test_size=args.test_size,
        scale_ratio=args.scale_ratio,
        srf_interp=args.srf_interp,
        degradation_mode="physical",
        diffusion_steps=args.diffusion_steps,
        lift_mode="normalized_adjoint",
        mtf_nyquist=args.mtf_nyquist,
        psf_truncate=args.psf_truncate,
        predictor="raw_direct",
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        spectral_hidden=args.spectral_hidden,
        dropout=args.dropout,
        batch_size=1,
        num_workers=0,
        seed=args.seed,
        device=args.device,
    )


def _build_model(args, info, device):
    backbone = RawMSIDirectPredictor(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_msi_bands"],
        total_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        residual_prediction=True,
        spectral_hidden=args.spectral_hidden,
    ).to(device)

    if args.baseline_checkpoint_type == "legacy":
        report = load_legacy_raw_direct_checkpoint(
            backbone,
            args.baseline_checkpoint,
            map_location=str(device),
        )
        print("BASELINE_LOAD legacy", report)
    else:
        epoch, best = load_checkpoint(
            backbone,
            args.baseline_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        print(f"BASELINE_LOAD standard epoch={epoch} best={best}")

    model = HeterogeneityGuidedSpectralPredictor(
        backbone,
        n_bands=info["n_bands"],
        total_steps=args.diffusion_steps,
        hidden_channels=args.refine_hidden,
        variant=args.variant,
        freeze_backbone=True,
    ).to(device)

    try:
        state = torch.load(
            args.refiner_checkpoint,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        state = torch.load(args.refiner_checkpoint, map_location=device)

    checkpoint_variant = state.get("variant", args.variant)
    if checkpoint_variant != args.variant:
        raise ValueError(
            f"refiner checkpoint variant={checkpoint_variant!r}, requested={args.variant!r}"
        )
    if "refiner" not in state:
        raise KeyError("refiner checkpoint does not contain key 'refiner'")
    model.refiner.load_state_dict(state["refiner"], strict=True)
    model.eval()
    print(
        f"REFINER_LOAD path={args.refiner_checkpoint} "
        f"epoch={state.get('epoch', 0)} "
        f"best_SAM_HIGH={state.get('best_high_sam', float('nan'))}"
    )
    return model


def _interior_spectra(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("expected BxCxHxW")
    if x.shape[-2] < 3 or x.shape[-1] < 3:
        raise ValueError("spatial size must be >=3")
    return (
        x[:, :, 1:-1, 1:-1]
        .permute(0, 2, 3, 1)
        .reshape(-1, x.shape[1])
        .float()
    )


def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    denom = (
        torch.linalg.vector_norm(a, dim=1)
        * torch.linalg.vector_norm(b, dim=1)
    ).clamp_min(eps)
    return (a * b).sum(dim=1) / denom


def _mean(values: torch.Tensor) -> float:
    values = values[torch.isfinite(values)]
    return float(values.mean().item()) if values.numel() else float("nan")


def _median(values: torch.Tensor) -> float:
    values = values[torch.isfinite(values)]
    return float(values.median().item()) if values.numel() else float("nan")


def _region_masks(risk: torch.Tensor, fraction: float):
    r = risk[:, 1:-1, 1:-1].reshape(-1).float()
    valid = torch.isfinite(r)
    finite = r[valid]
    lo = torch.quantile(finite, fraction)
    hi = torch.quantile(finite, 1.0 - fraction)
    return valid, r, r <= lo, r >= hi


def oracle_projection_components(
    refiner,
    base_x0: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Return ideal tangent direction and nested C4:L/full oracle projections."""
    base = base_x0.float()
    gt = target.float()
    base_norm = torch.linalg.vector_norm(base, dim=1, keepdim=True).clamp_min(eps)
    gt_norm = torch.linalg.vector_norm(gt, dim=1, keepdim=True).clamp_min(eps)
    u = base / base_norm
    v = gt / gt_norm

    uv = (u * v).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    ideal = v - uv * u

    broad = refiner.project_broadshape(ideal)
    allowed = refiner.project_tangent(
        broad,
        base,
        preserve_broadshape=True,
    )

    ideal_e = ideal.square().sum(dim=1)
    broad_e = broad.square().sum(dim=1)
    allowed_e = allowed.square().sum(dim=1)

    capture_c4l = broad_e / ideal_e.clamp_min(eps)
    capture_full = allowed_e / ideal_e.clamp_min(eps)

    # Best direction reachable in span{u, allowed}.  Since allowed lies in
    # u^perp and is the orthogonal projection of v onto the allowed subspace,
    # alpha*=1/<u,v> maximizes cosine with v for positive <u,v>.
    uv_scalar = uv.squeeze(1)
    alpha = torch.where(
        uv_scalar > eps,
        1.0 / uv_scalar.clamp_min(eps),
        torch.ones_like(uv_scalar),
    )
    oracle_unit = F.normalize(
        u + alpha.unsqueeze(1) * allowed,
        dim=1,
        eps=eps,
    )
    oracle_cos = (oracle_unit * v).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    base_cos = uv_scalar.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    base_sam = torch.acos(base_cos) * (180.0 / math.pi)
    oracle_sam = torch.acos(oracle_cos) * (180.0 / math.pi)

    return {
        "u": u,
        "v": v,
        "ideal": ideal,
        "broad": broad,
        "allowed": allowed,
        "ideal_energy": ideal_e,
        "capture_c4l": capture_c4l,
        "capture_full": capture_full,
        "base_sam": base_sam,
        "oracle_sam": oracle_sam,
    }


def _summarize_stage(
    refiner,
    base_x0: torch.Tensor,
    target: torch.Tensor,
    risk: torch.Tensor,
    network_update: torch.Tensor,
    *,
    region_fraction: float,
    eps: float,
) -> Dict[str, float]:
    oracle = oracle_projection_components(
        refiner,
        base_x0,
        target,
        eps=eps,
    )

    ideal_e = _interior_spectra(oracle["ideal"].unsqueeze(0)) if False else None
    # Crop the BxHxW scalar fields directly and the BxCxHxW vectors with
    # _interior_spectra so all arrays share exactly the same pixel ordering.
    ideal = _interior_spectra(oracle["ideal"])
    allowed = _interior_spectra(oracle["allowed"])
    update = _interior_spectra(network_update)

    ideal_energy = oracle["ideal_energy"][:, 1:-1, 1:-1].reshape(-1)
    capture_c4l = oracle["capture_c4l"][:, 1:-1, 1:-1].reshape(-1)
    capture_full = oracle["capture_full"][:, 1:-1, 1:-1].reshape(-1)
    base_sam = oracle["base_sam"][:, 1:-1, 1:-1].reshape(-1)
    oracle_sam = oracle["oracle_sam"][:, 1:-1, 1:-1].reshape(-1)

    _, r, low_mask, high_mask = _region_masks(risk, region_fraction)
    valid = (
        torch.isfinite(r)
        & torch.isfinite(ideal_energy)
        & (ideal_energy > eps)
        & torch.isfinite(capture_full)
    )

    update_norm = torch.linalg.vector_norm(update, dim=1)
    allowed_norm = torch.linalg.vector_norm(allowed, dim=1)
    align_valid = valid & (update_norm > eps) & (allowed_norm > eps)
    align = torch.full_like(ideal_energy, float("nan"))
    align[align_valid] = _safe_cos(
        update[align_valid],
        allowed[align_valid],
        eps,
    )
    align_gt = torch.full_like(ideal_energy, float("nan"))
    align_gt[align_valid] = _safe_cos(
        update[align_valid],
        ideal[align_valid],
        eps,
    )

    def group(mask: torch.Tensor, prefix: str) -> Dict[str, float]:
        m = valid & mask
        am = align_valid & mask
        gain = base_sam - oracle_sam
        result = {
            f"{prefix}N": int(m.sum().item()),
            f"{prefix}BASE_SAM": _mean(base_sam[m]),
            f"{prefix}CAPTURE_C4L": _mean(capture_c4l[m]),
            f"{prefix}ORACLE_CAPTURE": _mean(capture_full[m]),
            f"{prefix}ORACLE_ALLOWED_SAM": _mean(oracle_sam[m]),
            f"{prefix}ORACLE_ALLOWED_GAIN": _mean(gain[m]),
            f"{prefix}ALIGN": _mean(align[am]),
            f"{prefix}ALIGN_MEDIAN": _median(align[am]),
            f"{prefix}ALIGN_GT_TANGENT": _mean(align_gt[am]),
            f"{prefix}ALIGN_POSITIVE_FRAC": (
                float((align[am] > 0.0).float().mean().item())
                if int(am.sum().item()) > 0
                else float("nan")
            ),
            f"{prefix}NET_UPDATE_NORM": _mean(update_norm[m]),
            f"{prefix}ORACLE_ALLOWED_NORM": _mean(allowed_norm[m]),
        }
        return result

    summary = {}
    summary.update(group(torch.ones_like(valid, dtype=torch.bool), "ALL_"))
    summary.update(group(high_mask, "HIGH_"))
    summary.update(group(low_mask, "LOW_"))
    return summary


@torch.no_grad()
def _reverse_baseline(model, process, gt, hr_msi):
    terminal_lr = process.terminal_observation(gt)
    x_t = process.terminal_state(terminal_lr, target_size=tuple(gt.shape[-2:]))
    for t in range(process.total_steps, 0, -1):
        step = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        pred = model(x_t, hr_msi, step)
        x_t = process.reverse_update(x_t, pred, t)
    return x_t


def _print_stage(t: int, s: Dict[str, float]):
    print("=" * 132)
    print(
        f"T={t:02d} CAPACITY "
        f"ALL_C4L={s['ALL_CAPTURE_C4L']:.6f} "
        f"ALL_FULL={s['ALL_ORACLE_CAPTURE']:.6f} "
        f"HIGH_FULL={s['HIGH_ORACLE_CAPTURE']:.6f} "
        f"LOW_FULL={s['LOW_ORACLE_CAPTURE']:.6f}"
    )
    print(
        f"T={t:02d} ORACLE_SAM "
        f"ALL={s['ALL_BASE_SAM']:.6f}->{s['ALL_ORACLE_ALLOWED_SAM']:.6f} "
        f"GAIN={s['ALL_ORACLE_ALLOWED_GAIN']:.6f} "
        f"HIGH_GAIN={s['HIGH_ORACLE_ALLOWED_GAIN']:.6f} "
        f"LOW_GAIN={s['LOW_ORACLE_ALLOWED_GAIN']:.6f}"
    )
    print(
        f"T={t:02d} ALIGN "
        f"ALL={s['ALL_ALIGN']:+.6f} median={s['ALL_ALIGN_MEDIAN']:+.6f} "
        f"HIGH={s['HIGH_ALIGN']:+.6f} LOW={s['LOW_ALIGN']:+.6f} "
        f"POS_FRAC={s['ALL_ALIGN_POSITIVE_FRAC']:.6f} "
        f"ALIGN_GT={s['ALL_ALIGN_GT_TANGENT']:+.6f}"
    )
    print(
        f"T={t:02d} MAGNITUDE "
        f"NET_ALL={s['ALL_NET_UPDATE_NORM']:.8f} "
        f"NET_HIGH={s['HIGH_NET_UPDATE_NORM']:.8f} "
        f"NET_LOW={s['LOW_NET_UPDATE_NORM']:.8f} "
        f"ORACLE_ALLOWED={s['ALL_ORACLE_ALLOWED_NORM']:.8f}"
    )


def main():
    args = parse_args()
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    timesteps = _parse_timesteps(args.timesteps, args.diffusion_steps)

    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    _, test_loader, info = build_loaders(cfg)
    process = build_progressive_process(cfg)
    model = _build_model(args, info, device)

    print(
        f"DIAGNOSTIC dataset={args.dataset} timesteps={timesteps} "
        f"region_fraction={args.region_fraction} "
        f"C4L=[4,{model.refiner.low_end-1}] "
        "trajectory=actual_refined_reverse"
    )
    print(
        "INTERPRETATION "
        "high ORACLE_CAPTURE + low ALIGN => correction-information/direction problem; "
        "low ORACLE_CAPTURE => C4:L+tangent constraint is too restrictive; "
        "high capture and high alignment but weak realized SAM gain => inspect update magnitude/reverse coupling."
    )

    per_t = {t: [] for t in timesteps}
    final_rows = []

    with torch.no_grad():
        for batch in test_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            risk = ranked_msi_heterogeneity(hr_msi, eps=args.eps)

            baseline_final = _reverse_baseline(
                model.backbone,
                process,
                gt,
                hr_msi,
            )

            terminal_lr = process.terminal_observation(gt)
            x_t = process.terminal_state(
                terminal_lr,
                target_size=tuple(gt.shape[-2:]),
            )

            selected = set(timesteps)
            for t in range(process.total_steps, 0, -1):
                step = torch.full(
                    (x_t.shape[0],),
                    t,
                    dtype=torch.long,
                    device=device,
                )
                base_x0 = model.backbone(x_t, hr_msi, step)
                refined_x0, details = model.refiner(
                    x_t,
                    base_x0,
                    hr_msi,
                    step,
                    return_details=True,
                )
                if t in selected:
                    per_t[t].append(
                        _summarize_stage(
                            model.refiner,
                            base_x0,
                            gt,
                            risk,
                            details["pre_update"],
                            region_fraction=args.region_fraction,
                            eps=args.eps,
                        )
                    )
                x_t = process.reverse_update(x_t, refined_x0, t)

            refined_final = x_t
            base_metrics = calc_metrics(baseline_final, gt, args.scale_ratio)
            refined_metrics = calc_metrics(refined_final, gt, args.scale_ratio)
            final_rows.append(
                {
                    "baseline": base_metrics,
                    "refined": refined_metrics,
                }
            )

    results = {}
    for t in timesteps:
        rows = per_t[t]
        if not rows:
            raise RuntimeError(f"no diagnostic rows collected for t={t}")
        keys = rows[0].keys()
        summary = {}
        for key in keys:
            vals = [row[key] for row in rows]
            if key.endswith("N"):
                summary[key] = int(sum(vals))
            else:
                tensor = torch.tensor(vals, dtype=torch.float64)
                summary[key] = _mean(tensor)
        _print_stage(t, summary)
        results[str(t)] = summary

    def average_metric(group: str, key: str) -> float:
        values = torch.tensor(
            [row[group][key] for row in final_rows],
            dtype=torch.float64,
        )
        return _mean(values)

    final = {
        "baseline": {
            key: average_metric("baseline", key)
            for key in ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")
        },
        "refined": {
            key: average_metric("refined", key)
            for key in ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")
        },
    }
    print("=" * 132)
    print(
        "FINAL_CHECK "
        f"A0_PSNR={final['baseline']['PSNR']:.6f} "
        f"A0_SAM={final['baseline']['SAM']:.6f} "
        f"FULL_PSNR={final['refined']['PSNR']:.6f} "
        f"FULL_SAM={final['refined']['SAM']:.6f}"
    )

    payload = {
        "conditions": {
            "dataset": args.dataset,
            "test_size": args.test_size,
            "scale_ratio": args.scale_ratio,
            "timesteps": list(timesteps),
            "baseline_checkpoint": args.baseline_checkpoint,
            "baseline_checkpoint_type": args.baseline_checkpoint_type,
            "refiner_checkpoint": args.refiner_checkpoint,
            "variant": args.variant,
            "region_fraction": args.region_fraction,
            "degradation": "physical",
            "mtf_nyquist": args.mtf_nyquist,
            "psf_truncate": args.psf_truncate,
            "trajectory": "actual refined reverse diffusion trajectory",
            "oracle_gt_usage": "diagnostic only; never inference input",
            "c4l_range": [4, model.refiner.low_end - 1],
            "seed": args.seed,
        },
        "definitions": {
            "ideal_direction": "d*=v-<u,v>u",
            "capture_c4l": "||P_C4L(d*)||^2 / ||d*||^2",
            "oracle_capture": "||P_{C4L intersect tangent}(d*)||^2 / ||d*||^2",
            "align": "cos(network pre-update, oracle-allowed direction)",
            "oracle_allowed_sam": "best SAM in span{u, oracle-allowed direction}",
        },
        "timesteps": results,
        "final_check": final,
    }
    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
