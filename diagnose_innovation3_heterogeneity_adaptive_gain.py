"""Innovation-3 Stage-B2: zero-shot MSI-heterogeneity adaptive gain diagnostic.

Stage-B0.6 established that the terminal observable physical spectral residual is a
useful final-output correction direction with a scalar gain near alpha=4.
Stage-B1 showed that injecting this correction into the reverse trajectory is
inferior to a single terminal correction.

This diagnostic asks whether MSI heterogeneity should remain in Innovation 3 as
an adaptive *magnitude allocator* rather than a spectral-direction predictor.

Let q(p) in [0,1] be the ranked HR-MSI local spectral heterogeneity.  The gain is

    a_beta(p) = alpha0 * [1 + beta * (2 q(p) - 1)]

so beta>0 moves correction strength from low-heterogeneity pixels to
high-heterogeneity pixels, beta<0 is the anti-heterogeneity control, and beta=0
is the validated constant-gain baseline.

By default each spatially modulated update is L2-normalized per case to the same
energy as the beta=0 update.  This prevents a result from being explained by a
larger total update norm.

No network is trained. GT is used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import torch

from config import TrainConfig
from data_loader import build_loaders
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from metrics import MetricAverager, calc_metrics
from models import (
    HeterogeneityGuidedSpectralRefiner,
    RawMSIDirectPredictor,
    load_legacy_raw_direct_checkpoint,
    ranked_msi_heterogeneity,
)
from utils import ensure_dir, get_device, load_checkpoint, set_seed


DEFAULT_BETAS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 Stage-B2 heterogeneity-adaptive physical residual gain"
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

    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument(
        "--baseline_checkpoint",
        default="./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth",
    )
    p.add_argument(
        "--baseline_checkpoint_type",
        choices=["legacy", "standard"],
        default="legacy",
    )
    p.add_argument("--alpha0", type=float, default=4.0)
    p.add_argument(
        "--betas",
        default=",".join(str(v) for v in DEFAULT_BETAS),
        help="Comma-separated beta values in [-1,1]. Positive favors high heterogeneity.",
    )
    p.add_argument(
        "--no_norm_match",
        action="store_true",
        help="Disable per-case L2 matching to the beta=0 update energy.",
    )
    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument(
        "--output_json",
        default="./results/innovation3_heterogeneity_adaptive_gain_PaviaU.json",
    )
    return p.parse_args()


def _parse_betas(text: str) -> Tuple[float, ...]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            value = float(token)
            if value < -1.0 or value > 1.0:
                raise ValueError("--betas must lie in [-1,1]")
            if value not in values:
                values.append(value)
    if 0.0 not in values:
        values.append(0.0)
    return tuple(values)


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


def _build_baseline(args, info, device):
    model = RawMSIDirectPredictor(
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
            model, args.baseline_checkpoint, map_location=str(device)
        )
        print("BASELINE_LOAD legacy", report)
    else:
        epoch, best = load_checkpoint(
            model,
            args.baseline_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        print(f"BASELINE_LOAD standard epoch={epoch} best={best}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _pixel_sam_deg(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    dot = (pred.float() * target.float()).sum(dim=1)
    pn = torch.linalg.vector_norm(pred.float(), dim=1)
    tn = torch.linalg.vector_norm(target.float(), dim=1)
    cos = dot / (pn * tn).clamp_min(eps)
    return torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * (180.0 / math.pi)


def _region_sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    risk: torch.Tensor,
    fraction: float,
    eps: float,
) -> Tuple[float, float]:
    sam = _pixel_sam_deg(pred, target, eps)[:, 1:-1, 1:-1].reshape(-1)
    r = risk[:, 1:-1, 1:-1].reshape(-1)
    valid = torch.isfinite(sam) & torch.isfinite(r)
    sam = sam[valid]
    r = r[valid]
    lo = torch.quantile(r, fraction)
    hi = torch.quantile(r, 1.0 - fraction)
    return float(sam[r >= hi].mean().item()), float(sam[r <= lo].mean().item())


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _relative_norm(x: torch.Tensor, ref: torch.Tensor, eps: float) -> float:
    return float(
        (
            torch.linalg.vector_norm(x.float())
            / torch.linalg.vector_norm(ref.float()).clamp_min(eps)
        ).item()
    )


@torch.no_grad()
def main():
    args = parse_args()
    if args.alpha0 <= 0.0:
        raise ValueError("--alpha0 must be > 0")
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    betas = _parse_betas(args.betas)
    norm_match = not args.no_norm_match

    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    _, test_loader, info = build_loaders(cfg)
    process = build_progressive_process(cfg)
    baseline = _build_baseline(args, info, device)

    projector = HeterogeneityGuidedSpectralRefiner(
        n_bands=info["n_bands"],
        total_steps=args.diffusion_steps,
        hidden_channels=1,
        variant="full",
        eps=args.eps,
    ).to(device)
    projector.eval()
    for parameter in projector.parameters():
        parameter.requires_grad_(False)

    baseline_meter = MetricAverager()
    meters: Dict[float, MetricAverager] = {beta: MetricAverager() for beta in betas}
    regional = {
        beta: {
            "SAM_HIGH": [],
            "SAM_LOW": [],
            "UPDATE_NORM": [],
            "NORM_SCALE": [],
            "GAIN_HIGH": [],
            "GAIN_LOW": [],
        }
        for beta in betas
    }
    baseline_high: List[float] = []
    baseline_low: List[float] = []

    print(
        "DIAGNOSTIC innovation3_stage_B2 "
        f"dataset={args.dataset} alpha0={args.alpha0} betas={betas} "
        f"norm_match={norm_match} C4L=[4,{projector.low_end - 1}]"
    )
    print(
        "gain=alpha0*[1+beta*(2*H_M_rank-1)]; beta>0 favors high heterogeneity; "
        "beta<0 is anti-heterogeneity control; GT=evaluation_only"
    )

    for case_index, batch in enumerate(test_loader):
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)

        terminal_lr = process.terminal_observation(gt)
        base_pred = reconstruct_from_terminal_lr(
            baseline,
            process,
            terminal_lr,
            target_size=tuple(gt.shape[-2:]),
            hr_msi=hr_msi,
        )

        pred_terminal = process.terminal_observation(base_pred)
        native_residual = terminal_lr - pred_terminal
        terminal_lift = process.terminal_state(
            native_residual,
            target_size=tuple(gt.shape[-2:]),
        )
        broad = projector.project_broadshape(terminal_lift)
        r_phy = projector.project_tangent(
            broad,
            base_pred,
            preserve_broadshape=True,
        )

        risk = ranked_msi_heterogeneity(hr_msi, eps=args.eps)
        baseline_meter.update(calc_metrics(base_pred, gt, args.scale_ratio))
        h0, l0 = _region_sam(base_pred, gt, risk, args.region_fraction, args.eps)
        baseline_high.append(h0)
        baseline_low.append(l0)

        reference_update = float(args.alpha0) * r_phy
        reference_energy = torch.linalg.vector_norm(reference_update.float()).clamp_min(args.eps)

        for beta in betas:
            gain = float(args.alpha0) * (1.0 + float(beta) * (2.0 * risk - 1.0))
            raw_update = gain.unsqueeze(1) * r_phy

            scale = 1.0
            if norm_match:
                raw_energy = torch.linalg.vector_norm(raw_update.float()).clamp_min(args.eps)
                scale = float((reference_energy / raw_energy).item())
            update = raw_update * scale
            candidate = base_pred + update

            meters[beta].update(calc_metrics(candidate, gt, args.scale_ratio))
            high_sam, low_sam = _region_sam(
                candidate, gt, risk, args.region_fraction, args.eps
            )
            regional[beta]["SAM_HIGH"].append(high_sam)
            regional[beta]["SAM_LOW"].append(low_sam)
            regional[beta]["UPDATE_NORM"].append(
                _relative_norm(update, base_pred, args.eps)
            )
            regional[beta]["NORM_SCALE"].append(scale)

            interior = risk[:, 1:-1, 1:-1].reshape(-1)
            hi = torch.quantile(interior, 1.0 - args.region_fraction)
            lo = torch.quantile(interior, args.region_fraction)
            gain_eff = gain[:, 1:-1, 1:-1].reshape(-1) * scale
            regional[beta]["GAIN_HIGH"].append(float(gain_eff[interior >= hi].mean().item()))
            regional[beta]["GAIN_LOW"].append(float(gain_eff[interior <= lo].mean().item()))

        print(f"CASE index={case_index} done")

    base = baseline_meter.average()
    base_high = _mean(baseline_high)
    base_low = _mean(baseline_low)

    print("=" * 154)
    print(
        "BETA        PSNR          SAM     SAM_HIGH      SAM_LOW    dPSNR    dSAM_ALL   "
        "dSAM_HIGH    dSAM_LOW   UPDATE_NORM   NORM_SCALE   GAIN_HIGH    GAIN_LOW"
    )

    rows = []
    for beta in betas:
        metrics = meters[beta].average()
        high = _mean(regional[beta]["SAM_HIGH"])
        low = _mean(regional[beta]["SAM_LOW"])
        row = {
            "beta": float(beta),
            **metrics,
            "SAM_HIGH": high,
            "SAM_LOW": low,
            "dPSNR": metrics["PSNR"] - base["PSNR"],
            "dSAM": metrics["SAM"] - base["SAM"],
            "dSAM_HIGH": high - base_high,
            "dSAM_LOW": low - base_low,
            "UPDATE_NORM": _mean(regional[beta]["UPDATE_NORM"]),
            "NORM_SCALE": _mean(regional[beta]["NORM_SCALE"]),
            "GAIN_HIGH": _mean(regional[beta]["GAIN_HIGH"]),
            "GAIN_LOW": _mean(regional[beta]["GAIN_LOW"]),
        }
        rows.append(row)
        print(
            f"{beta:+.3f} "
            f"{metrics['PSNR']:11.6f} {metrics['SAM']:12.6f} "
            f"{high:12.6f} {low:12.6f} "
            f"{row['dPSNR']:+9.6f} {row['dSAM']:+11.6f} "
            f"{row['dSAM_HIGH']:+12.6f} {row['dSAM_LOW']:+11.6f} "
            f"{row['UPDATE_NORM']:13.8f} {row['NORM_SCALE']:12.8f} "
            f"{row['GAIN_HIGH']:12.6f} {row['GAIN_LOW']:11.6f}"
        )

    zero = min(rows, key=lambda row: abs(row["beta"]))
    positive = [row for row in rows if row["beta"] > 0.0]
    negative = [row for row in rows if row["beta"] < 0.0]
    best_positive = min(positive, key=lambda row: row["SAM"]) if positive else None
    best_negative = min(negative, key=lambda row: row["SAM"]) if negative else None
    best_overall = min(rows, key=lambda row: row["SAM"])

    print("=" * 154)
    print(
        "CONSTANT_GAIN "
        f"beta={zero['beta']:+.3f} PSNR={zero['PSNR']:.6f} SAM={zero['SAM']:.6f}"
    )
    if best_positive is not None:
        print(
            "BEST_HETERO "
            f"beta={best_positive['beta']:+.3f} PSNR={best_positive['PSNR']:.6f} "
            f"SAM={best_positive['SAM']:.6f} "
            f"vs_constant_dSAM={best_positive['SAM']-zero['SAM']:+.6f} "
            f"vs_constant_dPSNR={best_positive['PSNR']-zero['PSNR']:+.6f}"
        )
    if best_negative is not None:
        print(
            "BEST_ANTI_HETERO "
            f"beta={best_negative['beta']:+.3f} PSNR={best_negative['PSNR']:.6f} "
            f"SAM={best_negative['SAM']:.6f} "
            f"vs_constant_dSAM={best_negative['SAM']-zero['SAM']:+.6f}"
        )
    print(
        "BEST_OVERALL "
        f"beta={best_overall['beta']:+.3f} PSNR={best_overall['PSNR']:.6f} "
        f"SAM={best_overall['SAM']:.6f}"
    )

    payload = {
        "conditions": {
            "stage": "Innovation-3 Stage-B2",
            "dataset": args.dataset,
            "baseline_checkpoint": args.baseline_checkpoint,
            "baseline_checkpoint_type": args.baseline_checkpoint_type,
            "alpha0": args.alpha0,
            "betas": list(betas),
            "norm_match": norm_match,
            "test_size": args.test_size,
            "scale_ratio": args.scale_ratio,
            "diffusion_steps": args.diffusion_steps,
            "region_fraction": args.region_fraction,
            "c4l_range": [4, projector.low_end - 1],
            "seed": args.seed,
            "training": "none",
            "gt_usage": "evaluation only",
        },
        "definitions": {
            "gain": "alpha0*[1+beta*(2*H_M_rank-1)]",
            "beta_positive": "more gain on high heterogeneity and less on low heterogeneity",
            "beta_negative": "anti-heterogeneity control",
            "norm_match": "per-case L2 update energy matched to beta=0" if norm_match else "disabled",
            "R_TERM": "U_T[Y_H-D_T(X_hat)]",
            "R_PHY": "P_Tu P_C4L(R_TERM)",
        },
        "baseline_raw_direct": {
            **base,
            "SAM_HIGH": base_high,
            "SAM_LOW": base_low,
        },
        "rows": rows,
        "constant_gain": zero,
        "best_heterogeneity": best_positive,
        "best_anti_heterogeneity": best_negative,
        "best_overall": best_overall,
    }

    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
