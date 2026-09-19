"""Innovation-3 Stage-B0.5: zero-shot physical-residual causal injection.

This diagnostic tests whether the observable terminal HSI residual can *directly*
reduce the final Raw-Direct spectral-angle error before introducing any new
trainable module or MSI heterogeneity gate.

Baseline:
    Raw-Direct full reverse diffusion -> X_hat

Observable terminal residual:
    R_term = U_T[ Y_H - D_T(X_hat) ]

Allowed correction direction:
    R_phy = P_{T_u} P_{C4:L}(R_term),
    u = X_hat / ||X_hat||_2

Zero-shot intervention:
    X_hat(alpha) = X_hat + alpha * R_phy

The script sweeps signed scalar alpha values.  Positive-vs-negative behavior is
important causal evidence: a useful residual direction should improve SAM for
some positive alpha while the opposite sign should not produce the same
behavior.

No GT information is used to construct R_phy.  GT is used only for evaluation.
No network is trained, and no H_M gate is applied in this stage.
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


DEFAULT_ALPHAS = (-0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 zero-shot physical-residual causal-injection diagnostic"
    )
    p.add_argument(
        "--dataset",
        choices=["PaviaU", "Houston13", "Chikusei"],
        default="PaviaU",
    )
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
    p.add_argument(
        "--alphas",
        default=",".join(str(v) for v in DEFAULT_ALPHAS),
        help="Comma-separated signed injection strengths.",
    )
    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument(
        "--output_json",
        default="./results/innovation3_physical_residual_injection_PaviaU.json",
    )
    return p.parse_args()


def _parse_alphas(text: str) -> Tuple[float, ...]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("--alphas is empty")
    if not any(abs(v) < 1e-12 for v in values):
        values.append(0.0)

    # Preserve user order while removing exact duplicates.
    unique: List[float] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


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
            model,
            args.baseline_checkpoint,
            map_location=str(device),
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


def _pixel_sam_deg(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    dot = (pred.float() * target.float()).sum(dim=1)
    pn = torch.linalg.vector_norm(pred.float(), dim=1)
    tn = torch.linalg.vector_norm(target.float(), dim=1)
    cos = dot / (pn * tn).clamp_min(eps)
    return torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * (180.0 / math.pi)


def _region_masks(
    risk: torch.Tensor,
    fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top/bottom heterogeneity masks on the one-pixel interior."""
    r = risk[:, 1:-1, 1:-1].reshape(-1)
    valid = torch.isfinite(r)
    rv = r[valid]
    if rv.numel() == 0:
        raise RuntimeError("no finite heterogeneity values")
    lo = torch.quantile(rv, fraction)
    hi = torch.quantile(rv, 1.0 - fraction)
    return r >= hi, r <= lo


def _region_sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    risk: torch.Tensor,
    fraction: float,
) -> Tuple[float, float]:
    sam = _pixel_sam_deg(pred, target)
    sam = sam[:, 1:-1, 1:-1].reshape(-1)
    r = risk[:, 1:-1, 1:-1].reshape(-1)

    valid = torch.isfinite(sam) & torch.isfinite(r)
    sam = sam[valid]
    r = r[valid]
    lo = torch.quantile(r, fraction)
    hi = torch.quantile(r, 1.0 - fraction)

    high = sam[r >= hi]
    low = sam[r <= lo]
    return float(high.mean().item()), float(low.mean().item())


def _relative_update_norm(
    update: torch.Tensor,
    base: torch.Tensor,
    eps: float,
) -> float:
    numerator = torch.linalg.vector_norm(update.float()).item()
    denominator = torch.linalg.vector_norm(base.float()).clamp_min(eps).item()
    return float(numerator / denominator)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


@torch.no_grad()
def main():
    args = parse_args()
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    alphas = _parse_alphas(args.alphas)

    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    _, test_loader, info = build_loaders(cfg)
    process = build_progressive_process(cfg)
    baseline = _build_baseline(args, info, device)

    # Reuse the exact C4:L and broad-preserving tangent projections of the
    # first Innovation-3 Full refiner, without using its learned generator/gate.
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

    print(
        "DIAGNOSTIC innovation3_stage_B0.5 zero_shot_physical_residual_injection "
        f"dataset={args.dataset} alphas={alphas} "
        f"C4L=[4,{projector.low_end - 1}] region_fraction={args.region_fraction}"
    )
    print(
        "R_TERM=U_T[Y_H-D_T(X_hat)]; "
        "R_PHY=P_Tu P_C4L(R_TERM); "
        "X_alpha=X_hat+alpha*R_PHY; no_training no_HM_gate"
    )

    metric_meters: Dict[float, MetricAverager] = {
        alpha: MetricAverager() for alpha in alphas
    }
    regional: Dict[float, Dict[str, List[float]]] = {
        alpha: {"SAM_HIGH": [], "SAM_LOW": [], "UPDATE_NORM": []}
        for alpha in alphas
    }
    residual_norms: List[float] = []

    for batch_index, batch in enumerate(test_loader):
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

        # Inference-visible physical residual only.
        pred_terminal = process.terminal_observation(base_pred)
        terminal_native_residual = terminal_lr - pred_terminal
        terminal_lift = process.terminal_state(
            terminal_native_residual,
            target_size=tuple(gt.shape[-2:]),
        )
        broad = projector.project_broadshape(terminal_lift)
        r_phy = projector.project_tangent(
            broad,
            base_pred,
            preserve_broadshape=True,
        )

        risk = ranked_msi_heterogeneity(hr_msi, eps=args.eps)
        residual_norms.append(
            _relative_update_norm(r_phy, base_pred, args.eps)
        )

        for alpha in alphas:
            update = float(alpha) * r_phy
            candidate = base_pred + update

            metric_meters[alpha].update(
                calc_metrics(candidate, gt, args.scale_ratio)
            )
            high_sam, low_sam = _region_sam(
                candidate,
                gt,
                risk,
                args.region_fraction,
            )
            regional[alpha]["SAM_HIGH"].append(high_sam)
            regional[alpha]["SAM_LOW"].append(low_sam)
            regional[alpha]["UPDATE_NORM"].append(
                _relative_update_norm(update, base_pred, args.eps)
            )

        print(
            f"CASE index={batch_index} "
            f"RPHY_REL_NORM={residual_norms[-1]:.8f}"
        )

    rows: Dict[float, Dict[str, float]] = {}
    for alpha in alphas:
        metrics = metric_meters[alpha].average()
        rows[alpha] = {
            **metrics,
            "SAM_HIGH": _mean(regional[alpha]["SAM_HIGH"]),
            "SAM_LOW": _mean(regional[alpha]["SAM_LOW"]),
            "UPDATE_NORM": _mean(regional[alpha]["UPDATE_NORM"]),
        }

    zero_alpha = min(alphas, key=lambda v: abs(v))
    zero = rows[zero_alpha]
    print("=" * 132)
    print(
        "BASELINE_ALPHA0 "
        f"PSNR={zero['PSNR']:.6f} "
        f"SAM={zero['SAM']:.6f} "
        f"SAM_HIGH={zero['SAM_HIGH']:.6f} "
        f"SAM_LOW={zero['SAM_LOW']:.6f}"
    )
    print("-" * 132)
    print(
        "ALPHA       PSNR          SAM     SAM_HIGH      SAM_LOW   "
        "dSAM_ALL    dSAM_HIGH     dSAM_LOW   BENEFIT_RATIO   UPDATE_NORM"
    )

    table = []
    for alpha in alphas:
        row = rows[alpha]
        d_all = row["SAM"] - zero["SAM"]
        d_high = row["SAM_HIGH"] - zero["SAM_HIGH"]
        d_low = row["SAM_LOW"] - zero["SAM_LOW"]
        benefit = abs(d_high) / (abs(d_low) + args.eps)
        enriched = {
            "alpha": float(alpha),
            **row,
            "dSAM_ALL": float(d_all),
            "dSAM_HIGH": float(d_high),
            "dSAM_LOW": float(d_low),
            "BENEFIT_RATIO": float(benefit),
        }
        table.append(enriched)
        print(
            f"{alpha:+.3f} "
            f"{row['PSNR']:11.6f} "
            f"{row['SAM']:12.6f} "
            f"{row['SAM_HIGH']:12.6f} "
            f"{row['SAM_LOW']:12.6f} "
            f"{d_all:+11.6f} "
            f"{d_high:+12.6f} "
            f"{d_low:+12.6f} "
            f"{benefit:15.6f} "
            f"{row['UPDATE_NORM']:13.8f}"
        )

    positive = [row for row in table if row["alpha"] > 0.0]
    negative = [row for row in table if row["alpha"] < 0.0]
    best_positive = min(positive, key=lambda row: row["SAM"]) if positive else None
    best_negative = min(negative, key=lambda row: row["SAM"]) if negative else None

    print("=" * 132)
    if best_positive is not None:
        print(
            "BEST_POSITIVE "
            f"alpha={best_positive['alpha']:+.6f} "
            f"SAM={best_positive['SAM']:.6f} "
            f"dSAM={best_positive['dSAM_ALL']:+.6f} "
            f"dSAM_HIGH={best_positive['dSAM_HIGH']:+.6f} "
            f"dSAM_LOW={best_positive['dSAM_LOW']:+.6f} "
            f"PSNR={best_positive['PSNR']:.6f}"
        )
    if best_negative is not None:
        print(
            "BEST_NEGATIVE "
            f"alpha={best_negative['alpha']:+.6f} "
            f"SAM={best_negative['SAM']:.6f} "
            f"dSAM={best_negative['dSAM_ALL']:+.6f} "
            f"dSAM_HIGH={best_negative['dSAM_HIGH']:+.6f} "
            f"dSAM_LOW={best_negative['dSAM_LOW']:+.6f} "
            f"PSNR={best_negative['PSNR']:.6f}"
        )

    payload = {
        "conditions": {
            "stage": "Innovation-3 Stage-B0.5",
            "dataset": args.dataset,
            "baseline_checkpoint": args.baseline_checkpoint,
            "baseline_checkpoint_type": args.baseline_checkpoint_type,
            "test_size": args.test_size,
            "scale_ratio": args.scale_ratio,
            "srf_interp": args.srf_interp,
            "diffusion_steps": args.diffusion_steps,
            "mtf_nyquist": args.mtf_nyquist,
            "psf_truncate": args.psf_truncate,
            "region_fraction": args.region_fraction,
            "c4l_range": [4, projector.low_end - 1],
            "alphas": list(alphas),
            "seed": args.seed,
            "gt_usage": "evaluation only; correction uses observable LR-HSI and reconstructed HSI",
            "training": "none",
            "heterogeneity_gate": "none; H_M is used only to stratify evaluation regions",
        },
        "definitions": {
            "R_TERM": "U_T[Y_H-D_T(X_hat)]",
            "R_PHY": "P_Tu P_C4L(R_TERM)",
            "INTERVENTION": "X_hat(alpha)=X_hat+alpha*R_PHY",
            "SAM_HIGH": "top region_fraction of inference-visible MSI heterogeneity rank",
            "SAM_LOW": "bottom region_fraction of inference-visible MSI heterogeneity rank",
            "UPDATE_NORM": "||alpha*R_PHY||_2 / ||X_hat||_2",
            "BENEFIT_RATIO": "|dSAM_HIGH|/(|dSAM_LOW|+eps)",
        },
        "baseline_alpha0": zero,
        "mean_rphy_relative_norm": _mean(residual_norms),
        "sweep": table,
        "best_positive": best_positive,
        "best_negative": best_negative,
    }

    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
