"""Innovation-3 Stage-B1: in-loop physical spectral residual diagnostic.

This script compares the zero-shot final-output correction validated in Stage-B0.6
against injecting the same observable physical spectral residual into the
predicted x0 used by the reverse diffusion trajectory.

Observable residual for each predicted clean HSI:
    R_term(x0_hat) = U_T[ Y_H - D_T(x0_hat) ]

Allowed spectral correction:
    R_phy = P_{T_u} P_{C4:L}(R_term)

Modes:
    baseline       : no physical residual correction
    final_only     : standard reverse trajectory, then one correction at output
    inloop_late4   : correct predicted x0 only for reverse steps t<=4
    inloop_late8   : correct predicted x0 only for reverse steps t<=8
    inloop_all     : correct predicted x0 at every reverse step

No new network is trained. GT is used only for evaluation. HR-MSI heterogeneity is
used only to stratify high/low evaluation regions, never to construct updates.
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
from innovation1 import build_progressive_process, model_predict
from metrics import MetricAverager, calc_metrics
from models import (
    HeterogeneityGuidedSpectralRefiner,
    RawMSIDirectPredictor,
    load_legacy_raw_direct_checkpoint,
    ranked_msi_heterogeneity,
)
from utils import ensure_dir, get_device, load_checkpoint, set_seed


DEFAULT_ALPHAS = (1.0, 2.0, 4.0)
DEFAULT_MODES = ("final_only", "inloop_late4", "inloop_late8", "inloop_all")


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 Stage-B1 in-loop physical residual diagnostic"
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
    p.add_argument(
        "--alphas",
        default=",".join(str(v) for v in DEFAULT_ALPHAS),
        help="Comma-separated positive residual gains.",
    )
    p.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes among final_only,inloop_late4,inloop_late8,inloop_all.",
    )
    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument(
        "--output_json",
        default="./results/innovation3_physical_residual_inloop_PaviaU.json",
    )
    return p.parse_args()


def _parse_floats(text: str) -> Tuple[float, ...]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            value = float(token)
            if value <= 0.0:
                raise ValueError("--alphas must contain positive values")
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError("--alphas is empty")
    return tuple(values)


def _parse_modes(text: str) -> Tuple[str, ...]:
    allowed = {"final_only", "inloop_late4", "inloop_late8", "inloop_all"}
    modes: List[str] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            if token not in allowed:
                raise ValueError(f"unknown mode={token!r}; allowed={sorted(allowed)}")
            if token not in modes:
                modes.append(token)
    if not modes:
        raise ValueError("--modes is empty")
    return tuple(modes)


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


def _physical_direction(
    process,
    projector,
    terminal_lr: torch.Tensor,
    x0_hat: torch.Tensor,
) -> torch.Tensor:
    pred_terminal = process.terminal_observation(x0_hat)
    native_residual = terminal_lr - pred_terminal
    terminal_lift = process.terminal_state(
        native_residual,
        target_size=tuple(x0_hat.shape[-2:]),
    )
    broad = projector.project_broadshape(terminal_lift)
    return projector.project_tangent(
        broad,
        x0_hat,
        preserve_broadshape=True,
    )


def _active_inloop(mode: str, t: int) -> bool:
    if mode == "inloop_all":
        return True
    if mode == "inloop_late8":
        return t <= 8
    if mode == "inloop_late4":
        return t <= 4
    return False


@torch.no_grad()
def _reconstruct(
    model,
    process,
    projector,
    terminal_lr: torch.Tensor,
    hr_msi: torch.Tensor,
    *,
    target_size: Tuple[int, int],
    mode: str,
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x_t = process.terminal_state(terminal_lr, target_size=target_size)
    applied_steps = 0
    update_ratios: List[float] = []

    for t in range(process.total_steps, 0, -1):
        timestep = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        base_x0 = model_predict(model, x_t, timestep, hr_msi)

        if _active_inloop(mode, t):
            r_phy = _physical_direction(process, projector, terminal_lr, base_x0)
            update = float(alpha) * r_phy
            denom = torch.linalg.vector_norm(base_x0.float()).clamp_min(1e-8)
            ratio = torch.linalg.vector_norm(update.float()) / denom
            update_ratios.append(float(ratio.item()))
            x0_for_reverse = base_x0 + update
            applied_steps += 1
        else:
            x0_for_reverse = base_x0

        x_t = process.reverse_update(x_t, x0_for_reverse, t)

    if mode == "final_only":
        r_phy = _physical_direction(process, projector, terminal_lr, x_t)
        update = float(alpha) * r_phy
        denom = torch.linalg.vector_norm(x_t.float()).clamp_min(1e-8)
        ratio = torch.linalg.vector_norm(update.float()) / denom
        update_ratios.append(float(ratio.item()))
        x_t = x_t + update
        applied_steps = 1

    diagnostics = {
        "APPLIED_STEPS": float(applied_steps),
        "MEAN_UPDATE_NORM": (
            float(sum(update_ratios) / len(update_ratios)) if update_ratios else 0.0
        ),
        "MAX_UPDATE_NORM": max(update_ratios) if update_ratios else 0.0,
    }
    return x_t, diagnostics


def _pixel_sam_deg(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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
) -> Tuple[float, float]:
    sam = _pixel_sam_deg(pred, target)
    sam = sam[:, 1:-1, 1:-1].reshape(-1)
    r = risk[:, 1:-1, 1:-1].reshape(-1)

    valid = torch.isfinite(sam) & torch.isfinite(r)
    sam = sam[valid]
    r = r[valid]
    lo = torch.quantile(r, fraction)
    hi = torch.quantile(r, 1.0 - fraction)
    return (
        float(sam[r >= hi].mean().item()),
        float(sam[r <= lo].mean().item()),
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


@torch.no_grad()
def main():
    args = parse_args()
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    alphas = _parse_floats(args.alphas)
    modes = _parse_modes(args.modes)

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

    keys = [("baseline", 0.0)]
    for mode in modes:
        for alpha in alphas:
            keys.append((mode, float(alpha)))

    meters = {key: MetricAverager() for key in keys}
    regional = {
        key: {"SAM_HIGH": [], "SAM_LOW": [], "MEAN_UPDATE_NORM": [], "MAX_UPDATE_NORM": []}
        for key in keys
    }

    print(
        "DIAGNOSTIC innovation3_stage_B1 "
        f"dataset={args.dataset} modes={modes} alphas={alphas} "
        f"C4L=[4,{projector.low_end - 1}]"
    )
    print(
        "R_TERM=U_T[Y_H-D_T(x0_hat)]; R_PHY=P_Tu P_C4L(R_TERM); "
        "GT=evaluation_only; H_M=stratification_only; no_training"
    )

    for case_index, batch in enumerate(test_loader):
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        terminal_lr = process.terminal_observation(gt)
        risk = ranked_msi_heterogeneity(hr_msi, eps=args.eps)

        for mode, alpha in keys:
            if mode == "baseline":
                pred, diag = _reconstruct(
                    baseline,
                    process,
                    projector,
                    terminal_lr,
                    hr_msi,
                    target_size=tuple(gt.shape[-2:]),
                    mode="none",
                    alpha=0.0,
                )
            else:
                pred, diag = _reconstruct(
                    baseline,
                    process,
                    projector,
                    terminal_lr,
                    hr_msi,
                    target_size=tuple(gt.shape[-2:]),
                    mode=mode,
                    alpha=alpha,
                )

            meters[(mode, alpha)].update(calc_metrics(pred, gt, args.scale_ratio))
            high_sam, low_sam = _region_sam(pred, gt, risk, args.region_fraction)
            regional[(mode, alpha)]["SAM_HIGH"].append(high_sam)
            regional[(mode, alpha)]["SAM_LOW"].append(low_sam)
            regional[(mode, alpha)]["MEAN_UPDATE_NORM"].append(diag["MEAN_UPDATE_NORM"])
            regional[(mode, alpha)]["MAX_UPDATE_NORM"].append(diag["MAX_UPDATE_NORM"])

        print(f"CASE index={case_index} done")

    rows = []
    baseline_metrics = meters[("baseline", 0.0)].average()
    baseline_high = _mean(regional[("baseline", 0.0)]["SAM_HIGH"])
    baseline_low = _mean(regional[("baseline", 0.0)]["SAM_LOW"])

    baseline_row = {
        "mode": "baseline",
        "alpha": 0.0,
        **baseline_metrics,
        "SAM_HIGH": baseline_high,
        "SAM_LOW": baseline_low,
        "dPSNR": 0.0,
        "dSAM": 0.0,
        "dSAM_HIGH": 0.0,
        "dSAM_LOW": 0.0,
        "MEAN_UPDATE_NORM": 0.0,
        "MAX_UPDATE_NORM": 0.0,
    }
    rows.append(baseline_row)

    print("=" * 144)
    print(
        "MODE            ALPHA       PSNR          SAM     SAM_HIGH      SAM_LOW    "
        "dPSNR    dSAM_ALL   dSAM_HIGH    dSAM_LOW   MEAN_UPD_NORM   MAX_UPD_NORM"
    )
    print(
        f"{'baseline':<15} {0.0:+.3f} "
        f"{baseline_metrics['PSNR']:11.6f} {baseline_metrics['SAM']:12.6f} "
        f"{baseline_high:12.6f} {baseline_low:12.6f} "
        f"{0.0:+9.6f} {0.0:+11.6f} {0.0:+12.6f} {0.0:+11.6f} "
        f"{0.0:15.8f} {0.0:14.8f}"
    )

    for mode in modes:
        for alpha in alphas:
            key = (mode, float(alpha))
            metrics = meters[key].average()
            high = _mean(regional[key]["SAM_HIGH"])
            low = _mean(regional[key]["SAM_LOW"])
            mean_upd = _mean(regional[key]["MEAN_UPDATE_NORM"])
            max_upd = _mean(regional[key]["MAX_UPDATE_NORM"])
            row = {
                "mode": mode,
                "alpha": float(alpha),
                **metrics,
                "SAM_HIGH": high,
                "SAM_LOW": low,
                "dPSNR": metrics["PSNR"] - baseline_metrics["PSNR"],
                "dSAM": metrics["SAM"] - baseline_metrics["SAM"],
                "dSAM_HIGH": high - baseline_high,
                "dSAM_LOW": low - baseline_low,
                "MEAN_UPDATE_NORM": mean_upd,
                "MAX_UPDATE_NORM": max_upd,
            }
            rows.append(row)
            print(
                f"{mode:<15} {alpha:+.3f} "
                f"{metrics['PSNR']:11.6f} {metrics['SAM']:12.6f} "
                f"{high:12.6f} {low:12.6f} "
                f"{row['dPSNR']:+9.6f} {row['dSAM']:+11.6f} "
                f"{row['dSAM_HIGH']:+12.6f} {row['dSAM_LOW']:+11.6f} "
                f"{mean_upd:15.8f} {max_upd:14.8f}"
            )

    candidates = [row for row in rows if row["mode"] != "baseline"]
    best_sam = min(candidates, key=lambda row: row["SAM"])
    best_psnr = max(candidates, key=lambda row: row["PSNR"])

    print("=" * 144)
    print(
        "BEST_SAM "
        f"mode={best_sam['mode']} alpha={best_sam['alpha']:+.6f} "
        f"PSNR={best_sam['PSNR']:.6f} SAM={best_sam['SAM']:.6f} "
        f"dPSNR={best_sam['dPSNR']:+.6f} dSAM={best_sam['dSAM']:+.6f} "
        f"dSAM_HIGH={best_sam['dSAM_HIGH']:+.6f} dSAM_LOW={best_sam['dSAM_LOW']:+.6f}"
    )
    print(
        "BEST_PSNR "
        f"mode={best_psnr['mode']} alpha={best_psnr['alpha']:+.6f} "
        f"PSNR={best_psnr['PSNR']:.6f} SAM={best_psnr['SAM']:.6f} "
        f"dPSNR={best_psnr['dPSNR']:+.6f} dSAM={best_psnr['dSAM']:+.6f}"
    )

    payload = {
        "conditions": {
            "stage": "Innovation-3 Stage-B1",
            "dataset": args.dataset,
            "baseline_checkpoint": args.baseline_checkpoint,
            "baseline_checkpoint_type": args.baseline_checkpoint_type,
            "test_size": args.test_size,
            "scale_ratio": args.scale_ratio,
            "diffusion_steps": args.diffusion_steps,
            "mtf_nyquist": args.mtf_nyquist,
            "psf_truncate": args.psf_truncate,
            "region_fraction": args.region_fraction,
            "c4l_range": [4, projector.low_end - 1],
            "alphas": list(alphas),
            "modes": list(modes),
            "seed": args.seed,
            "training": "none",
            "gt_usage": "evaluation only",
            "heterogeneity_usage": "evaluation stratification only",
        },
        "definitions": {
            "R_TERM": "U_T[Y_H-D_T(x0_hat)]",
            "R_PHY": "P_Tu P_C4L(R_TERM)",
            "final_only": "one correction after the standard reverse trajectory",
            "inloop_late4": "correct predicted x0 for reverse timesteps t<=4",
            "inloop_late8": "correct predicted x0 for reverse timesteps t<=8",
            "inloop_all": "correct predicted x0 at every reverse timestep",
        },
        "baseline": baseline_row,
        "rows": rows,
        "best_sam": best_sam,
        "best_psnr": best_psnr,
    }

    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
