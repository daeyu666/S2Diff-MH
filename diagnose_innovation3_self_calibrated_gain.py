"""Innovation-3 Stage-B3: self-calibrated terminal physical spectral residual.

Stage-B0.6 found a useful final-output physical residual direction and a GT-tuned
scalar optimum near alpha=4 on the current PaviaU diagnostic. Stage-B2 showed
that MSI heterogeneity-based spatial gain redistribution does not improve the
constant-gain result.

This stage removes the GT-tuned scalar gain. For the final Raw-Direct estimate
X_hat, define the observable terminal residual

    e = Y_H - D_T(X_hat)

and the allowed HR spectral correction

    r = P_{T_u} P_{C4:L} U_T(e).

Because D_T is linear, the scalar gain that minimizes terminal observation
closure along r has the closed form

    alpha_phys = <e, D_T(r)> / ||D_T(r)||^2.

By default alpha_phys is constrained to be non-negative (one-dimensional
non-negative least squares). No GT information enters alpha_phys. GT is used
only for evaluation.

The script compares:
    baseline
    fixed alpha (default 4.0; diagnostic reference only)
    self-calibrated alpha_phys
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 self-calibrated terminal physical spectral residual"
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
        "--fixed_alpha",
        type=float,
        default=4.0,
        help="GT-sweep reference only; not part of the self-calibrated method.",
    )
    p.add_argument(
        "--allow_negative_alpha",
        action="store_true",
        help="Disable the default alpha_phys>=0 non-negative constraint.",
    )
    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument(
        "--output_json",
        default="./results/innovation3_self_calibrated_gain_PaviaU.json",
    )
    return p.parse_args()


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


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    x = torch.tensor(values, dtype=torch.float64)
    return float(x.std(unbiased=False).item())


def _closure_norm(residual: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(residual.float()).item())


def _physical_direction(process, projector, terminal_lr, base_pred):
    pred_terminal = process.terminal_observation(base_pred)
    e = terminal_lr - pred_terminal
    terminal_lift = process.terminal_state(
        e,
        target_size=tuple(base_pred.shape[-2:]),
    )
    broad = projector.project_broadshape(terminal_lift)
    r_phy = projector.project_tangent(
        broad,
        base_pred,
        preserve_broadshape=True,
    )
    return e, r_phy


def _self_calibrated_alpha(
    process,
    residual_native: torch.Tensor,
    r_phy: torch.Tensor,
    *,
    nonnegative: bool,
    eps: float,
) -> Tuple[float, torch.Tensor]:
    q = process.terminal_observation(r_phy)
    numerator = (residual_native.float() * q.float()).sum()
    denominator = q.float().square().sum().clamp_min(eps)
    alpha = numerator / denominator
    if nonnegative:
        alpha = alpha.clamp_min(0.0)
    return float(alpha.item()), q


@torch.no_grad()
def main():
    args = parse_args()
    if args.fixed_alpha <= 0.0:
        raise ValueError("--fixed_alpha must be > 0")
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")

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
        eps=max(args.eps, 1e-8),
    ).to(device)
    projector.eval()
    for parameter in projector.parameters():
        parameter.requires_grad_(False)

    modes = ("baseline", "fixed", "selfcal")
    meters: Dict[str, MetricAverager] = {mode: MetricAverager() for mode in modes}
    regional: Dict[str, Dict[str, List[float]]] = {
        mode: {"SAM_HIGH": [], "SAM_LOW": []} for mode in modes
    }

    alphas: List[float] = []
    closure_before: List[float] = []
    closure_fixed: List[float] = []
    closure_selfcal: List[float] = []
    closure_reduction_fixed: List[float] = []
    closure_reduction_selfcal: List[float] = []

    print(
        "DIAGNOSTIC innovation3_stage_B3 self_calibrated_terminal_physical_residual "
        f"dataset={args.dataset} fixed_alpha={args.fixed_alpha} "
        f"nonnegative={not args.allow_negative_alpha} C4L=[4,{projector.low_end - 1}]"
    )
    print(
        "alpha_phys=<e,D_T(r)>/||D_T(r)||^2; "
        "e=Y_H-D_T(X_hat); r=P_Tu P_C4L U_T(e); GT=evaluation_only"
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

        e, r_phy = _physical_direction(process, projector, terminal_lr, base_pred)
        alpha_phys, q = _self_calibrated_alpha(
            process,
            e,
            r_phy,
            nonnegative=not args.allow_negative_alpha,
            eps=args.eps,
        )

        fixed_pred = base_pred + float(args.fixed_alpha) * r_phy
        selfcal_pred = base_pred + alpha_phys * r_phy
        candidates = {
            "baseline": base_pred,
            "fixed": fixed_pred,
            "selfcal": selfcal_pred,
        }

        risk = ranked_msi_heterogeneity(hr_msi, eps=max(args.eps, 1e-8))
        for mode, pred in candidates.items():
            meters[mode].update(calc_metrics(pred, gt, args.scale_ratio))
            h, l = _region_sam(
                pred, gt, risk, args.region_fraction, max(args.eps, 1e-8)
            )
            regional[mode]["SAM_HIGH"].append(h)
            regional[mode]["SAM_LOW"].append(l)

        c0 = _closure_norm(e)
        cf = _closure_norm(e - float(args.fixed_alpha) * q)
        cs = _closure_norm(e - alpha_phys * q)
        alphas.append(alpha_phys)
        closure_before.append(c0)
        closure_fixed.append(cf)
        closure_selfcal.append(cs)
        closure_reduction_fixed.append(1.0 - cf / max(c0, args.eps))
        closure_reduction_selfcal.append(1.0 - cs / max(c0, args.eps))

        print(
            f"CASE index={case_index} alpha_phys={alpha_phys:.8f} "
            f"closure_before={c0:.8e} closure_fixed={cf:.8e} "
            f"closure_selfcal={cs:.8e} "
            f"selfcal_reduction={100.0*closure_reduction_selfcal[-1]:.4f}%"
        )

    rows = {}
    base_metrics = meters["baseline"].average()
    base_high = _mean(regional["baseline"]["SAM_HIGH"])
    base_low = _mean(regional["baseline"]["SAM_LOW"])

    print("=" * 142)
    print(
        "MODE          PSNR          SAM     SAM_HIGH      SAM_LOW    "
        "dPSNR    dSAM_ALL   dSAM_HIGH    dSAM_LOW"
    )
    for mode in modes:
        metrics = meters[mode].average()
        high = _mean(regional[mode]["SAM_HIGH"])
        low = _mean(regional[mode]["SAM_LOW"])
        row = {
            **metrics,
            "SAM_HIGH": high,
            "SAM_LOW": low,
            "dPSNR": metrics["PSNR"] - base_metrics["PSNR"],
            "dSAM": metrics["SAM"] - base_metrics["SAM"],
            "dSAM_HIGH": high - base_high,
            "dSAM_LOW": low - base_low,
        }
        rows[mode] = row
        print(
            f"{mode:<12} "
            f"{metrics['PSNR']:11.6f} {metrics['SAM']:12.6f} "
            f"{high:12.6f} {low:12.6f} "
            f"{row['dPSNR']:+9.6f} {row['dSAM']:+11.6f} "
            f"{row['dSAM_HIGH']:+12.6f} {row['dSAM_LOW']:+11.6f}"
        )

    print("=" * 142)
    print(
        "ALPHA_PHYS "
        f"mean={_mean(alphas):.8f} std={_std(alphas):.8f} "
        f"min={min(alphas):.8f} max={max(alphas):.8f}"
    )
    print(
        "CLOSURE "
        f"before={_mean(closure_before):.8e} "
        f"fixed={_mean(closure_fixed):.8e} "
        f"selfcal={_mean(closure_selfcal):.8e} "
        f"fixed_reduction={100.0*_mean(closure_reduction_fixed):.4f}% "
        f"selfcal_reduction={100.0*_mean(closure_reduction_selfcal):.4f}%"
    )
    print(
        "SELFCAL_VS_FIXED "
        f"dPSNR={rows['selfcal']['PSNR']-rows['fixed']['PSNR']:+.6f} "
        f"dSAM={rows['selfcal']['SAM']-rows['fixed']['SAM']:+.6f}"
    )

    payload = {
        "conditions": {
            "stage": "Innovation-3 Stage-B3",
            "dataset": args.dataset,
            "baseline_checkpoint": args.baseline_checkpoint,
            "baseline_checkpoint_type": args.baseline_checkpoint_type,
            "fixed_alpha_reference": args.fixed_alpha,
            "nonnegative_alpha": not args.allow_negative_alpha,
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
            "e": "Y_H-D_T(X_hat)",
            "r": "P_Tu P_C4L U_T(e)",
            "alpha_phys": "<e,D_T(r)>/||D_T(r)||^2",
            "selfcal_output": "X_hat+alpha_phys*r",
            "fixed_output": "X_hat+fixed_alpha*r",
        },
        "rows": rows,
        "alpha_phys": {
            "per_case": alphas,
            "mean": _mean(alphas),
            "std": _std(alphas),
            "min": min(alphas),
            "max": max(alphas),
        },
        "closure": {
            "before_mean": _mean(closure_before),
            "fixed_mean": _mean(closure_fixed),
            "selfcal_mean": _mean(closure_selfcal),
            "fixed_reduction_mean": _mean(closure_reduction_fixed),
            "selfcal_reduction_mean": _mean(closure_reduction_selfcal),
        },
    }

    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
