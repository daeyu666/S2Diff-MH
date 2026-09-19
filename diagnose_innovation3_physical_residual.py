"""Diagnose whether observable physical HSI residuals provide the missing correction direction.

The first Innovation-3 Full experiment showed that MSI heterogeneity localizes
where to refine, while the learned HSI-only residual branch yields only a very
small SAM improvement.  This diagnostic tests two inference-visible physical
residual candidates before designing another trainable module:

  TERM:
      U_T( Y_H - D_T(x0_hat) )

  STATE:
      x_t - D~_t(x0_hat)

TERM uses the fixed observed LR-HSI at the terminal sensor degradation.
STATE uses the current reverse-diffusion state and the calibrated progressive
forward operator.  Both candidates are projected into the same C4:L+tangent
space as the current Full refiner, then compared with the GT-only oracle
spectral-angle descent direction.

GT is used only for this diagnostic alignment measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Tuple

import torch

from config import TrainConfig
from data_loader import build_loaders
from innovation1 import build_progressive_process
from models import (
    HeterogeneityGuidedSpectralPredictor,
    RawMSIDirectPredictor,
    load_legacy_raw_direct_checkpoint,
    ranked_msi_heterogeneity,
)
from utils import ensure_dir, get_device, load_checkpoint, set_seed
from diagnose_innovation3_oracle_capacity import oracle_projection_components


CANDIDATES = ("TERM", "STATE", "LEARNED")


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 physical-residual correction-direction diagnostic"
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
    p.add_argument("--variant", choices=["full"], default="full")
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
        default="./results/innovation3_physical_residual_PaviaU.json",
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
        variant="full",
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
    if state.get("variant", "full") != "full":
        raise ValueError("physical-residual diagnostic requires a Full refiner checkpoint")
    model.refiner.load_state_dict(state["refiner"], strict=True)
    model.eval()
    print(
        f"REFINER_LOAD path={args.refiner_checkpoint} "
        f"epoch={state.get('epoch', 0)} best_SAM_HIGH={state.get('best_high_sam', float('nan'))}"
    )
    return model


def _interior_spectra(x: torch.Tensor) -> torch.Tensor:
    return (
        x[:, :, 1:-1, 1:-1]
        .permute(0, 2, 3, 1)
        .reshape(-1, x.shape[1])
        .float()
    )


def _cos(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    denom = (
        torch.linalg.vector_norm(a, dim=1)
        * torch.linalg.vector_norm(b, dim=1)
    ).clamp_min(eps)
    return (a * b).sum(dim=1) / denom


def _mean(x: torch.Tensor) -> float:
    x = x[torch.isfinite(x)]
    return float(x.mean().item()) if x.numel() else float("nan")


def _median(x: torch.Tensor) -> float:
    x = x[torch.isfinite(x)]
    return float(x.median().item()) if x.numel() else float("nan")


def _allowed_projection(refiner, residual: torch.Tensor, base_x0: torch.Tensor):
    broad = refiner.project_broadshape(residual)
    return refiner.project_tangent(
        broad,
        base_x0,
        preserve_broadshape=True,
    )


def _candidate_stats(
    candidate: torch.Tensor,
    oracle_allowed: torch.Tensor,
    ideal: torch.Tensor,
    risk: torch.Tensor,
    *,
    fraction: float,
    eps: float,
) -> Dict[str, float]:
    c = _interior_spectra(candidate)
    a = _interior_spectra(oracle_allowed)
    d = _interior_spectra(ideal)
    r = risk[:, 1:-1, 1:-1].reshape(-1).float()

    cn = torch.linalg.vector_norm(c, dim=1)
    an = torch.linalg.vector_norm(a, dim=1)
    dn = torch.linalg.vector_norm(d, dim=1)
    valid = (
        torch.isfinite(r)
        & torch.isfinite(cn)
        & torch.isfinite(an)
        & torch.isfinite(dn)
        & (cn > eps)
        & (an > eps)
        & (dn > eps)
    )
    align_allowed = torch.full_like(cn, float("nan"))
    align_ideal = torch.full_like(cn, float("nan"))
    align_allowed[valid] = _cos(c[valid], a[valid], eps)
    align_ideal[valid] = _cos(c[valid], d[valid], eps)

    rv = r[torch.isfinite(r)]
    lo = torch.quantile(rv, fraction)
    hi = torch.quantile(rv, 1.0 - fraction)
    high = r >= hi
    low = r <= lo

    def group(mask, prefix):
        m = valid & mask
        return {
            f"{prefix}ALIGN_ALLOWED": _mean(align_allowed[m]),
            f"{prefix}ALIGN_ALLOWED_MEDIAN": _median(align_allowed[m]),
            f"{prefix}ALIGN_IDEAL": _mean(align_ideal[m]),
            f"{prefix}POSITIVE_FRAC": (
                float((align_allowed[m] > 0.0).float().mean().item())
                if int(m.sum().item()) > 0
                else float("nan")
            ),
            f"{prefix}NORM": _mean(cn[m]),
        }

    out = {}
    out.update(group(torch.ones_like(valid, dtype=torch.bool), "ALL_"))
    out.update(group(high, "HIGH_"))
    out.update(group(low, "LOW_"))
    return out


def _print_candidate(t: int, name: str, s: Dict[str, float]):
    print(
        f"T={t:02d} {name} "
        f"ALIGN_ALLOWED all={s['ALL_ALIGN_ALLOWED']:+.6f} "
        f"high={s['HIGH_ALIGN_ALLOWED']:+.6f} "
        f"low={s['LOW_ALIGN_ALLOWED']:+.6f} "
        f"median={s['ALL_ALIGN_ALLOWED_MEDIAN']:+.6f} "
        f"pos={s['ALL_POSITIVE_FRAC']:.6f} "
        f"ALIGN_IDEAL={s['ALL_ALIGN_IDEAL']:+.6f} "
        f"NORM all={s['ALL_NORM']:.8f} high={s['HIGH_NORM']:.8f}"
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
        f"DIAGNOSTIC physical_residual_direction timesteps={timesteps} "
        f"C4L=[4,{model.refiner.low_end-1}] trajectory=actual_refined_reverse"
    )
    print(
        "TERM=U_T(Y_H-D_T(x0_hat)); "
        "STATE=x_t-D~_t(x0_hat); "
        "LEARNED=current trained Full pre-update"
    )

    per_t = {t: [] for t in timesteps}
    with torch.no_grad():
        for batch in test_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            risk = ranked_msi_heterogeneity(hr_msi, eps=args.eps)
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
                    oracle = oracle_projection_components(
                        model.refiner,
                        base_x0,
                        gt,
                        eps=args.eps,
                    )

                    pred_terminal = process.terminal_observation(base_x0)
                    terminal_native_residual = terminal_lr - pred_terminal
                    terminal_lift = process.terminal_state(
                        terminal_native_residual,
                        target_size=tuple(gt.shape[-2:]),
                    )
                    terminal_allowed = _allowed_projection(
                        model.refiner,
                        terminal_lift,
                        base_x0,
                    )

                    predicted_state = process.state_at(base_x0, t)
                    state_residual = x_t - predicted_state
                    state_allowed = _allowed_projection(
                        model.refiner,
                        state_residual,
                        base_x0,
                    )

                    candidates = {
                        "TERM": terminal_allowed,
                        "STATE": state_allowed,
                        "LEARNED": details["pre_update"],
                    }
                    row = {}
                    for name, candidate in candidates.items():
                        row[name] = _candidate_stats(
                            candidate,
                            oracle["allowed"],
                            oracle["ideal"],
                            risk,
                            fraction=args.region_fraction,
                            eps=args.eps,
                        )
                    per_t[t].append(row)

                x_t = process.reverse_update(x_t, refined_x0, t)

    results = {}
    for t in timesteps:
        rows = per_t[t]
        stage = {}
        print("=" * 132)
        for name in CANDIDATES:
            keys = rows[0][name].keys()
            summary = {}
            for key in keys:
                values = torch.tensor(
                    [row[name][key] for row in rows],
                    dtype=torch.float64,
                )
                summary[key] = _mean(values)
            stage[name] = summary
            _print_candidate(t, name, summary)

        best = max(
            CANDIDATES,
            key=lambda name: stage[name]["HIGH_ALIGN_ALLOWED"],
        )
        print(
            f"T={t:02d} BEST_HIGH_ALIGN={best} "
            f"value={stage[best]['HIGH_ALIGN_ALLOWED']:+.6f}"
        )
        results[str(t)] = stage

    print("=" * 132)
    for name in CANDIDATES:
        values = torch.tensor(
            [results[str(t)][name]["HIGH_ALIGN_ALLOWED"] for t in timesteps],
            dtype=torch.float64,
        )
        print(
            f"CONSISTENCY {name} "
            f"mean_HIGH_ALIGN_ALLOWED={_mean(values):+.6f} "
            f"positive_steps={int((values > 0).sum().item())}/{len(timesteps)}"
        )

    payload = {
        "conditions": {
            "dataset": args.dataset,
            "timesteps": list(timesteps),
            "baseline_checkpoint": args.baseline_checkpoint,
            "refiner_checkpoint": args.refiner_checkpoint,
            "region_fraction": args.region_fraction,
            "trajectory": "actual refined reverse diffusion trajectory",
            "c4l_range": [4, model.refiner.low_end - 1],
            "gt_usage": "diagnostic oracle alignment only",
            "seed": args.seed,
        },
        "definitions": {
            "TERM": "P_allowed U_T(Y_H-D_T(x0_hat))",
            "STATE": "P_allowed (x_t-D~_t(x0_hat))",
            "LEARNED": "trained Full pre-update",
            "ALIGN_ALLOWED": "cos(candidate, oracle allowed correction)",
            "ALIGN_IDEAL": "cos(candidate, unconstrained ideal tangent direction)",
        },
        "timesteps": results,
    }
    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
