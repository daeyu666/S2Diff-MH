"""Diagnose whether the spectral-risk regions found with GT-HSI are identifiable from inference-time inputs.

This is the final screening step before designing Innovation 3.  The diagnostic
uses the *actual reverse-diffusion trajectory*, not teacher-forced x_t states.

Observable maps are constructed only from quantities available during inference:

  H_M:
      3x3 local spectral heterogeneity of unit-normalized HR-MSI.

  H_t:
      3x3 local spectral heterogeneity of the current diffusion state x_t.

Because MSI and HSI live in different spectral dimensions, raw heterogeneity
magnitudes are not subtracted directly.  Each map is converted to a within-patch
rank in [0,1], then we define

  MISSING = relu(rank(H_M) - rank(H_t))
  DISAGREE = abs(rank(H_M) - rank(H_t))
  RISK = rank(H_M) * MISSING

RISK targets locations where MSI observes strong local material/spatial
variation while the current HSI state under-represents that variation.

GT-HSI is used only to compute SAM and diagnostic reference/controls.  It is
never used to construct H_M, H_t, MISSING, DISAGREE, or RISK.

At t in {1,3,6,9,12}, the script reports correlation and hard-SAM concentration
for every observable candidate.  It also accumulates RISK across the actual
reverse trajectory and tests whether the mean trajectory risk predicts the
FINAL reconstruction SAM.

The GT-derived local spectral heterogeneity from diagnose_boundary_mixedpixel.py
is retained only as a diagnostic ceiling/reference, not an inference input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from config import TrainConfig
from data_loader import build_loaders
from diagnose_boundary_mixedpixel import (
    _partial_spearman,
    spatial_material_proxies,
)
from diagnose_spectral_shape import _rankdata, _spearman
from innovation1 import build_progressive_process, model_predict
from main import build_model
from models import load_legacy_raw_direct_checkpoint
from utils import ensure_dir, get_device, load_checkpoint, set_seed


CANDIDATES = ("MSI_H", "XT_H", "MISSING", "DISAGREE", "RISK")


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 diagnostic: inference-observable heterogeneity risk"
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
    p.add_argument(
        "--checkpoint",
        default="./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth",
    )
    p.add_argument("--checkpoint_type", choices=["legacy", "standard"], default="legacy")

    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--proxy_quantile", type=float, default=0.25)
    p.add_argument("--hard_fraction", type=float, default=0.10)
    p.add_argument("--rho_threshold", type=float, default=0.20)
    p.add_argument("--enrichment_threshold", type=float, default=1.25)
    p.add_argument(
        "--output_json",
        default="./results/observable_heterogeneity_diagnostic_PaviaU.json",
    )
    return p.parse_args()


def _parse_timesteps(text: str, total_steps: int) -> Tuple[int, ...]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if not values:
        raise ValueError("--timesteps is empty")
    if len(values) != len(set(values)):
        raise ValueError("--timesteps contains duplicates")
    if any(v < 1 or v > total_steps for v in values):
        raise ValueError(f"timesteps must lie in [1,{total_steps}]")
    return values


def observable_heterogeneity(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """3x3 local spectral heterogeneity of unit-normalized spectra.

    Returns BxHxW.  One-pixel borders are NaN so all candidates share the same
    interior support as the GT diagnostic reference.
    """
    if x.ndim != 4:
        raise ValueError("x must be BxCxHxW")
    b, _, h, w = x.shape
    if h < 3 or w < 3:
        raise ValueError("need spatial size >=3x3")

    xf = x.float()
    norm = torch.linalg.vector_norm(xf, dim=1, keepdim=True)
    unit = xf / norm.clamp_min(eps)
    unit_pad = F.pad(unit, (1, 1, 1, 1), mode="reflect")
    mean = F.avg_pool2d(unit_pad, kernel_size=3, stride=1)
    mean_sq = F.avg_pool2d(unit_pad.square(), kernel_size=3, stride=1)
    var = (mean_sq - mean.square()).clamp_min(0.0)
    out = torch.sqrt(var.sum(dim=1).clamp_min(0.0))
    out[:, 0, :] = float("nan")
    out[:, -1, :] = float("nan")
    out[:, :, 0] = float("nan")
    out[:, :, -1] = float("nan")
    return out


def _rank01_map(x: torch.Tensor) -> torch.Tensor:
    """Rank finite values of BxHxW map to [0,1], preserving NaNs."""
    flat = x.reshape(-1)
    valid = torch.isfinite(flat)
    out = torch.full_like(flat, float("nan"), dtype=torch.float32)
    n = int(valid.sum().item())
    if n == 0:
        return out.reshape_as(x)
    if n == 1:
        out[valid] = 0.5
        return out.reshape_as(x)
    ranks = _rankdata(flat[valid]).float()
    out[valid] = ranks / float(n - 1)
    return out.reshape_as(x)


def observable_risk_maps(
    hr_msi: torch.Tensor,
    x_t: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Construct inference-only risk maps on the HR spatial grid."""
    hm = observable_heterogeneity(hr_msi, eps=eps)
    ht = observable_heterogeneity(x_t, eps=eps)
    rm = _rank01_map(hm)
    rt = _rank01_map(ht)
    missing = torch.relu(rm - rt)
    disagree = torch.abs(rm - rt)
    risk = rm * missing
    return {
        "MSI_H": rm.reshape(-1),
        "XT_H": rt.reshape(-1),
        "MISSING": missing.reshape(-1),
        "DISAGREE": disagree.reshape(-1),
        "RISK": risk.reshape(-1),
    }


def _pixel_sam(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.shape != gt.shape:
        raise ValueError("pred and gt shapes must match")
    p = pred.float()
    s = gt.float()
    dot = (p * s).sum(dim=1)
    pn = torch.linalg.vector_norm(p, dim=1)
    sn = torch.linalg.vector_norm(s, dim=1)
    cos = dot / (pn * sn).clamp_min(eps)
    sam = torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * (180.0 / math.pi)
    return sam.reshape(-1)


def _mean(x: torch.Tensor) -> float:
    return float(x.mean().item()) if x.numel() else float("nan")


def _ratio(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
    return a / b


def _top_overlap(a: torch.Tensor, b: torch.Tensor, fraction: float) -> Tuple[float, float]:
    """Return recall of b-high by a-high and Jaccard of high sets."""
    qa = torch.quantile(a, 1.0 - fraction)
    qb = torch.quantile(b, 1.0 - fraction)
    ha = a >= qa
    hb = b >= qb
    inter = (ha & hb).sum().float()
    recall = inter / hb.sum().clamp_min(1)
    union = (ha | hb).sum().float()
    jaccard = inter / union.clamp_min(1)
    return float(recall.item()), float(jaccard.item())


def summarize_candidate(
    proxy: torch.Tensor,
    sam: torch.Tensor,
    *,
    gt_heterogeneity: torch.Tensor,
    spectral_norm: torch.Tensor,
    intensity_gradient: torch.Tensor,
    valid: torch.Tensor,
    proxy_quantile: float,
    hard_fraction: float,
) -> Dict[str, float]:
    mask = (
        valid
        & torch.isfinite(proxy)
        & torch.isfinite(sam)
        & torch.isfinite(gt_heterogeneity)
        & torch.isfinite(spectral_norm)
        & torch.isfinite(intensity_gradient)
    )
    if int(mask.sum().item()) < 64:
        raise RuntimeError("too few valid pixels")

    p = proxy[mask]
    s = sam[mask]
    g = gt_heterogeneity[mask]
    nrm = spectral_norm[mask]
    igrad = intensity_gradient[mask]

    lo = p <= torch.quantile(p, proxy_quantile)
    hi = p >= torch.quantile(p, 1.0 - proxy_quantile)
    hard = s >= torch.quantile(s, 1.0 - hard_fraction)

    high_population = float(hi.float().mean().item())
    hard_in_high = float(hi[hard].float().mean().item())
    enrichment = _ratio(hard_in_high, high_population)
    gt_recall, gt_jaccard = _top_overlap(p, g, proxy_quantile)

    return {
        "n_valid": int(mask.sum().item()),
        "rho_proxy_sam": _spearman(p, s),
        "partial_rho_proxy_sam_given_norm_intensitygrad": _partial_spearman(
            p, s, [nrm, igrad]
        ),
        "sam_high": _mean(s[hi]),
        "sam_low": _mean(s[lo]),
        "delta_sam": _mean(s[hi]) - _mean(s[lo]),
        "sam_ratio": _ratio(_mean(s[hi]), _mean(s[lo])),
        "hard_in_high_fraction": hard_in_high,
        "high_population_fraction": high_population,
        "hard_concentration_enrichment": enrichment,
        "rho_proxy_gt_heterogeneity": _spearman(p, g),
        "gt_heterogeneity_high_recall": gt_recall,
        "gt_heterogeneity_high_jaccard": gt_jaccard,
    }


def _print_candidate(label: str, name: str, m: Dict[str, float]):
    print(
        f"{label} {name}_SAM "
        f"rho={m['rho_proxy_sam']:+.6f} "
        f"partial={m['partial_rho_proxy_sam_given_norm_intensitygrad']:+.6f} "
        f"high={m['sam_high']:.6f}deg low={m['sam_low']:.6f}deg "
        f"delta={m['delta_sam']:+.6f}deg ratio={m['sam_ratio']:.6f}"
    )
    print(
        f"{label} {name}_HARD "
        f"hard_in_high={m['hard_in_high_fraction']:.6f} "
        f"high_population={m['high_population_fraction']:.6f} "
        f"enrichment={m['hard_concentration_enrichment']:.6f}"
    )
    print(
        f"{label} {name}_GT_HETERO_LINK "
        f"rho={m['rho_proxy_gt_heterogeneity']:+.6f} "
        f"high_recall={m['gt_heterogeneity_high_recall']:.6f} "
        f"high_jaccard={m['gt_heterogeneity_high_jaccard']:.6f}"
    )


def main():
    args = parse_args()
    if not 0.0 < args.proxy_quantile < 0.5:
        raise ValueError("--proxy_quantile must lie in (0,0.5)")
    if not 0.0 < args.hard_fraction < 0.5:
        raise ValueError("--hard_fraction must lie in (0,0.5)")
    timesteps = _parse_timesteps(args.timesteps, args.diffusion_steps)

    set_seed(args.seed)
    device = get_device(args.device)
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
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        spectral_hidden=args.spectral_hidden,
        dropout=args.dropout,
        batch_size=1,
        num_workers=0,
        seed=args.seed,
        device=args.device,
    )

    _, test_loader, info = build_loaders(cfg)
    process = build_progressive_process(cfg)
    model = build_model(cfg, info, device)
    if args.checkpoint_type == "legacy":
        report = load_legacy_raw_direct_checkpoint(
            model, args.checkpoint, map_location=str(device)
        )
        print("CHECKPOINT_LOAD legacy", report)
    else:
        epoch, best = load_checkpoint(
            model, args.checkpoint, map_location=str(device), load_optimizer=False
        )
        print(f"CHECKPOINT_LOAD standard epoch={epoch} best_metric={best}")
    model.eval()

    print(
        "OBSERVABLES use only HR-MSI and the current actual reverse state x_t. "
        "GT-HSI is used only for SAM, diagnostic controls, and a reference heterogeneity ceiling."
    )
    print(
        "CROSS_MODAL_CONTROL raw MSI/HSI heterogeneity magnitudes are NOT directly subtracted; "
        "within-patch rank maps are compared because the spectral dimensionalities differ."
    )
    print(
        "RISK = rank(H_M) * relu(rank(H_M)-rank(H_t)); "
        "it targets MSI-visible heterogeneity under-represented by the current HSI state."
    )

    results: Dict[str, Dict[str, object]] = {}
    final_results = []
    counts = Counter()

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            if batch_index > 0:
                raise RuntimeError("diagnostic currently expects the standard single test patch")
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            gt_ref = spatial_material_proxies(gt, eps=args.eps)
            gt_hetero = gt_ref["heterogeneity"]
            base_valid = gt_ref["valid"]

            terminal_lr = process.terminal_observation(gt)
            x_t = process.terminal_state(
                terminal_lr, target_size=tuple(gt.shape[-2:])
            )

            risk_history = []
            selected_set = set(timesteps)
            for t in range(process.total_steps, 0, -1):
                step = torch.full(
                    (x_t.shape[0],), t, dtype=torch.long, device=device
                )
                risk_maps = observable_risk_maps(hr_msi, x_t, eps=args.eps)
                pred_x0 = model_predict(model, x_t, step, hr_msi)

                if t in selected_set:
                    sam = _pixel_sam(pred_x0, gt, eps=args.eps)
                    stage = {}
                    for name in CANDIDATES:
                        stage[name] = summarize_candidate(
                            risk_maps[name],
                            sam,
                            gt_heterogeneity=gt_hetero,
                            spectral_norm=gt_ref["spectral_norm"],
                            intensity_gradient=gt_ref["intensity_gradient"],
                            valid=base_valid,
                            proxy_quantile=args.proxy_quantile,
                            hard_fraction=args.hard_fraction,
                        )
                    results[str(t)] = stage
                    risk_history.append(risk_maps["RISK"].clone())

                x_t = process.reverse_update(x_t, pred_x0, t)

            final_pred = x_t
            final_sam = _pixel_sam(final_pred, gt, eps=args.eps)

            # Static MSI heterogeneity is available before inference.
            terminal_maps = observable_risk_maps(
                hr_msi,
                process.terminal_state(
                    terminal_lr, target_size=tuple(gt.shape[-2:])
                ),
                eps=args.eps,
            )
            final_static = summarize_candidate(
                terminal_maps["MSI_H"],
                final_sam,
                gt_heterogeneity=gt_hetero,
                spectral_norm=gt_ref["spectral_norm"],
                intensity_gradient=gt_ref["intensity_gradient"],
                valid=base_valid,
                proxy_quantile=args.proxy_quantile,
                hard_fraction=args.hard_fraction,
            )

            if not risk_history:
                raise RuntimeError("no selected timestep risk maps were accumulated")
            stacked = torch.stack(risk_history, dim=0)
            traj_mean = torch.nanmean(stacked, dim=0)
            traj_max = torch.nan_to_num(stacked, nan=-1.0).max(dim=0).values
            traj_max[traj_max < 0.0] = float("nan")

            final_traj_mean = summarize_candidate(
                traj_mean,
                final_sam,
                gt_heterogeneity=gt_hetero,
                spectral_norm=gt_ref["spectral_norm"],
                intensity_gradient=gt_ref["intensity_gradient"],
                valid=base_valid,
                proxy_quantile=args.proxy_quantile,
                hard_fraction=args.hard_fraction,
            )
            final_traj_max = summarize_candidate(
                traj_max,
                final_sam,
                gt_heterogeneity=gt_hetero,
                spectral_norm=gt_ref["spectral_norm"],
                intensity_gradient=gt_ref["intensity_gradient"],
                valid=base_valid,
                proxy_quantile=args.proxy_quantile,
                hard_fraction=args.hard_fraction,
            )
            final_results = {
                "MSI_H": final_static,
                "TRAJ_RISK_MEAN": final_traj_mean,
                "TRAJ_RISK_MAX": final_traj_max,
            }

    for t in timesteps:
        stage = results[str(t)]
        print("=" * 124)
        for name in CANDIDATES:
            _print_candidate(f"T={t:02d}", name, stage[name])
            m = stage[name]
            if m["rho_proxy_sam"] >= args.rho_threshold:
                counts[f"{name}_RHO"] += 1
            if m["partial_rho_proxy_sam_given_norm_intensitygrad"] >= args.rho_threshold:
                counts[f"{name}_PARTIAL"] += 1
            if m["delta_sam"] > 0.0:
                counts[f"{name}_DELTA"] += 1
            if m["hard_concentration_enrichment"] >= args.enrichment_threshold:
                counts[f"{name}_HARD"] += 1

    print("=" * 124)
    for name, summary in final_results.items():
        _print_candidate("FINAL", name, summary)

    n = len(timesteps)
    print("=" * 124)
    for name in CANDIDATES:
        print(
            f"CONSISTENCY_{name} "
            f"rhoStrong={counts[f'{name}_RHO']}/{n} "
            f"partialStrong={counts[f'{name}_PARTIAL']}/{n} "
            f"samDeltaPositive={counts[f'{name}_DELTA']}/{n} "
            f"hardEnriched={counts[f'{name}_HARD']}/{n}"
        )
    print(
        "INTERPRETATION_RULE Prefer an inference-observable candidate only if SAM separation and hard-pixel "
        "concentration are stable across >=4/5 selected timesteps and the FINAL trajectory aggregate remains "
        "positive.  GT_HETERO_LINK is a diagnostic reference showing whether the observable candidate actually "
        "tracks the GT-derived local spectral-heterogeneity pattern; it is not an input to the candidate."
    )

    payload = {
        "conditions": {
            "dataset": args.dataset,
            "test_size": args.test_size,
            "scale_ratio": args.scale_ratio,
            "timesteps": list(timesteps),
            "checkpoint": args.checkpoint,
            "checkpoint_type": args.checkpoint_type,
            "trajectory": "actual reverse diffusion from terminal LR-HSI",
            "degradation": "physical",
            "mtf_nyquist": args.mtf_nyquist,
            "psf_truncate": args.psf_truncate,
            "proxy_quantile": args.proxy_quantile,
            "hard_fraction": args.hard_fraction,
            "rho_threshold": args.rho_threshold,
            "enrichment_threshold": args.enrichment_threshold,
            "seed": args.seed,
        },
        "observable_definitions": {
            "MSI_H": "ranked 3x3 local heterogeneity of unit-normalized HR-MSI",
            "XT_H": "ranked 3x3 local heterogeneity of unit-normalized current reverse state x_t",
            "MISSING": "relu(MSI_H - XT_H) after within-patch ranking",
            "DISAGREE": "abs(MSI_H - XT_H) after within-patch ranking",
            "RISK": "MSI_H * MISSING",
            "gt_usage": "evaluation only: SAM, controls, GT-heterogeneity reference",
        },
        "timesteps": results,
        "final": final_results,
        "consistency_counts": dict(counts),
    }
    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
