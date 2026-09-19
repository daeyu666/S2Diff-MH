"""Diagnose whether residual spectral-angle error is spatially concentrated at material boundaries.

This is the spatial follow-up to diagnose_spectral_shape.py and
diagnose_lowfreq_shape.py.  It does not train a new model.

The previous diagnostics established that the residual orthogonal spectral error
of Raw-MSI Direct is dominated by the low DCT third, but is not well explained
by low-order offset/slope/curvature modes.  Here we ask whether that broad,
higher-order spectral-shape remainder is spatially localized at material
transitions / spectrally heterogeneous neighborhoods.

No semantic labels or abundance GT are assumed.  Two GT-HSI-derived diagnostic
proxies are therefore used only for analysis:

  BOUNDARY:
      spatial gradient magnitude of unit-normalized GT spectra.  This responds
      to spectral-shape transitions rather than simple brightness edges.

  HETEROGENEITY:
      3x3 local variance of unit-normalized GT spectra.  This is a local
      spectral-heterogeneity surrogate, NOT a claim of true sub-pixel abundance
      or mixed-pixel ground truth.

For each t in {1,3,6,9,12}, and for the final reverse reconstruction, report:

  * Spearman(proxy, per-pixel SAM)
  * rank-partial Spearman controlling GT spectral norm + intensity gradient
  * high-proxy quartile vs low-proxy quartile SAM gap
  * hard-SAM top-10% concentration inside high-proxy quartile
  * enrichment of low-frequency fraction FL, within-low C4:L fraction, and
    continuum REMAINDER in boundary/heterogeneous regions.

P-values are intentionally not reported because neighboring pixels are strongly
spatially correlated; effect direction/magnitude and consistency across
timesteps are the useful screening evidence.
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
from diagnose_lowfreq_shape import _continuum_basis, decompose_lowfreq_error
from diagnose_spectral_shape import (
    _dct_matrix,
    _rankdata,
    _spearman,
    decompose_spectral_error,
)
from innovation1 import (
    build_progressive_process,
    model_predict,
    reconstruct_from_terminal_lr,
)
from main import build_model
from models import load_legacy_raw_direct_checkpoint
from utils import ensure_dir, get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 diagnostic: boundary / local spectral heterogeneity vs residual SAM"
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
    p.add_argument("--a_min", type=float, default=1e-6)
    p.add_argument(
        "--proxy_quantile",
        type=float,
        default=0.25,
        help="bottom/top fraction for homogeneous/interior vs boundary/heterogeneous comparison",
    )
    p.add_argument(
        "--hard_fraction",
        type=float,
        default=0.10,
        help="fraction of highest-SAM pixels used for concentration analysis",
    )
    p.add_argument("--rho_threshold", type=float, default=0.20)
    p.add_argument("--enrichment_threshold", type=float, default=1.25)
    p.add_argument(
        "--output_json",
        default="./results/boundary_mixedpixel_diagnostic_PaviaU.json",
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


def _pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.double()
    y = y.double()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().sum() * y.square().sum())
    if not torch.isfinite(denom) or float(denom.item()) <= eps:
        return float("nan")
    return float(((x * y).sum() / denom).item())


def _partial_spearman(
    x: torch.Tensor,
    y: torch.Tensor,
    controls: Iterable[torch.Tensor],
) -> float:
    controls = list(controls)
    mask = torch.isfinite(x) & torch.isfinite(y)
    for control in controls:
        mask &= torch.isfinite(control)
    if int(mask.sum().item()) < 16:
        return float("nan")

    rx = _rankdata(x[mask]).double()
    ry = _rankdata(y[mask]).double()
    z_cols = [_rankdata(control[mask]).double() for control in controls]
    if z_cols:
        z = torch.stack(z_cols, dim=1)
        z = (z - z.mean(dim=0, keepdim=True)) / z.std(
            dim=0, unbiased=False, keepdim=True
        ).clamp_min(1e-12)
        design = torch.cat(
            [torch.ones((z.shape[0], 1), dtype=z.dtype, device=z.device), z],
            dim=1,
        )
    else:
        design = torch.ones((rx.numel(), 1), dtype=rx.dtype, device=rx.device)

    beta_x = torch.linalg.lstsq(design, rx.unsqueeze(1)).solution.squeeze(1)
    beta_y = torch.linalg.lstsq(design, ry.unsqueeze(1)).solution.squeeze(1)
    resid_x = rx - design @ beta_x
    resid_y = ry - design @ beta_y
    return _pearson(resid_x, resid_y)


def spatial_material_proxies(gt: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """Return flattened spatial proxies matching B,H,W flatten order.

    BOUNDARY is a central-difference gradient of unit-normalized spectra.
    HETEROGENEITY is the sqrt summed channel variance in a reflect-padded 3x3
    neighborhood of unit-normalized spectra.
    """
    if gt.ndim != 4:
        raise ValueError("gt must be BxCxHxW")
    b, c, h, w = gt.shape
    if h < 3 or w < 3:
        raise ValueError("need spatial size >=3x3")

    norm = torch.linalg.vector_norm(gt.float(), dim=1, keepdim=True)
    unit = gt.float() / norm.clamp_min(eps)

    boundary = torch.full((b, h, w), float("nan"), device=gt.device)
    dx = 0.5 * (unit[:, :, 1:-1, 2:] - unit[:, :, 1:-1, :-2])
    dy = 0.5 * (unit[:, :, 2:, 1:-1] - unit[:, :, :-2, 1:-1])
    boundary[:, 1:-1, 1:-1] = torch.sqrt(
        (dx.square() + dy.square()).sum(dim=1).clamp_min(0.0)
    )

    unit_pad = F.pad(unit, (1, 1, 1, 1), mode="reflect")
    local_mean = F.avg_pool2d(unit_pad, kernel_size=3, stride=1)
    local_mean_sq = F.avg_pool2d(unit_pad.square(), kernel_size=3, stride=1)
    local_var = (local_mean_sq - local_mean.square()).clamp_min(0.0)
    heterogeneity = torch.sqrt(local_var.sum(dim=1).clamp_min(0.0))
    heterogeneity[:, 0, :] = float("nan")
    heterogeneity[:, -1, :] = float("nan")
    heterogeneity[:, :, 0] = float("nan")
    heterogeneity[:, :, -1] = float("nan")

    intensity = gt.float().mean(dim=1)
    intensity_grad = torch.full((b, h, w), float("nan"), device=gt.device)
    idx = 0.5 * (intensity[:, 1:-1, 2:] - intensity[:, 1:-1, :-2])
    idy = 0.5 * (intensity[:, 2:, 1:-1] - intensity[:, :-2, 1:-1])
    intensity_grad[:, 1:-1, 1:-1] = torch.sqrt(idx.square() + idy.square())

    spectral_norm = norm[:, 0]
    valid = (
        torch.isfinite(boundary)
        & torch.isfinite(heterogeneity)
        & torch.isfinite(intensity_grad)
        & (spectral_norm > math.sqrt(eps))
    )

    return {
        "boundary": boundary.reshape(-1),
        "heterogeneity": heterogeneity.reshape(-1),
        "intensity_gradient": intensity_grad.reshape(-1),
        "spectral_norm": spectral_norm.reshape(-1),
        "valid": valid.reshape(-1),
    }


def _mean(x: torch.Tensor) -> float:
    return float(x.mean().item()) if x.numel() else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return numerator / denominator


def _proxy_summary(
    proxy: torch.Tensor,
    sam: torch.Tensor,
    valid: torch.Tensor,
    *,
    spectral_norm: torch.Tensor,
    intensity_gradient: torch.Tensor,
    fl: torch.Tensor,
    c4l_within_low: torch.Tensor,
    continuum_remainder: torch.Tensor,
    proxy_quantile: float,
    hard_fraction: float,
) -> Dict[str, float]:
    mask = (
        valid
        & torch.isfinite(proxy)
        & torch.isfinite(sam)
        & torch.isfinite(spectral_norm)
        & torch.isfinite(intensity_gradient)
        & torch.isfinite(fl)
        & torch.isfinite(c4l_within_low)
        & torch.isfinite(continuum_remainder)
    )
    if int(mask.sum().item()) < 64:
        raise RuntimeError("too few valid pixels for spatial diagnostic")

    p = proxy[mask]
    s = sam[mask]
    nrm = spectral_norm[mask]
    igrad = intensity_gradient[mask]
    flv = fl[mask]
    c4 = c4l_within_low[mask]
    rem = continuum_remainder[mask]

    lo_thr = torch.quantile(p, proxy_quantile)
    hi_thr = torch.quantile(p, 1.0 - proxy_quantile)
    low = p <= lo_thr
    high = p >= hi_thr

    sam_hard_thr = torch.quantile(s, 1.0 - hard_fraction)
    hard = s >= sam_hard_thr

    high_rate = float(high.float().mean().item())
    hard_high_rate = float(high[hard].float().mean().item())
    enrichment = _ratio(hard_high_rate, high_rate)

    return {
        "n_valid": int(mask.sum().item()),
        "rho_proxy_sam": _spearman(p, s),
        "partial_rho_proxy_sam_given_norm_intensitygrad": _partial_spearman(
            p, s, [nrm, igrad]
        ),
        "sam_high_proxy": _mean(s[high]),
        "sam_low_proxy": _mean(s[low]),
        "delta_sam_high_minus_low": _mean(s[high]) - _mean(s[low]),
        "ratio_sam_high_over_low": _ratio(_mean(s[high]), _mean(s[low])),
        "hard_pixel_fraction": hard_fraction,
        "high_proxy_population_fraction": high_rate,
        "hard_in_high_proxy_fraction": hard_high_rate,
        "hard_concentration_enrichment": enrichment,
        "delta_FL_high_minus_low": _mean(flv[high]) - _mean(flv[low]),
        "delta_C4L_within_low_high_minus_low": _mean(c4[high]) - _mean(c4[low]),
        "delta_continuum_remainder_high_minus_low": _mean(rem[high]) - _mean(rem[low]),
        "rho_proxy_FL": _spearman(p, flv),
        "rho_proxy_C4L_within_low": _spearman(p, c4),
        "rho_proxy_continuum_remainder": _spearman(p, rem),
    }


def analyze_prediction(
    pred: torch.Tensor,
    gt: torch.Tensor,
    proxies: Dict[str, torch.Tensor],
    *,
    dct: torch.Tensor,
    continuum_basis: torch.Tensor,
    eps: float,
    a_min: float,
    proxy_quantile: float,
    hard_fraction: float,
) -> Dict[str, object]:
    spectral = decompose_spectral_error(
        pred,
        gt,
        dct=dct,
        wavelengths=None,
        eps=eps,
        a_min=a_min,
    )
    lowfreq = decompose_lowfreq_error(
        pred,
        gt,
        dct=dct,
        continuum_basis=continuum_basis,
        wavelengths=None,
        eps=eps,
        a_min=a_min,
    )

    valid = proxies["valid"] & spectral["valid"] & lowfreq["valid"]
    sam = spectral["sam"]
    fl = spectral["FL"]
    c4l = lowfreq["FL_C4L"]
    remainder = lowfreq["FC_REMAINDER"]

    boundary = _proxy_summary(
        proxies["boundary"],
        sam,
        valid,
        spectral_norm=proxies["spectral_norm"],
        intensity_gradient=proxies["intensity_gradient"],
        fl=fl,
        c4l_within_low=c4l,
        continuum_remainder=remainder,
        proxy_quantile=proxy_quantile,
        hard_fraction=hard_fraction,
    )
    heterogeneity = _proxy_summary(
        proxies["heterogeneity"],
        sam,
        valid,
        spectral_norm=proxies["spectral_norm"],
        intensity_gradient=proxies["intensity_gradient"],
        fl=fl,
        c4l_within_low=c4l,
        continuum_remainder=remainder,
        proxy_quantile=proxy_quantile,
        hard_fraction=hard_fraction,
    )

    mask = valid & torch.isfinite(sam)
    return {
        "mean_sam_deg": _mean(sam[mask]),
        "median_sam_deg": float(sam[mask].median().item()),
        "boundary": boundary,
        "heterogeneity": heterogeneity,
    }


def _print_proxy(label: str, proxy_name: str, m: Dict[str, float]):
    print(
        f"{label} {proxy_name}_SAM "
        f"rho={m['rho_proxy_sam']:+.6f} "
        f"partial_rho_norm_igrad={m['partial_rho_proxy_sam_given_norm_intensitygrad']:+.6f} "
        f"high={m['sam_high_proxy']:.6f}deg low={m['sam_low_proxy']:.6f}deg "
        f"delta={m['delta_sam_high_minus_low']:+.6f}deg ratio={m['ratio_sam_high_over_low']:.6f}"
    )
    print(
        f"{label} {proxy_name}_HARD_CONCENTRATION "
        f"hard_in_high={m['hard_in_high_proxy_fraction']:.6f} "
        f"high_population={m['high_proxy_population_fraction']:.6f} "
        f"enrichment={m['hard_concentration_enrichment']:.6f}"
    )
    print(
        f"{label} {proxy_name}_SPECTRAL_LINK "
        f"dFL={m['delta_FL_high_minus_low']:+.6f} "
        f"dC4L={m['delta_C4L_within_low_high_minus_low']:+.6f} "
        f"dREMAINDER={m['delta_continuum_remainder_high_minus_low']:+.6f} "
        f"rhoFL={m['rho_proxy_FL']:+.6f} "
        f"rhoC4L={m['rho_proxy_C4L_within_low']:+.6f} "
        f"rhoREMAINDER={m['rho_proxy_continuum_remainder']:+.6f}"
    )


def _print_stage(label: str, result: Dict[str, object]):
    print("=" * 124)
    print(
        f"{label} SUMMARY meanSAM={result['mean_sam_deg']:.6f}deg "
        f"medianSAM={result['median_sam_deg']:.6f}deg"
    )
    _print_proxy(label, "BOUNDARY", result["boundary"])
    _print_proxy(label, "HETERO", result["heterogeneity"])


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
            model,
            args.checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        print(f"CHECKPOINT_LOAD standard epoch={epoch} best_metric={best}")
    model.eval()

    dct = _dct_matrix(info["n_bands"], device=device, dtype=torch.float32)
    continuum_basis = _continuum_basis(
        info["n_bands"], device=device, dtype=torch.float32
    )

    print(
        "PROXY_DEFINITION BOUNDARY=central spatial gradient of unit-normalized GT spectra; "
        "HETERO=3x3 local variance of unit-normalized GT spectra."
    )
    print(
        "CAUTION HETERO is only a local spectral-heterogeneity surrogate; "
        "it is not abundance GT and must not be described as proven mixed pixels."
    )
    print(
        f"SCREEN_RULE rho_threshold={args.rho_threshold:.3f} "
        f"enrichment_threshold={args.enrichment_threshold:.3f} "
        "and require >=4/5 timestep consistency before using the spatial mechanism as motivation."
    )

    per_t: Dict[str, Dict[str, object]] = {}
    final_results = []

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            proxies = spatial_material_proxies(gt, eps=args.eps)

            for t in timesteps:
                x_t = process.state_at(gt, t)
                step = torch.full(
                    (gt.shape[0],), t, dtype=torch.long, device=device
                )
                pred = model_predict(model, x_t, step, hr_msi)
                key = str(t)
                if key in per_t:
                    raise RuntimeError(
                        "This diagnostic currently expects one test patch; "
                        "merge logic should be added before using multiple test batches."
                    )
                per_t[key] = analyze_prediction(
                    pred,
                    gt,
                    proxies,
                    dct=dct,
                    continuum_basis=continuum_basis,
                    eps=args.eps,
                    a_min=args.a_min,
                    proxy_quantile=args.proxy_quantile,
                    hard_fraction=args.hard_fraction,
                )

            terminal_lr = process.terminal_observation(gt)
            final_pred = reconstruct_from_terminal_lr(
                model,
                process,
                terminal_lr,
                target_size=tuple(gt.shape[-2:]),
                hr_msi=hr_msi,
            )
            final_results.append(
                analyze_prediction(
                    final_pred,
                    gt,
                    proxies,
                    dct=dct,
                    continuum_basis=continuum_basis,
                    eps=args.eps,
                    a_min=args.a_min,
                    proxy_quantile=args.proxy_quantile,
                    hard_fraction=args.hard_fraction,
                )
            )

    if len(final_results) != 1:
        raise RuntimeError("expected the standard single test patch")
    final_result = final_results[0]

    counts = Counter()
    for t in timesteps:
        result = per_t[str(t)]
        _print_stage(f"T={t:02d}", result)
        for proxy_name in ("boundary", "heterogeneity"):
            m = result[proxy_name]
            prefix = "BOUNDARY" if proxy_name == "boundary" else "HETERO"
            if m["rho_proxy_sam"] >= args.rho_threshold:
                counts[f"{prefix}_RHO_STRONG"] += 1
            if (
                m["partial_rho_proxy_sam_given_norm_intensitygrad"]
                >= args.rho_threshold
            ):
                counts[f"{prefix}_PARTIAL_RHO_STRONG"] += 1
            if m["delta_sam_high_minus_low"] > 0.0:
                counts[f"{prefix}_SAM_DELTA_POSITIVE"] += 1
            if (
                m["hard_concentration_enrichment"]
                >= args.enrichment_threshold
            ):
                counts[f"{prefix}_HARD_ENRICHED"] += 1
            if m["delta_FL_high_minus_low"] > 0.0:
                counts[f"{prefix}_FL_ENRICHED"] += 1
            if m["delta_C4L_within_low_high_minus_low"] > 0.0:
                counts[f"{prefix}_C4L_ENRICHED"] += 1
            if m["delta_continuum_remainder_high_minus_low"] > 0.0:
                counts[f"{prefix}_REMAINDER_ENRICHED"] += 1

    _print_stage("FINAL", final_result)

    n = len(timesteps)
    print("=" * 124)
    print(
        "CONSISTENCY_BOUNDARY "
        f"rhoStrong={counts['BOUNDARY_RHO_STRONG']}/{n} "
        f"partialRhoStrong={counts['BOUNDARY_PARTIAL_RHO_STRONG']}/{n} "
        f"samDeltaPositive={counts['BOUNDARY_SAM_DELTA_POSITIVE']}/{n} "
        f"hardEnriched={counts['BOUNDARY_HARD_ENRICHED']}/{n} "
        f"FLenriched={counts['BOUNDARY_FL_ENRICHED']}/{n} "
        f"C4Lenriched={counts['BOUNDARY_C4L_ENRICHED']}/{n} "
        f"remainderEnriched={counts['BOUNDARY_REMAINDER_ENRICHED']}/{n}"
    )
    print(
        "CONSISTENCY_HETERO "
        f"rhoStrong={counts['HETERO_RHO_STRONG']}/{n} "
        f"partialRhoStrong={counts['HETERO_PARTIAL_RHO_STRONG']}/{n} "
        f"samDeltaPositive={counts['HETERO_SAM_DELTA_POSITIVE']}/{n} "
        f"hardEnriched={counts['HETERO_HARD_ENRICHED']}/{n} "
        f"FLenriched={counts['HETERO_FL_ENRICHED']}/{n} "
        f"C4Lenriched={counts['HETERO_C4L_ENRICHED']}/{n} "
        f"remainderEnriched={counts['HETERO_REMAINDER_ENRICHED']}/{n}"
    )
    print(
        "INTERPRETATION_RULE A spatial-material motivation is supported only if the proxy has "
        "consistent positive SAM separation/concentration across >=4/5 timesteps, survives the "
        "rank-partial control for spectral norm + brightness-gradient where possible, and the FINAL "
        "reconstruction shows the same direction.  Spectral-link terms (FL/C4L/REMAINDER) are secondary "
        "evidence tying spatial concentration to the previously diagnosed broad-shape residual."
    )

    payload = {
        "conditions": {
            "dataset": args.dataset,
            "test_size": args.test_size,
            "scale_ratio": args.scale_ratio,
            "timesteps": list(timesteps),
            "checkpoint": args.checkpoint,
            "checkpoint_type": args.checkpoint_type,
            "degradation": "physical",
            "mtf_nyquist": args.mtf_nyquist,
            "psf_truncate": args.psf_truncate,
            "proxy_quantile": args.proxy_quantile,
            "hard_fraction": args.hard_fraction,
            "rho_threshold": args.rho_threshold,
            "enrichment_threshold": args.enrichment_threshold,
            "seed": args.seed,
        },
        "proxy_definition": {
            "boundary": "central spatial gradient magnitude of unit-normalized GT spectra",
            "heterogeneity": "3x3 local variance magnitude of unit-normalized GT spectra",
            "heterogeneity_caveat": "surrogate only; not abundance or true mixed-pixel GT",
            "controls": [
                "GT spectral norm",
                "GT mean-reflectance spatial gradient",
            ],
        },
        "timesteps": per_t,
        "final": final_result,
        "consistency_counts": dict(counts),
    }
    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
