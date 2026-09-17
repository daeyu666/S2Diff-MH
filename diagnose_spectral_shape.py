"""Diagnose wavelength-scale composition of per-pixel orthogonal spectral error.

This diagnostic is intentionally written for Innovation-3 screening on the
registered Raw-MSI Direct baseline.  It does *not* train a new module.

For each requested diffusion timestep t, it compares the physical state x_t
and the clean-spectrum prediction X_hat_0 against the HR-HSI target X.  For a
pixel spectrum s and prediction s_hat,

    a       = <s, s_hat> / ||s||^2
    e_perp  = s_hat - a s
    q       = ||e_perp|| / (a ||s||),   a > 0

so q = tan(SAM) exactly (up to numerical precision).  Therefore a partial
correlation rho(F_k, SAM | q) is mathematically degenerate and is NOT used as
innovation evidence.  Instead we diagnose:

  1) where the remaining orthogonal-error energy lives (low/mid/high DCT),
  2) whether high-SAM/high-q pixels are enriched in one spectral scale,
  3) which spectral scale Raw-MSI Direct removes least effectively from x_t,
  4) whether the same signature is stable across t=1,3,6,9,12.

If an optional wavelength file is provided, spectra are first interpolated to
an equally spaced wavelength grid before DCT/derivative analysis.  Without it,
the native band index is used and the script prints that limitation explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from config import TrainConfig
from data_loader import build_loaders
from innovation1 import build_progressive_process, model_predict
from main import build_model
from models import load_legacy_raw_direct_checkpoint
from utils import ensure_dir, get_device, load_checkpoint, set_seed


BANDS = ("L", "M", "H")


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 diagnostic: wavelength-scale composition of orthogonal spectral error"
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
    p.add_argument(
        "--checkpoint_type",
        choices=["legacy", "standard"],
        default="legacy",
        help="legacy uses load_legacy_raw_direct_checkpoint; standard uses utils.load_checkpoint",
    )

    p.add_argument(
        "--wavelengths",
        default="",
        help="optional .npy or text file containing one wavelength center per HSI band",
    )
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--a_min", type=float, default=1e-6)
    p.add_argument("--quartile", type=float, default=0.25)
    p.add_argument(
        "--output_json",
        default="./results/spectral_shape_diagnostic_PaviaU.json",
    )
    return p.parse_args()


def _parse_timesteps(text: str, total_steps: int) -> Tuple[int, ...]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if not values:
        raise ValueError("--timesteps is empty")
    if len(set(values)) != len(values):
        raise ValueError("--timesteps contains duplicates")
    for value in values:
        if value < 1 or value > total_steps:
            raise ValueError(f"timestep {value} outside [1,{total_steps}]")
    return values


def _load_wavelengths(path: str, n_bands: int) -> Optional[np.ndarray]:
    if not path:
        return None
    if path.lower().endswith(".npy"):
        values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    else:
        values = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)
    if values.size != n_bands:
        raise ValueError(f"wavelength count {values.size} != n_bands {n_bands}")
    if not np.all(np.isfinite(values)):
        raise ValueError("wavelength file contains non-finite values")
    if not np.all(np.diff(values) > 0):
        raise ValueError("wavelength centers must be strictly increasing")
    return values


def _uniform_resample(
    spectra: torch.Tensor,
    wavelengths: Optional[np.ndarray],
) -> Tuple[torch.Tensor, float]:
    """Resample [N,C] spectra to a uniform wavelength grid when centers exist."""
    if wavelengths is None:
        return spectra, 1.0

    c = spectra.shape[1]
    src = torch.as_tensor(wavelengths, device=spectra.device, dtype=spectra.dtype)
    dst = torch.linspace(src[0], src[-1], c, device=spectra.device, dtype=spectra.dtype)
    right = torch.searchsorted(src, dst, right=False).clamp(1, c - 1)
    left = right - 1
    x0 = src[left]
    x1 = src[right]
    weight = (dst - x0) / (x1 - x0).clamp_min(torch.finfo(spectra.dtype).eps)
    out = spectra[:, left] * (1.0 - weight.unsqueeze(0)) + spectra[:, right] * weight.unsqueeze(0)
    spacing = float((dst[1] - dst[0]).item()) if c > 1 else 1.0
    return out, spacing


def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    """Orthonormal DCT-II matrix C with coeff = x @ C.T."""
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    j = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    matrix = torch.cos(math.pi / n * (j + 0.5) * k)
    matrix = matrix * math.sqrt(2.0 / n)
    matrix[0] = matrix[0] / math.sqrt(2.0)
    return matrix


def _flatten_spectra(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("expected BxCxHxW tensor")
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Fast rank transform; exact ties are rare for these continuous diagnostics."""
    order = torch.argsort(x)
    ranks = torch.empty_like(x, dtype=torch.float64)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float64)
    return ranks


def _pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.double()
    y = y.double()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum()) * (y.square().sum()))
    if not torch.isfinite(denom) or float(denom.item()) <= eps:
        return float("nan")
    return float((x * y).sum().div(denom).item())


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return float("nan")
    return _pearson(_rankdata(x[mask]), _rankdata(y[mask]))


def _nan_vector(n: int, device) -> torch.Tensor:
    return torch.full((n,), float("nan"), device=device, dtype=torch.float32)


def decompose_spectral_error(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    dct: torch.Tensor,
    wavelengths: Optional[np.ndarray],
    eps: float,
    a_min: float,
) -> Dict[str, torch.Tensor]:
    """Return per-pixel angular-error magnitude and wavelength-scale composition."""
    p = _flatten_spectra(estimate).float()
    s = _flatten_spectra(target).float()
    p, spacing = _uniform_resample(p, wavelengths)
    s, _ = _uniform_resample(s, wavelengths)

    n = s.shape[0]
    s_norm2 = s.square().sum(dim=1)
    s_norm = torch.sqrt(s_norm2.clamp_min(eps))
    p_norm = torch.sqrt(p.square().sum(dim=1).clamp_min(eps))
    dot = (s * p).sum(dim=1)
    a = dot / s_norm2.clamp_min(eps)
    cos = dot / (s_norm * p_norm).clamp_min(eps)
    sam_rad = torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    sam_deg = sam_rad * (180.0 / math.pi)

    valid = (s_norm > math.sqrt(eps)) & (p_norm > math.sqrt(eps)) & (a > a_min)
    idx = torch.nonzero(valid, as_tuple=False).flatten()

    result = {
        "valid": valid,
        "sam": sam_deg,
        "a": a,
        "q": _nan_vector(n, s.device),
        "EL": _nan_vector(n, s.device),
        "EM": _nan_vector(n, s.device),
        "EH": _nan_vector(n, s.device),
        "ET": _nan_vector(n, s.device),
        "FL": _nan_vector(n, s.device),
        "FM": _nan_vector(n, s.device),
        "FH": _nan_vector(n, s.device),
        "R1": _nan_vector(n, s.device),
        "R2": _nan_vector(n, s.device),
        "parseval_rel": _nan_vector(n, s.device),
        "tan_identity_abs": _nan_vector(n, s.device),
    }
    if idx.numel() == 0:
        return result

    sv = s[idx]
    pv = p[idx]
    av = a[idx]
    sn = s_norm[idx]
    e_perp = pv - av.unsqueeze(1) * sv
    denom = (av * sn).clamp_min(eps)
    ebar = e_perp / denom.unsqueeze(1)
    q = torch.linalg.vector_norm(ebar, dim=1)

    coeff = ebar @ dct.T
    energy = coeff.square()
    c = energy.shape[1]
    b1 = c // 3
    b2 = 2 * c // 3
    if b1 < 1 or b2 <= b1 or b2 >= c:
        raise ValueError(f"need at least 3 spectral bands, got {c}")
    e_l = energy[:, :b1].sum(dim=1)
    e_m = energy[:, b1:b2].sum(dim=1)
    e_h = energy[:, b2:].sum(dim=1)
    e_t = energy.sum(dim=1).clamp_min(eps)

    d1 = torch.diff(ebar, dim=1) / spacing
    d2 = torch.diff(ebar, n=2, dim=1) / (spacing * spacing)
    # Scale-free roughness of the normalized orthogonal spectral shape.
    r1 = d1.square().sum(dim=1) / e_t
    r2 = d2.square().sum(dim=1) / e_t

    parseval = (e_t - q.square()).abs() / q.square().clamp_min(eps)
    tan_id = (q - torch.tan(sam_rad[idx])).abs()

    result["q"][idx] = q
    result["EL"][idx] = e_l
    result["EM"][idx] = e_m
    result["EH"][idx] = e_h
    result["ET"][idx] = e_t
    result["FL"][idx] = e_l / e_t
    result["FM"][idx] = e_m / e_t
    result["FH"][idx] = e_h / e_t
    result["R1"][idx] = r1
    result["R2"][idx] = r2
    result["parseval_rel"][idx] = parseval
    result["tan_identity_abs"][idx] = tan_id
    return result


def _mean(x: torch.Tensor) -> float:
    return float(x.mean().item()) if x.numel() else float("nan")


def _median(x: torch.Tensor) -> float:
    return float(x.median().item()) if x.numel() else float("nan")


def summarize_stage(d: Dict[str, torch.Tensor], quartile: float) -> Dict[str, object]:
    mask = d["valid"]
    n_valid = int(mask.sum().item())
    n_total = int(mask.numel())
    if n_valid < 8:
        raise RuntimeError(f"too few valid spectra: {n_valid}/{n_total}")

    sam = d["sam"][mask]
    q = d["q"][mask]
    energies = {band: d[f"E{band}"][mask] for band in BANDS}
    fractions = {band: d[f"F{band}"][mask] for band in BANDS}
    total_energy = sum(v.sum() for v in energies.values()).clamp_min(1e-12)

    q_lo = torch.quantile(sam, quartile)
    q_hi = torch.quantile(sam, 1.0 - quartile)
    low = sam <= q_lo
    high = sam >= q_hi

    global_share = {band: float((energies[band].sum() / total_energy).item()) for band in BANDS}
    mean_fraction = {band: _mean(fractions[band]) for band in BANDS}
    median_fraction = {band: _median(fractions[band]) for band in BANDS}
    hard_delta = {
        band: _mean(fractions[band][high]) - _mean(fractions[band][low])
        for band in BANDS
    }
    rho_fraction_q = {band: _spearman(fractions[band], q) for band in BANDS}
    rho_fraction_sam = {band: _spearman(fractions[band], sam) for band in BANDS}

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "mean_sam_deg": _mean(sam),
        "median_sam_deg": _median(sam),
        "mean_q": _mean(q),
        "median_q": _median(q),
        "rho_q_sam": _spearman(q, sam),
        "global_energy_share": global_share,
        "mean_pixel_fraction": mean_fraction,
        "median_pixel_fraction": median_fraction,
        "hard_minus_easy_fraction": hard_delta,
        "rho_fraction_q": rho_fraction_q,
        "rho_fraction_sam": rho_fraction_sam,
        "mean_R1": _mean(d["R1"][mask]),
        "mean_R2": _mean(d["R2"][mask]),
        "rho_R1_q": _spearman(d["R1"][mask], q),
        "rho_R2_q": _spearman(d["R2"][mask], q),
        "parseval_rel_mean": _mean(d["parseval_rel"][mask]),
        "parseval_rel_max": float(d["parseval_rel"][mask].max().item()),
        "tan_identity_abs_mean": _mean(d["tan_identity_abs"][mask]),
        "tan_identity_abs_max": float(d["tan_identity_abs"][mask].max().item()),
    }


def summarize_reduction(
    input_d: Dict[str, torch.Tensor],
    pred_d: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    mask = input_d["valid"] & pred_d["valid"]
    if int(mask.sum().item()) < 8:
        raise RuntimeError("too few common valid pixels for input->prediction reduction")

    reduction = {}
    for band in BANDS:
        e_in = input_d[f"E{band}"][mask].sum().double()
        e_out = pred_d[f"E{band}"][mask].sum().double()
        reduction[band] = float((1.0 - e_out / e_in.clamp_min(1e-18)).item())
    q2_in = input_d["q"][mask].square().sum().double()
    q2_out = pred_d["q"][mask].square().sum().double()
    total_reduction = float((1.0 - q2_out / q2_in.clamp_min(1e-18)).item())
    weakest = min(BANDS, key=lambda band: reduction[band])
    return {
        "common_valid": int(mask.sum().item()),
        "normalized_orthogonal_energy_reduction": reduction,
        "total_q2_reduction": total_reduction,
        "weakest_reduced_band": weakest,
    }


def _merge_decompositions(items: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    items = list(items)
    keys = items[0].keys()
    return {key: torch.cat([item[key].detach().cpu() for item in items], dim=0) for key in keys}


def _fmt_triplet(data: Dict[str, float]) -> str:
    return " ".join(f"{band}={data[band]:+.6f}" for band in BANDS)


def _print_timestep(t: int, pred: Dict[str, object], inp: Dict[str, object], reduction: Dict[str, object]):
    print("=" * 112)
    print(
        f"T={t:02d} PRED meanSAM={pred['mean_sam_deg']:.6f}deg meanQ={pred['mean_q']:.8f} "
        f"rho(Q,SAM)={pred['rho_q_sam']:+.6f} valid={pred['n_valid']}/{pred['n_total']}"
    )
    print(f"T={t:02d} PRED_GLOBAL_SHARE {_fmt_triplet(pred['global_energy_share'])}")
    print(f"T={t:02d} PRED_MEAN_FRACTION {_fmt_triplet(pred['mean_pixel_fraction'])}")
    print(f"T={t:02d} HARD_MINUS_EASY_F {_fmt_triplet(pred['hard_minus_easy_fraction'])}")
    print(f"T={t:02d} RHO_F_Q {_fmt_triplet(pred['rho_fraction_q'])}")
    print(
        f"T={t:02d} ROUGHNESS meanR1={pred['mean_R1']:.6f} meanR2={pred['mean_R2']:.6f} "
        f"rhoR1Q={pred['rho_R1_q']:+.6f} rhoR2Q={pred['rho_R2_q']:+.6f}"
    )
    print(
        f"T={t:02d} INPUT_GLOBAL_SHARE {_fmt_triplet(inp['global_energy_share'])} "
        f"inputSAM={inp['mean_sam_deg']:.6f}deg"
    )
    print(
        f"T={t:02d} ERROR_REDUCTION {_fmt_triplet(reduction['normalized_orthogonal_energy_reduction'])} "
        f"TOTAL_Q2={reduction['total_q2_reduction']:+.6f} "
        f"WEAKEST={reduction['weakest_reduced_band']}"
    )
    print(
        f"T={t:02d} SANITY parseval_mean={pred['parseval_rel_mean']:.3e} "
        f"parseval_max={pred['parseval_rel_max']:.3e} "
        f"q_minus_tanSAM_mean={pred['tan_identity_abs_mean']:.3e} "
        f"q_minus_tanSAM_max={pred['tan_identity_abs_max']:.3e}"
    )


def main():
    args = parse_args()
    if not 0.0 < args.quartile < 0.5:
        raise ValueError("--quartile must lie in (0,0.5)")
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
    wavelengths = _load_wavelengths(args.wavelengths, info["n_bands"])
    process = build_progressive_process(cfg)
    model = build_model(cfg, info, device)
    if args.checkpoint_type == "legacy":
        report = load_legacy_raw_direct_checkpoint(model, args.checkpoint, map_location=str(device))
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
    b1 = info["n_bands"] // 3
    b2 = 2 * info["n_bands"] // 3
    if wavelengths is None:
        wavelength_mode = "native_band_index"
        print(
            "WAVELENGTH_MODE=native_band_index WARNING=no wavelength centers supplied; "
            "DCT frequency is with respect to band index, not verified physical wavelength spacing."
        )
    else:
        wavelength_mode = "uniform_wavelength_resampling"
        print(
            f"WAVELENGTH_MODE=uniform_wavelength_resampling source={args.wavelengths} "
            f"range=[{wavelengths[0]:.6f},{wavelengths[-1]:.6f}]"
        )
    print(
        f"DCT_SPLIT n_bands={info['n_bands']} L=[0,{b1-1}] "
        f"M=[{b1},{b2-1}] H=[{b2},{info['n_bands']-1}]"
    )
    print(
        "NOTE: rho(F,SAM|q) is intentionally NOT computed because q=tan(SAM) for a>0; "
        "conditioning on q removes SAM variation by definition."
    )

    per_t_input = {t: [] for t in timesteps}
    per_t_pred = {t: [] for t in timesteps}

    with torch.no_grad():
        for batch in test_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            for t in timesteps:
                x_t = process.state_at(gt, t)
                step = torch.full((gt.shape[0],), t, dtype=torch.long, device=device)
                pred = model_predict(model, x_t, step, hr_msi)
                per_t_input[t].append(
                    decompose_spectral_error(
                        x_t,
                        gt,
                        dct=dct,
                        wavelengths=wavelengths,
                        eps=args.eps,
                        a_min=args.a_min,
                    )
                )
                per_t_pred[t].append(
                    decompose_spectral_error(
                        pred,
                        gt,
                        dct=dct,
                        wavelengths=wavelengths,
                        eps=args.eps,
                        a_min=args.a_min,
                    )
                )

    results = {}
    dominant_counts = Counter()
    weakest_counts = Counter()
    hard_enrich_counts = Counter()

    for t in timesteps:
        inp_d = _merge_decompositions(per_t_input[t])
        pred_d = _merge_decompositions(per_t_pred[t])
        inp_summary = summarize_stage(inp_d, args.quartile)
        pred_summary = summarize_stage(pred_d, args.quartile)
        reduction = summarize_reduction(inp_d, pred_d)
        _print_timestep(t, pred_summary, inp_summary, reduction)

        dominant = max(BANDS, key=lambda band: pred_summary["global_energy_share"][band])
        hard_enrich = max(BANDS, key=lambda band: pred_summary["hard_minus_easy_fraction"][band])
        dominant_counts[dominant] += 1
        weakest_counts[reduction["weakest_reduced_band"]] += 1
        hard_enrich_counts[hard_enrich] += 1
        results[str(t)] = {
            "input": inp_summary,
            "prediction": pred_summary,
            "reduction": reduction,
            "dominant_residual_band": dominant,
            "hard_pixel_enrichment_band": hard_enrich,
        }

    print("=" * 112)
    print("CONSISTENCY_DOMINANT_RESIDUAL " + " ".join(f"{b}={dominant_counts[b]}/{len(timesteps)}" for b in BANDS))
    print("CONSISTENCY_WEAKEST_REDUCTION " + " ".join(f"{b}={weakest_counts[b]}/{len(timesteps)}" for b in BANDS))
    print("CONSISTENCY_HARD_PIXEL_ENRICHMENT " + " ".join(f"{b}={hard_enrich_counts[b]}/{len(timesteps)}" for b in BANDS))
    print(
        "INTERPRETATION_RULE: require a stable direction across >=4/5 timesteps before using wavelength-scale "
        "composition as Innovation-3 motivation.  A large F_k does not mean that frequency is intrinsically "
        "more harmful per unit energy; SAM depends on total orthogonal-error magnitude.  The useful evidence is "
        "a stable residual concentration and/or systematically weaker correction of one spectral scale."
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
            "wavelength_mode": wavelength_mode,
            "wavelength_file": args.wavelengths or None,
            "dct_split": {"L": [0, b1 - 1], "M": [b1, b2 - 1], "H": [b2, info["n_bands"] - 1]},
            "quartile": args.quartile,
            "seed": args.seed,
        },
        "mathematical_control": {
            "identity": "q = ||e_perp||/(a||s||) = tan(SAM) for a>0",
            "partial_correlation_F_SAM_given_q": "not computed because it is mathematically degenerate",
        },
        "timesteps": results,
        "consistency": {
            "dominant_residual": {b: dominant_counts[b] for b in BANDS},
            "weakest_reduction": {b: weakest_counts[b] for b in BANDS},
            "hard_pixel_enrichment": {b: hard_enrich_counts[b] for b in BANDS},
        },
    }
    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
