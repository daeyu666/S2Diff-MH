"""Refine the low-frequency spectral-error diagnosis for Innovation 3.

This script is a follow-up to diagnose_spectral_shape.py.  The previous
screening showed that the remaining per-pixel orthogonal spectral error is
consistently dominated by the low DCT third and that hard/high-SAM pixels are
enriched in that low-frequency component.  The purpose here is to determine
*what kind* of broad spectral-shape error is inside that low-frequency band.

Two complementary decompositions are reported at each requested timestep:

1) low-DCT subdivision, using exactly the same orthonormal DCT-II basis as the
   previous diagnostic:

       C0, C1, C2:3, C4:L

   where L ends at the previous low/mid split.  We report both total-error
   fractions and fractions *within the low-frequency energy only*.  The latter
   is the key control: it asks which low-frequency subtype is enriched in hard
   pixels after factoring out the already-established increase in total low
   frequency content.

2) orthonormal polynomial-continuum projection of normalized orthogonal error:

       OFFSET, SLOPE, CURVATURE, REMAINDER

   on a uniform wavelength coordinate.  OFFSET/SLOPE/CURVATURE are the QR-
   orthonormalized spans of {1, lambda, lambda^2}; REMAINDER is everything not
   explained by those broad continuum modes.

As in diagnose_spectral_shape.py, q = tan(SAM) for a>0.  We therefore do not
compute partial correlations of composition with SAM conditioned on q.  The
innovation evidence is instead based on stable composition/enrichment across
separate timesteps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from config import TrainConfig
from data_loader import build_loaders
from diagnose_spectral_shape import (
    _dct_matrix,
    _flatten_spectra,
    _load_wavelengths,
    _mean,
    _median,
    _parse_timesteps,
    _spearman,
    _uniform_resample,
)
from innovation1 import build_progressive_process, model_predict
from main import build_model
from models import load_legacy_raw_direct_checkpoint
from utils import ensure_dir, get_device, load_checkpoint, set_seed


LOW_COMPONENTS = ("C0", "C1", "C23", "C4L")
CONTINUUM_COMPONENTS = ("OFFSET", "SLOPE", "CURVATURE", "REMAINDER")


def parse_args():
    p = argparse.ArgumentParser(description="Innovation-3 low-frequency spectral-shape refinement diagnostic")
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
    p.add_argument("--checkpoint", default="./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth")
    p.add_argument("--checkpoint_type", choices=["legacy", "standard"], default="legacy")

    p.add_argument("--wavelengths", default="", help="optional .npy or text wavelength-center file")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--a_min", type=float, default=1e-6)
    p.add_argument("--quartile", type=float, default=0.25)
    p.add_argument("--output_json", default="./results/lowfreq_shape_diagnostic_PaviaU.json")
    return p.parse_args()


def _continuum_basis(n_bands: int, device, dtype) -> torch.Tensor:
    """Return orthonormal columns spanning constant, linear, quadratic modes."""
    x = torch.linspace(-1.0, 1.0, n_bands, device=device, dtype=dtype)
    design = torch.stack([torch.ones_like(x), x, x.square()], dim=1)
    q, _ = torch.linalg.qr(design, mode="reduced")
    return q


def _nan(n: int, device) -> torch.Tensor:
    return torch.full((n,), float("nan"), device=device, dtype=torch.float32)


def decompose_lowfreq_error(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    dct: torch.Tensor,
    continuum_basis: torch.Tensor,
    wavelengths: Optional[np.ndarray],
    eps: float,
    a_min: float,
) -> Dict[str, torch.Tensor]:
    p = _flatten_spectra(estimate).float()
    s = _flatten_spectra(target).float()
    p, _ = _uniform_resample(p, wavelengths)
    s, _ = _uniform_resample(s, wavelengths)

    n, c = s.shape
    low_end = c // 3
    if low_end <= 4:
        raise ValueError(f"need low-frequency band to contain coefficient 4, got n_bands={c}")

    s_norm2 = s.square().sum(dim=1)
    s_norm = torch.sqrt(s_norm2.clamp_min(eps))
    p_norm = torch.sqrt(p.square().sum(dim=1).clamp_min(eps))
    dot = (s * p).sum(dim=1)
    a = dot / s_norm2.clamp_min(eps)
    cos = dot / (s_norm * p_norm).clamp_min(eps)
    sam = torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * (180.0 / math.pi)
    valid = (s_norm > math.sqrt(eps)) & (p_norm > math.sqrt(eps)) & (a > a_min)
    idx = torch.nonzero(valid, as_tuple=False).flatten()

    result: Dict[str, torch.Tensor] = {"valid": valid, "sam": sam, "q": _nan(n, s.device)}
    for name in LOW_COMPONENTS:
        result[f"E_{name}"] = _nan(n, s.device)
        result[f"FT_{name}"] = _nan(n, s.device)
        result[f"FL_{name}"] = _nan(n, s.device)
    for name in CONTINUUM_COMPONENTS:
        result[f"EC_{name}"] = _nan(n, s.device)
        result[f"FC_{name}"] = _nan(n, s.device)
    result["E_LOW"] = _nan(n, s.device)
    result["E_TOTAL"] = _nan(n, s.device)

    if idx.numel() == 0:
        return result

    sv = s[idx]
    pv = p[idx]
    av = a[idx]
    sn = s_norm[idx]
    e_perp = pv - av.unsqueeze(1) * sv
    ebar = e_perp / (av * sn).clamp_min(eps).unsqueeze(1)
    qmag = torch.linalg.vector_norm(ebar, dim=1)

    coeff = ebar @ dct.T
    energy = coeff.square()
    e_total = energy.sum(dim=1).clamp_min(eps)
    e_low = energy[:, :low_end].sum(dim=1).clamp_min(eps)

    low_energy = {
        "C0": energy[:, 0],
        "C1": energy[:, 1],
        "C23": energy[:, 2:4].sum(dim=1),
        "C4L": energy[:, 4:low_end].sum(dim=1),
    }

    continuum_coeff = ebar @ continuum_basis
    continuum_energy = continuum_coeff.square()
    broad_sum = continuum_energy.sum(dim=1)
    continuum = {
        "OFFSET": continuum_energy[:, 0],
        "SLOPE": continuum_energy[:, 1],
        "CURVATURE": continuum_energy[:, 2],
        "REMAINDER": (e_total - broad_sum).clamp_min(0.0),
    }

    result["q"][idx] = qmag
    result["E_LOW"][idx] = e_low
    result["E_TOTAL"][idx] = e_total
    for name, value in low_energy.items():
        result[f"E_{name}"][idx] = value
        result[f"FT_{name}"][idx] = value / e_total
        result[f"FL_{name}"][idx] = value / e_low
    for name, value in continuum.items():
        result[f"EC_{name}"][idx] = value
        result[f"FC_{name}"][idx] = value / e_total
    return result


def _merge(items):
    items = list(items)
    return {k: torch.cat([x[k].detach().cpu() for x in items], dim=0) for k in items[0]}


def _summarize(d: Dict[str, torch.Tensor], quartile: float) -> Dict[str, object]:
    mask = d["valid"]
    if int(mask.sum().item()) < 8:
        raise RuntimeError("too few valid spectra")
    sam = d["sam"][mask]
    q = d["q"][mask]
    lo_thr = torch.quantile(sam, quartile)
    hi_thr = torch.quantile(sam, 1.0 - quartile)
    easy = sam <= lo_thr
    hard = sam >= hi_thr

    low_global_total = {}
    low_global_within = {}
    low_mean_within = {}
    low_hard_delta_total = {}
    low_hard_delta_within = {}
    low_rho_within_q = {}

    total_sum = d["E_TOTAL"][mask].sum().clamp_min(1e-12)
    low_sum = d["E_LOW"][mask].sum().clamp_min(1e-12)
    for name in LOW_COMPONENTS:
        e = d[f"E_{name}"][mask]
        ft = d[f"FT_{name}"][mask]
        fl = d[f"FL_{name}"][mask]
        low_global_total[name] = float((e.sum() / total_sum).item())
        low_global_within[name] = float((e.sum() / low_sum).item())
        low_mean_within[name] = _mean(fl)
        low_hard_delta_total[name] = _mean(ft[hard]) - _mean(ft[easy])
        low_hard_delta_within[name] = _mean(fl[hard]) - _mean(fl[easy])
        low_rho_within_q[name] = _spearman(fl, q)

    continuum_global = {}
    continuum_mean = {}
    continuum_hard_delta = {}
    continuum_rho_q = {}
    for name in CONTINUUM_COMPONENTS:
        e = d[f"EC_{name}"][mask]
        f = d[f"FC_{name}"][mask]
        continuum_global[name] = float((e.sum() / total_sum).item())
        continuum_mean[name] = _mean(f)
        continuum_hard_delta[name] = _mean(f[hard]) - _mean(f[easy])
        continuum_rho_q[name] = _spearman(f, q)

    return {
        "n_valid": int(mask.sum().item()),
        "mean_sam_deg": _mean(sam),
        "median_sam_deg": _median(sam),
        "mean_q": _mean(q),
        "low_total_share": float((low_sum / total_sum).item()),
        "low_dct_global_total_fraction": low_global_total,
        "low_dct_global_within_low": low_global_within,
        "low_dct_mean_pixel_within_low": low_mean_within,
        "low_dct_hard_minus_easy_total_fraction": low_hard_delta_total,
        "low_dct_hard_minus_easy_within_low": low_hard_delta_within,
        "low_dct_rho_within_low_q": low_rho_within_q,
        "continuum_global_fraction": continuum_global,
        "continuum_mean_pixel_fraction": continuum_mean,
        "continuum_hard_minus_easy_fraction": continuum_hard_delta,
        "continuum_rho_fraction_q": continuum_rho_q,
    }


def _fmt(data: Dict[str, float], names) -> str:
    return " ".join(f"{name}={data[name]:+.6f}" for name in names)


def _print_timestep(t: int, s: Dict[str, object]):
    print("=" * 120)
    print(
        f"T={t:02d} LOW_SUMMARY meanSAM={s['mean_sam_deg']:.6f}deg meanQ={s['mean_q']:.8f} "
        f"LOW_TOTAL_SHARE={s['low_total_share']:.6f} valid={s['n_valid']}"
    )
    print(f"T={t:02d} LOW_DCT_GLOBAL_TOTAL {_fmt(s['low_dct_global_total_fraction'], LOW_COMPONENTS)}")
    print(f"T={t:02d} LOW_DCT_WITHIN_LOW {_fmt(s['low_dct_global_within_low'], LOW_COMPONENTS)}")
    print(f"T={t:02d} LOW_DCT_HARD_MINUS_EASY_WITHIN_LOW {_fmt(s['low_dct_hard_minus_easy_within_low'], LOW_COMPONENTS)}")
    print(f"T={t:02d} LOW_DCT_RHO_WITHIN_LOW_Q {_fmt(s['low_dct_rho_within_low_q'], LOW_COMPONENTS)}")
    print(f"T={t:02d} CONTINUUM_GLOBAL {_fmt(s['continuum_global_fraction'], CONTINUUM_COMPONENTS)}")
    print(f"T={t:02d} CONTINUUM_HARD_MINUS_EASY {_fmt(s['continuum_hard_minus_easy_fraction'], CONTINUUM_COMPONENTS)}")
    print(f"T={t:02d} CONTINUUM_RHO_Q {_fmt(s['continuum_rho_fraction_q'], CONTINUUM_COMPONENTS)}")


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
        epoch, best = load_checkpoint(model, args.checkpoint, map_location=str(device), load_optimizer=False)
        print(f"CHECKPOINT_LOAD standard epoch={epoch} best_metric={best}")
    model.eval()

    dct = _dct_matrix(info["n_bands"], device=device, dtype=torch.float32)
    continuum_basis = _continuum_basis(info["n_bands"], device=device, dtype=torch.float32)
    low_end = info["n_bands"] // 3
    if wavelengths is None:
        wavelength_mode = "native_band_index"
        print("WAVELENGTH_MODE=native_band_index WARNING=physical wavelength spacing not supplied")
    else:
        wavelength_mode = "uniform_wavelength_resampling"
        print(f"WAVELENGTH_MODE=uniform_wavelength_resampling source={args.wavelengths}")
    print(
        f"LOW_DCT_SPLIT n_bands={info['n_bands']} C0=[0] C1=[1] C23=[2,3] "
        f"C4L=[4,{low_end-1}] previous_L=[0,{low_end-1}]"
    )
    print("CONTINUUM_BASIS=orthonormal_QR({1,lambda,lambda^2}); REMAINDER=orthogonal complement")
    print("KEY_CONTROL=interpret HARD_MINUS_EASY_WITHIN_LOW, not only total low-frequency share")

    per_t = {t: [] for t in timesteps}
    with torch.no_grad():
        for batch in test_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            for t in timesteps:
                x_t = process.state_at(gt, t)
                step = torch.full((gt.shape[0],), t, dtype=torch.long, device=device)
                pred = model_predict(model, x_t, step, hr_msi)
                per_t[t].append(
                    decompose_lowfreq_error(
                        pred,
                        gt,
                        dct=dct,
                        continuum_basis=continuum_basis,
                        wavelengths=wavelengths,
                        eps=args.eps,
                        a_min=args.a_min,
                    )
                )

    results = {}
    low_dom = Counter()
    low_hard = Counter()
    continuum_dom = Counter()
    continuum_hard = Counter()

    for t in timesteps:
        merged = _merge(per_t[t])
        summary = _summarize(merged, args.quartile)
        _print_timestep(t, summary)

        dom_low = max(LOW_COMPONENTS, key=lambda x: summary["low_dct_global_within_low"][x])
        hard_low = max(LOW_COMPONENTS, key=lambda x: summary["low_dct_hard_minus_easy_within_low"][x])
        dom_cont = max(CONTINUUM_COMPONENTS, key=lambda x: summary["continuum_global_fraction"][x])
        hard_cont = max(CONTINUUM_COMPONENTS, key=lambda x: summary["continuum_hard_minus_easy_fraction"][x])
        low_dom[dom_low] += 1
        low_hard[hard_low] += 1
        continuum_dom[dom_cont] += 1
        continuum_hard[hard_cont] += 1
        results[str(t)] = {
            "summary": summary,
            "dominant_low_dct_component": dom_low,
            "hard_pixel_low_dct_enrichment": hard_low,
            "dominant_continuum_component": dom_cont,
            "hard_pixel_continuum_enrichment": hard_cont,
        }

    print("=" * 120)
    print("CONSISTENCY_LOW_DCT_DOMINANT " + " ".join(f"{x}={low_dom[x]}/{len(timesteps)}" for x in LOW_COMPONENTS))
    print("CONSISTENCY_LOW_DCT_HARD_ENRICHMENT " + " ".join(f"{x}={low_hard[x]}/{len(timesteps)}" for x in LOW_COMPONENTS))
    print("CONSISTENCY_CONTINUUM_DOMINANT " + " ".join(f"{x}={continuum_dom[x]}/{len(timesteps)}" for x in CONTINUUM_COMPONENTS))
    print("CONSISTENCY_CONTINUUM_HARD_ENRICHMENT " + " ".join(f"{x}={continuum_hard[x]}/{len(timesteps)}" for x in CONTINUUM_COMPONENTS))
    print(
        "INTERPRETATION_RULE: use a low-frequency subtype as Innovation-3 motivation only if its direction is "
        "stable across >=4/5 timesteps, especially in LOW_DCT_HARD_MINUS_EASY_WITHIN_LOW and the independent "
        "polynomial-continuum decomposition.  C1 is slope-like DCT content, while the polynomial SLOPE term is "
        "the direct linear-continuum diagnostic."
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
            "low_end_exclusive": low_end,
            "quartile": args.quartile,
            "seed": args.seed,
        },
        "timesteps": results,
        "consistency": {
            "low_dct_dominant": dict(low_dom),
            "low_dct_hard_enrichment": dict(low_hard),
            "continuum_dominant": dict(continuum_dom),
            "continuum_hard_enrichment": dict(continuum_hard),
        },
    }
    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
