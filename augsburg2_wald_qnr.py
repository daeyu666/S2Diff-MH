"""Full-resolution HSI-MSI QNR: original HR-MSI spatial and LR-HSI spectral references.

NO panchromatic image (genuine or synthetic), no 4-band projection of the
reconstructed HSI, and NO 10m reference HSI.

F: fused HSI at 10m, H: observed HSI at 30m, M: observed S2 MSI at 10m.
L(M): 3x3 area averaging of real S2 MSI to the HSI 30m grid.
Q: valid-masked, local UIQI; window HR48, LR16, 80%-valid default.

Dlambda = average_{i<j, i,j in all 242 HSI bands}
  |Q(F_i,F_j) - Q(H_i,H_j)|.
Ds = mean_j mean_{i in S_j}
  |Q(F_i,M_j) - Q(H_i,L(M)_j)|,
where S_j is the subset of HSI wavelengths with nonnegligible response
in the measured MSI channel j (>=1% of the channel's peak SRF response).
QNR = max(0,1-Dlambda) * max(0,1-Ds).

These are the conventional QNR spectral/spatial distortion components
adapted to HSI-MSI fusion (hypersharpening). Do NOT mistake the older
four-band projected index or PAN-pansharpening QNR for this measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _windowed_uiqi_matrix(a, b, mask, window, min_valid_fraction=0.8):
    """Return all channel-pair UIQI, weighted by valid pixels across windows.

    a: HxWxC, b: HxWxD.  One BLAS covariance per window avoids 29k Python
    calls and allows all 242 HSI band pairs to be evaluated exactly.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.asarray(mask, dtype=bool)
    if a.ndim != 3 or b.ndim != 3 or a.shape[:2] != b.shape[:2] or mask.shape != a.shape[:2]:
        raise ValueError("UIQI expects matching spatial HxWxC, HxWxD and 2D mask")
    h, w, c = a.shape
    d = b.shape[2]
    acc = np.zeros((c, d), dtype=np.float64)
    total_weight = 0
    for y in range(0, h, window):
        for x in range(0, w, window):
            aa0 = a[y:y + window, x:x + window]
            bb0 = b[y:y + window, x:x + window]
            vmask = (
                mask[y:y + window, x:x + window]
                & np.isfinite(aa0).all(axis=2)
                & np.isfinite(bb0).all(axis=2)
            )
            n = int(vmask.sum())
            if n < 8 or n / vmask.size < min_valid_fraction:
                continue
            aa = np.asarray(aa0[vmask], dtype=np.float64)
            bb = np.asarray(bb0[vmask], dtype=np.float64)
            mu_a = aa.mean(axis=0)
            mu_b = bb.mean(axis=0)
            ca = aa - mu_a[None, :]
            cb = bb - mu_b[None, :]
            var_a = np.mean(ca * ca, axis=0)
            var_b = np.mean(cb * cb, axis=0)
            cov = (ca.T @ cb) / n
            luminance = (
                2.0 * mu_a[:, None] * mu_b[None, :]
                / (mu_a[:, None] ** 2 + mu_b[None, :] ** 2 + 1e-12)
            )
            varsum = var_a[:, None] + var_b[None, :]
            contrast = np.divide(
                2.0 * cov, varsum,
                out=np.ones_like(cov),
                where=varsum >= 1e-12,
            )
            acc += np.clip(luminance * contrast, -1.0, 1.0) * n
            total_weight += n
    if total_weight == 0:
        raise ValueError("No valid local UIQI windows; check input validity mask")
    return acc / total_weight


def hsi_msi_qnr(
    fused, observed_lr_hsi, observed_hr_msi, valid_hr, srf_weights,
    gains=None, biases=None, window_hr=48, min_valid_fraction=0.8,
    support_fraction=0.01,
):
    """QNR with 242-band HSI spectral and 4-band real MSI spatial references."""
    f = np.asarray(fused)
    hsi = np.asarray(observed_lr_hsi)
    msi = np.asarray(observed_hr_msi)
    mask = np.asarray(valid_hr, dtype=bool)
    r = np.asarray(srf_weights, dtype=np.float64)
    if f.ndim != 3 or f.shape[-1] != 242:
        raise ValueError("Fused high-resolution HSI must be HxWx242")
    height, width, bands = f.shape
    if r.shape != (4, bands) or msi.shape != (height, width, 4) or mask.shape != (height, width):
        raise ValueError("Expected SRF 4x242, real MSI HxWx4 and mask HxW")
    if (height % 3 or width % 3 or
        hsi.shape != (height // 3, width // 3, bands)):
        raise ValueError("Observed LR-HSI must be one-third the HR-MSI spatial resolution")
    if not isinstance(window_hr, int) or window_hr < 24 or window_hr % 3:
        raise ValueError("window_hr must be >=24 and divisible by 3")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0,1]")
    if not 0 < support_fraction < 1:
        raise ValueError("support_fraction must lie in (0,1)")
    if not np.isfinite(r).all() or (r < 0).any() or not np.allclose(r.sum(axis=1), 1., atol=1e-4):
        raise ValueError("Invalid fixed Sentinel-2 SRF 4x242")
    gain = np.ones(4, dtype=np.float32) if gains is None else np.asarray(gains, dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32) if biases is None else np.asarray(biases, dtype=np.float32)
    if gain.shape != (4,) or bias.shape != (4,) or not np.isfinite(gain).all() or not np.isfinite(bias).all():
        raise ValueError("Radiometry requires four finite gains and biases")

    calibrated_msi = msi.astype(np.float32) * gain[None, None, :] + bias[None, None, :]
    msi_lr = calibrated_msi.reshape(
        height // 3, 3, width // 3, 3, 4
    ).mean(axis=(1, 3))
    mask_lr = mask.reshape(
        height // 3, 3, width // 3, 3
    ).all(axis=(1, 3))
    lrwin = window_hr // 3

    # Spectral quality is calculated on *all 242 observed HSI bands*.
    q_f = _windowed_uiqi_matrix(f, f, mask, window_hr, min_valid_fraction)
    q_h = _windowed_uiqi_matrix(hsi, hsi, mask_lr, lrwin, min_valid_fraction)
    pairs = np.triu_indices(bands, k=1)
    dl = float(np.abs(q_f[pairs] - q_h[pairs]).mean())

    # Spatial quality is calculated against actual 10m MSI measurements.
    # The SRF is used ONLY to identify spectral coverage, NOT to project HSI
    # to four bands, and not to construct pseudo-PAN.
    q_fm = _windowed_uiqi_matrix(f, calibrated_msi, mask, window_hr, min_valid_fraction)
    q_hm = _windowed_uiqi_matrix(hsi, msi_lr, mask_lr, lrwin, min_valid_fraction)
    support = r >= (r.max(axis=1, keepdims=True) * support_fraction)
    spatial_per_channel = []
    coverage_counts = []
    for j in range(4):
        candidates = np.flatnonzero(support[j])
        if not len(candidates):
            raise ValueError(f"SRF band {j} has no supported HSI wavelengths")
        coverage_counts.append(int(len(candidates)))
        spatial_per_channel.append(
            float(np.abs(q_fm[candidates, j] - q_hm[candidates, j]).mean())
        )
    ds = float(np.mean(spatial_per_channel))
    return {
        "QNR": float(max(0., 1. - dl) * max(0., 1. - ds)),
        "Dlambda": dl,
        "Ds": ds,
        "index_name": "HSI-MSI QNR (LR-HSI spectral, real HR-MSI spatial)",
        "qnr_equations": "QNR-HSI-MSI-spectral-all242-spatial-SRF-covered-p=q=alpha=beta=1",
        "spectral_pair_count": int(bands * (bands - 1) // 2),
        "spatial_pair_count": int(sum(coverage_counts)),
        "spatial_support_counts": coverage_counts,
        "spatial_per_msi_band": spatial_per_channel,
        "srf_support_fraction_of_peak": float(support_fraction),
        "high_window": int(window_hr),
        "low_window": int(lrwin),
        "window_min_valid_fraction": float(min_valid_fraction),
        "high_valid_pixels": int(mask.sum()),
        "low_valid_pixels": int(mask_lr.sum()),
        "spectral_reference": "all_242_bands_of_observed_30m_HSI",
        "spatial_reference": "original_unwarped_real_10m_Sentinel-2_B2_B3_B4_B8",
        "source_HSI": "observed_30m_HSI",
        "source_MSI": "actual_10m_Sentinel-2",
        "pan_used": False,
        "srf_projection_used": False,
        "full_HR_HSI_ground_truth_used": False,
    }


def evaluate_cache(
    wald_root, fused, radiometry_json,
    window_hr=48, min_valid_fraction=0.8,
    support_fraction=0.01,
):
    root = Path(wald_root)
    with (root / "full" / "meta.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("region") != "sub_area_2":
        raise ValueError("Original-scale Wald QNR restricted to sub_area_2")
    for split in ("train", "validation", "test"):
        with (root / split / "meta.json").open(encoding="utf-8") as handle:
            splitmeta = json.load(handle)
        if (splitmeta.get("msi_source") != "real_Sentinel_2_Wald_30m"
            or splitmeta.get("target") != "30m_EnMAP_like"
            or splitmeta.get("gt_source") != "observed_30m_HSI_only"
            or int(splitmeta.get("scale_ratio", -1)) != 3):
            raise ValueError(f"Invalid strict Wald split metadata: {split}")
    with (root / "wald_psf.json").open(encoding="utf-8") as handle:
        psf = json.load(handle)
    if int(psf.get("scale_ratio", -1)) != 3:
        raise ValueError("Expected Wald scale x3")
    with Path(radiometry_json).open(encoding="utf-8") as handle:
        rad = json.load(handle)
    if (rad.get("dataset") != "Augsburg-2-Wald"
        or rad.get("uses_EnMAP10_reference") is not False):
        raise ValueError("Require train-only Augsburg-2-Wald radiometry")
    gain = np.asarray(rad["gain"], dtype=np.float32)
    bias = np.asarray(rad["bias"], dtype=np.float32)
    if gain.shape != (4,) or bias.shape != (4,) or not np.isfinite(gain).all() or not np.isfinite(bias).all():
        raise ValueError("Wald radiometry must contain four finite gains and biases")
    return hsi_msi_qnr(
        np.load(fused, mmap_mode="r"),
        np.load(root / "full" / "lr_hsi.npy", mmap_mode="r"),
        np.load(root / "full" / "hr_msi.npy", mmap_mode="r"),
        np.load(root / "full" / "valid_mask.npy", mmap_mode="r"),
        np.load(root / "srf_weights.npy"),
        gains=gain, biases=bias,
        window_hr=window_hr,
        min_valid_fraction=min_valid_fraction,
        support_fraction=support_fraction,
    )


def main():
    parser = argparse.ArgumentParser(description="Augsburg-2 HSI-MSI QNR with original observations")
    parser.add_argument("--wald_root", default="./data/augsburg2_wald")
    parser.add_argument("--fused", required=True)
    parser.add_argument("--radiometry_json", default="./data/calibration/Augsburg2_Wald_radiometry.json")
    parser.add_argument("--window_hr", type=int, default=48)
    parser.add_argument("--min_valid_fraction", type=float, default=0.8)
    parser.add_argument("--support_fraction", type=float, default=0.01)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()
    quality = evaluate_cache(
        args.wald_root, args.fused, args.radiometry_json,
        window_hr=args.window_hr,
        min_valid_fraction=args.min_valid_fraction,
        support_fraction=args.support_fraction,
    )
    print(
        f"WALD_HSI_MSI_QNR QNR={quality['QNR']:.6f} "
        f"Dlambda={quality['Dlambda']:.6f} Ds={quality['Ds']:.6f} "
        f"SPECTRAL_BANDS=242 SPATIAL_REFERENCE=real_HR_MSI"
    )
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"QNR_JSON={path}")


if __name__ == "__main__":
    main()
