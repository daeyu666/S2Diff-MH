"""Augsburg-2 Wald classical QNR *formulas* with an explicit PAN source.

QNR (Alparone 2008) requires a single higher-resolution PAN image P and a
lower-resolution blurred/downsampled PAN P_L.  The real Augsburg inputs have
NO true PAN; their four Sentinel-2 bands can only define a synthetic proxy.
The default score is therefore labelled *proxy-PAN standard-form QNR*, not a
measurement of classical real-PAN QNR.

F = fused 10m HSI242, H = observed 30m HSI242, M = real S2 MSI10m4.
R = fixed 4x242 S2 SRF.  The 4-channel MS-style pair is:
  F4 = R(F), H4 = R(H).
For P use either externally provided HR/LR PAN pair (required for actual
PAN-based QNR), or P=mean4(calibrated M), P_L=area_mean3(P).  The proxy
is reproducible but is NOT a sensor PAN and NOT evidence of 242-band fidelity.

  Dlambda = (1/6) sum_i<j |UIQI(F4_i,F4_j) - UIQI(H4_i,H4_j)|
  Ds = (1/4) sum_i |UIQI(F4_i,P) - UIQI(H4_i,P_L)|
  QNR = max(0,1-Dlambda) * max(0,1-Ds)

Exponents p=q=alpha=beta=1; masked local UIQI uses paired nonoverlapping
48x48/16x16 windows by default, >=80% valid fraction, valid-pixel weights.
Both repositories use the EXACT same implementation and spectral inputs.
This is distinct from the superseded 16-cross-band "modified" D_s.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _masked_uiqi(a, b, mask, window, min_valid_fraction=0.8):
    """Average local UIQI, weighting each window by number of valid pixels."""
    if a.shape != b.shape or a.shape != mask.shape or a.ndim != 2:
        raise ValueError("UIQI spatial arrays must share 2D shape")
    h, w = a.shape
    value_sum, weight_sum = 0.0, 0.0
    for y in range(0, h, window):
        for x in range(0, w, window):
            yh = min(y + window, h)
            xw = min(x + window, w)
            tile_mask = mask[y:yh, x:xw]
            total = tile_mask.size
            valid = tile_mask & np.isfinite(a[y:yh, x:xw]) & np.isfinite(b[y:yh, x:xw])
            n = int(valid.sum())
            if n < 8 or n / total < float(min_valid_fraction):
                continue
            aa = np.asarray(a[y:yh, x:xw][valid], dtype=np.float64)
            bb = np.asarray(b[y:yh, x:xw][valid], dtype=np.float64)
            ma, mb = float(aa.mean()), float(bb.mean())
            va = float(((aa - ma) ** 2).mean())
            vb = float(((bb - mb) ** 2).mean())
            cov = float(((aa - ma) * (bb - mb)).mean())
            luminance = 2.0 * ma * mb / (ma * ma + mb * mb + 1e-12)
            contrast = 1.0 if va + vb < 1e-12 else 2.0 * cov / (va + vb)
            value_sum += float(np.clip(luminance * contrast, -1.0, 1.0)) * n
            weight_sum += n
    if weight_sum == 0:
        raise ValueError("No valid UIQI windows; check spatial mask and window sizes")
    return value_sum / weight_sum


def standard_qnr(
    fused,
    observed_lr_hsi,
    observed_hr_msi,
    valid_hr,
    srf_weights,
    gains=None,
    biases=None,
    window_hr=48,
    min_valid_fraction=0.8,
    *,
    pan_hr=None,
    pan_lr=None,
):
    """Classical QNR equations on four SRF-projected HSI bands.

    With pan_hr/pan_lr absent, constructs a fixed **synthetic** PAN from the
    calibrated four real MSI bands (unweighted mean).  It is never labelled
    real-PAN QNR.  To evaluate with a genuine PAN sensor, provide BOTH actual
    HR PAN and its precomputed sensor-MTF-matched LR image.
    """
    fused = np.asarray(fused)
    lr = np.asarray(observed_lr_hsi)
    msi = np.asarray(observed_hr_msi)
    mask = np.asarray(valid_hr, dtype=bool)
    r = np.asarray(srf_weights, dtype=np.float64)
    if fused.ndim != 3 or fused.shape[-1] != 242:
        raise ValueError("Fused prediction must be HxWx242")
    h, w, _ = fused.shape
    if r.shape != (4, 242) or msi.shape != (h, w, 4) or mask.shape != (h, w):
        raise ValueError("Expected SRF 4x242, MSI HxWx4 and mask HxW")
    if h % 3 or w % 3 or lr.shape != (h // 3, w // 3, 242):
        raise ValueError("Observed 30m HSI must match one-third of 10m dimensions")
    if not isinstance(window_hr, int) or window_hr < 24 or window_hr % 3:
        raise ValueError("window_hr must be an integer >=24 divisible by 3")
    if not (0 < min_valid_fraction <= 1):
        raise ValueError("min_valid_fraction must be in (0,1]")
    if not np.isfinite(r).all() or (r < 0).any() or not np.allclose(r.sum(1), 1., atol=1e-4):
        raise ValueError("Invalid fixed Sentinel-2 response matrix")
    gain = np.ones(4, dtype=np.float32) if gains is None else np.asarray(gains, dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32) if biases is None else np.asarray(biases, dtype=np.float32)
    if gain.shape != (4,) or bias.shape != (4,) or not np.isfinite(gain).all() or not np.isfinite(bias).all():
        raise ValueError("Radiometry requires four finite gains and biases")
    if (pan_hr is None) != (pan_lr is None):
        raise ValueError("Classical PAN QNR requires both pan_hr and pan_lr; no silent PAN degradation")

    # Same observed MSI calibration as inference, with no geometric pre-warp.
    corrected_msi = (
        msi.astype(np.float32) * gain[None, None, :] + bias[None, None, :]
    )
    high_ms = (fused.astype(np.float32) @ r.T.astype(np.float32)).astype(np.float32)
    low_ms = (lr.astype(np.float32) @ r.T.astype(np.float32)).astype(np.float32)
    mask_low = mask.reshape(h // 3, 3, w // 3, 3).all(axis=(1, 3))

    if pan_hr is None:
        # This is not a real PAN: four-band mean is an explicit reproducible
        # intensity proxy.  Applying true-PAN QNR formulas does not change that.
        high_pan = corrected_msi.mean(axis=2, dtype=np.float32)
        low_pan = high_pan.reshape(h // 3, 3, w // 3, 3).mean(axis=(1, 3))
        pan_origin = "synthetic_equal_mean_of_four_calibrated_real_S2_bands"
        is_genuine_pan = False
    else:
        high_pan = np.asarray(pan_hr, dtype=np.float32)
        low_pan = np.asarray(pan_lr, dtype=np.float32)
        if high_pan.shape != (h, w) or low_pan.shape != (h // 3, w // 3):
            raise ValueError("Supplied PAN must have aligned 10m/30m HxW shapes")
        pan_origin = "external_user_supplied_PAN_and_degraded_PAN"
        is_genuine_pan = True  # Provenance is user-declared, not sensor-verified.

    hiwin, lowin = window_hr, window_hr // 3
    def highq(a, b):
        return _masked_uiqi(a, b, mask, hiwin, min_valid_fraction)
    def lowq(a, b):
        return _masked_uiqi(a, b, mask_low, lowin, min_valid_fraction)

    spectral = [
        abs(highq(high_ms[:, :, i], high_ms[:, :, j]) -
            lowq(low_ms[:, :, i], low_ms[:, :, j]))
        for i in range(4) for j in range(i + 1, 4)
    ]
    spatial = [
        abs(highq(high_ms[:, :, i], high_pan) -
            lowq(low_ms[:, :, i], low_pan))
        for i in range(4)
    ]
    dl, ds = float(np.mean(spectral)), float(np.mean(spatial))
    return {
        "QNR": float(max(0., 1. - dl) * max(0., 1. - ds)),
        "Dlambda": dl,
        "Ds": ds,
        "index_name": "classical-QNR-equations/observed-PAN" if is_genuine_pan
                      else "classical-QNR-equations/synthetic-MSI-mean-PAN-proxy",
        "qnr_equations": "Alparone-2008-classical-formula-p=q=alpha=beta=1",
        "pan_origin": pan_origin,
        "is_genuine_pan": is_genuine_pan,
        "low_resolution_multispectral_reference": "SRF_projected_observed_30m_HSI",
        "spectral_pair_count": len(spectral),
        "spatial_pair_count": len(spatial),
        "high_window": int(hiwin),
        "low_window": int(lowin),
        "window_min_valid_fraction": float(min_valid_fraction),
        "high_valid_pixels": int(mask.sum()),
        "low_valid_pixels": int(mask_low.sum()),
        "spectral_domain": "four Sentinel-2 SRF-projected channels (not 242-band fidelity)",
        "source_HSI": "observed_30m_HSI",
        "source_MSI": "observed_real_Sentinel-2_10m",
        "full_HR_HSI_ground_truth_used": False,
    }


def evaluate_cache(wald_root, fused, radiometry_json, window_hr=48,
                   min_valid_fraction=0.8, *, pan_hr=None, pan_lr=None):
    root = Path(wald_root)
    with (root / "full" / "meta.json").open(encoding="utf-8") as f:
        full_meta = json.load(f)
    if full_meta.get("region") != "sub_area_2":
        raise ValueError("Strict Wald evaluation requires original-scale sub_area_2")

    # Common validation used in both repositories; no EnMAP10 label is read.
    for split in ("train", "validation", "test"):
        with (root / split / "meta.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        if (meta.get("msi_source") != "real_Sentinel_2_Wald_30m"
            or meta.get("target") != "30m_EnMAP_like"
            or meta.get("gt_source") != "observed_30m_HSI_only"
            or int(meta.get("scale_ratio", -1)) != 3):
            raise ValueError("Expected strict Augsburg-2 Wald cache: " + split)

    with (root / "wald_psf.json").open(encoding="utf-8") as f:
        psf = json.load(f)
    if int(psf.get("scale_ratio", -1)) != 3:
        raise ValueError("Expected Wald x3 operator")
    with Path(radiometry_json).open(encoding="utf-8") as f:
        rad = json.load(f)
    if (rad.get("dataset") != "Augsburg-2-Wald"
        or rad.get("uses_EnMAP10_reference") is not False):
        raise ValueError("Require train-only Augsburg2 Wald radiometry (no EnMAP10)")
    gain, bias = np.asarray(rad["gain"], dtype=np.float32), np.asarray(rad["bias"], dtype=np.float32)
    if gain.shape != (4,) or bias.shape != (4,) or not np.isfinite(gain).all() or not np.isfinite(bias).all():
        raise ValueError("Expected four finite radiometry gains and biases")
    return standard_qnr(
        np.load(fused, mmap_mode="r"),
        np.load(root / "full" / "lr_hsi.npy", mmap_mode="r"),
        np.load(root / "full" / "hr_msi.npy", mmap_mode="r"),
        np.load(root / "full" / "valid_mask.npy", mmap_mode="r"),
        np.load(root / "srf_weights.npy"),
        gains=gain, biases=bias,
        window_hr=window_hr, min_valid_fraction=min_valid_fraction,
        pan_hr=None if pan_hr is None else np.load(pan_hr, mmap_mode="r"),
        pan_lr=None if pan_lr is None else np.load(pan_lr, mmap_mode="r"),
    )


def main():
    p = argparse.ArgumentParser(
        description="Augsburg original-scale classical-formula QNR with an explicit PAN proxy"
    )
    p.add_argument("--wald_root", default="./data/augsburg2_wald")
    p.add_argument("--fused", required=True)
    p.add_argument("--radiometry_json", default="./data/calibration/Augsburg2_Wald_radiometry.json")
    p.add_argument("--window_hr", type=int, default=48)
    p.add_argument("--min_valid_fraction", type=float, default=0.8)
    p.add_argument("--pan_hr", default="", help="Optional real 10m PAN .npy (requires --pan_lr)")
    p.add_argument("--pan_lr", default="", help="Optional PSF-degraded 30m PAN .npy (requires --pan_hr)")
    p.add_argument("--output_json", default="")
    args = p.parse_args()
    results = evaluate_cache(
        args.wald_root, args.fused, args.radiometry_json,
        window_hr=args.window_hr, min_valid_fraction=args.min_valid_fraction,
        pan_hr=args.pan_hr or None, pan_lr=args.pan_lr or None,
    )
    print(
        f"WALD_STANDARD_FORM_QNR QNR={results['QNR']:.6f} "
        f"Dlambda={results['Dlambda']:.6f} Ds={results['Ds']:.6f} "
        f"PAN_ORIGIN={results['pan_origin']} GENUINE_PAN={results['is_genuine_pan']}"
    )
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"QNR_JSON={path}")


if __name__ == "__main__":
    main()
