"""Inspect and optionally propose a *bounded* Augsburg-2 Sentinel-2 SRF correction.

Never uses EnMAP10, never edits active SRF files, never fits an unrestricted
4x242 mixing matrix. Fits only band-centre shifts and small width changes
around the measured S2A/S2B spectral response curves, with spatial holdout.

The HSI is the Region-2 30m EnMAP-like image. Real 10m S2 is block-
averaged to 30m. Sensor/time/PSF/misregistration mismatch remains a
confounder: a low spectral residual is NOT proof of physical SRF correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from srf_utils import estimate_band_widths, interp_srf_to_hsi_wavelengths
from augsburg_real import infer_s2_platform, _require_rasterio

NAMES = ("B2", "B3", "B4", "B8")


def parse_args():
    p = argparse.ArgumentParser(description="Augsburg-2 SRF integrity and heldout calibration diagnostic")
    p.add_argument("--wald_root", default="./data/augsburg2_wald")
    p.add_argument("--real_cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--srf_csv", default="", help="Measured SRF CSV override, otherwise from real-cache protocol.json")
    p.add_argument("--srf_columns", default="", help="Comma-separated B2,B3,B4,B8 SRF CSV column names")
    p.add_argument("--mode", choices=("diagnose", "fit"), default="diagnose")
    p.add_argument("--output_dir", default="./data/calibration/augsburg2_srf_check")
    p.add_argument("--train_fraction", type=float, default=0.70)
    p.add_argument("--stable_fraction", type=float, default=0.75)
    p.add_argument("--max_shift_nm", type=float, default=4.0)
    p.add_argument("--shift_step_nm", type=float, default=2.0)
    p.add_argument("--width_fraction", type=float, default=0.04)
    p.add_argument("--min_corr_gain", type=float, default=0.005)
    p.add_argument("--min_rmse_reduction", type=float, default=0.02)
    p.add_argument("--max_registration_corr_gain", type=float, default=0.02)
    return p.parse_args()


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denom) if denom > 1e-18 else float("nan")


def _fit_affine(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    vx = np.mean((x - x.mean()) ** 2)
    if vx < 1e-12:
        return 1.0, 0.0
    gain = np.mean((x - x.mean()) * (y - y.mean())) / vx
    return float(gain), float(y.mean() - gain * x.mean())


def _metrics(x, y, gain, bias):
    residual = gain * np.asarray(x, np.float64) + bias - np.asarray(y, np.float64)
    return {
        "corr": _pearson(x, y),
        "l1_after_affine": float(np.mean(np.abs(residual))),
        "rmse_after_affine": float(np.sqrt(np.mean(residual ** 2))),
    }


def _resolve_official_csv(args):
    meta_path = os.path.join(args.real_cache_root, "protocol.json")
    protocol = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            protocol = json.load(f)
    csv_path = args.srf_csv or protocol.get("srf_path", "")
    columns = ([v.strip() for v in args.srf_columns.split(",") if v.strip()]
               if args.srf_columns else protocol.get("srf_band_columns", []))
    if not csv_path or not os.path.isfile(csv_path):
        return None, columns, protocol
    if len(columns) != 4:
        raise ValueError("Must have four SRF columns in B2,B3,B4,B8 order")
    actual = protocol.get("s2_platform", "UNKNOWN")
    detected = protocol.get("s2_platform_detected", "UNKNOWN")
    if actual in ("S2A", "S2B", "S2C") and detected not in ("UNKNOWN", actual):
        warnings.warn(
            f"Platform override={actual} differs from detected={detected}; SRF is not verified",
            stacklevel=2,
        )
    for b, col in zip(NAMES, columns):
        if not str(col).upper().endswith(" " + b):
            raise ValueError(f"Expected column for {b}, got {col}")
        if actual in ("S2A", "S2B") and not str(col).upper().startswith(actual):
            warnings.warn(f"{b}: {col} does not match metadata platform={actual}", stacklevel=2)
    return csv_path, columns, protocol


def _stable_masks(hsi, msi, valid, frac_train, frac_stable):
    """Spatial holdout, optionally exclude high-contrast edges before spectral fit."""
    h, w, _ = hsi.shape
    # Coarse broadband gradients are useful to screen likely PSF/registration
    # mismatches; selection uses no 10m HSI references.
    hgray = np.mean(hsi, axis=2)
    mgray = np.mean(msi, axis=2)
    gh = np.hypot(*np.gradient(hgray))
    gm = np.hypot(*np.gradient(mgray))
    if not np.any(valid):
        raise ValueError("No valid paired HSI/MSI pixels")
    ghscale = float(np.median(gh[valid])) + 1e-6
    gmscale = float(np.median(gm[valid])) + 1e-6
    score = gh / ghscale + gm / gmscale
    cut = float(np.quantile(score[valid], frac_stable))
    reliable = valid & (score <= cut)
    split_col = min(w - 2, max(2, int(round(w * frac_train))))
    xx = np.arange(w)[None, :]
    # Exclude a one-pixel buffer across holdout boundary.
    train = reliable & (xx < split_col - 1)
    heldout = reliable & (xx > split_col + 1)
    if int(train.sum()) < 100 or int(heldout.sum()) < 100:
        raise ValueError(
            f"Insufficient stable split pixels: train={int(train.sum())} "
            f"holdout={int(heldout.sum())}"
        )
    return train, heldout, int(split_col), float(cut)


def _geometry_lag_diagnostics(hproj, msi, mask):
    """Cross-check whether a +/-1 low-resolution-pixel shift improves correlation."""
    h, w = mask.shape
    result = []
    for b in range(4):
        ref = hproj[..., b]
        actual = msi[..., b]
        base = _pearson(actual[mask], ref[mask])
        best = {"dy": 0, "dx": 0, "corr": base}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y0, y1 = max(0, -dy), min(h, h - dy)
                x0, x1 = max(0, -dx), min(w, w - dx)
                if y0 >= y1 or x0 >= x1:
                    continue
                r = ref[y0:y1, x0:x1]
                s = actual[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
                both = (mask[y0:y1, x0:x1]
                        & mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx])
                if int(both.sum()) < 100:
                    continue
                corr = _pearson(s[both], r[both])
                if np.isfinite(corr) and corr > best["corr"]:
                    best = {"dy": dy, "dx": dx, "corr": float(corr)}
        result.append({
            "band": NAMES[b], "zero_shift_corr": float(base), "best_lag": best,
            "lag_corr_gain": float(best["corr"] - base),
        })
    return result


def _heldout_fractional_lag_test(hproj, msi, train, heldout, valid, step=0.5):
    """Fit a shared 30m-grid offset on TRAIN and report its effect on HOLDOUT.

    Compare candidate shifts on precisely the same eroded mask to avoid
    gaining correlation by excluding difficult pixels or selecting edges.
    Each candidate obtains its own training-only affine radiometry fit.
    A competing *zero-lag Gaussian blur* is optimized exclusively on the
    training subset, then evaluated on the same spatial holdout. This
    distinguishes (part of) subpixel resampling's smoothing benefit from
    genuine translation; it cannot identify exact sensor geolocation error.
    This is only a diagnostic; it neither warps cached MSI nor edits SRF.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates, minimum_filter

    h, w, channels = hproj.shape
    if hproj.shape != msi.shape or channels != 4:
        raise ValueError("Lag diagnostic needs matched HxWx4 projection and MSI")
    # Safe for bilinear sampling anywhere within +/- 1 LR pixel.
    safe = minimum_filter(valid.astype(np.uint8), size=3, mode="constant",
                          cval=0).astype(bool)
    tr, ho = train & safe, heldout & safe
    if int(tr.sum()) < 100 or int(ho.sum()) < 100:
        print(f"SRF_LAG_HOLD_SKIPPED train={int(tr.sum())} holdout={int(ho.sum())}")
        return {"status": "insufficient_safe_pixels",
                "train_pixels": int(tr.sum()), "holdout_pixels": int(ho.sum())}

    ys, xs = np.indices((h, w), dtype=np.float64)
    offsets = np.arange(-1., 1. + step / 2., step, dtype=np.float64)
    predictions = {}
    best = {"dy": 0., "dx": 0., "train_mean_corr": -float("inf")}
    for dy in offsets:
        for dx in offsets:
            shifted = np.stack([
                map_coordinates(msi[..., band], [ys + dy, xs + dx],
                                order=1, mode="nearest", prefilter=False)
                for band in range(4)
            ], axis=-1)
            mean_corr = float(np.mean([
                _pearson(shifted[..., band][tr], hproj[..., band][tr])
                for band in range(4)
            ]))
            if not np.isfinite(mean_corr):
                continue
            predictions[(float(dy), float(dx))] = shifted
            if mean_corr > best["train_mean_corr"]:
                best = {"dy": float(dy), "dx": float(dx),
                        "train_mean_corr": mean_corr}
    if not predictions:
        raise ValueError("No finite candidates for 30m subpixel lag diagnosis")

    aligned = predictions[(best["dy"], best["dx"])]
    unaligned = predictions[(0., 0.)]
    # Symmetric smoothing without a change of pixel coordinates: if this
    # achieves essentially the same holdout gain as a half-pixel shift,
    # translation cannot be inferred from correlation improvements alone.
    # Select the blur width ONLY on the training subset. Keep the same mask
    # as zero-shift / shifted candidates.
    blur_candidates = [
        (0., 0.), (0., .35), (0., .5), (0., .75), (0., 1.),
        (.35, .35), (.5, .5), (.75, .75), (1., 1.),
    ]
    best_blur = None
    best_blur_train_corr = -float("inf")
    blurred = unaligned
    for sigma_y, sigma_x in blur_candidates:
        candidate = (
            msi if sigma_y == 0. and sigma_x == 0.
            else gaussian_filter(
                msi, sigma=(sigma_y, sigma_x, 0.), mode="nearest"
            )
        )
        corr = float(np.mean([
            _pearson(candidate[..., b][tr], hproj[..., b][tr])
            for b in range(4)
        ]))
        if np.isfinite(corr) and corr > best_blur_train_corr:
            best_blur_train_corr = corr
            best_blur = {"sigma_y": sigma_y, "sigma_x": sigma_x,
                         "train_mean_corr": corr}
            blurred = candidate
    if best_blur is None:
        raise RuntimeError("No finite zero-displacement blur candidate")
    print(
        f"SRF_BLUR_SELECTED SIGMA_30M=({best_blur['sigma_y']:.2f},"
        f"{best_blur['sigma_x']:.2f}) TRAIN_MEAN_CORR="
        f"{best_blur_train_corr:.6f}"
    )
    band_report = []
    for b, name in enumerate(NAMES):
        ref = hproj[..., b]
        x0 = unaligned[..., b]
        x1 = aligned[..., b]
        xb = blurred[..., b]
        g0, z0 = _fit_affine(x0[tr], ref[tr])
        g1, z1 = _fit_affine(x1[tr], ref[tr])
        gb, zb = _fit_affine(xb[tr], ref[tr])
        base = _metrics(x0[ho], ref[ho], g0, z0)
        moved = _metrics(x1[ho], ref[ho], g1, z1)
        blur = _metrics(xb[ho], ref[ho], gb, zb)
        record = {
            "band": name, "baseline": base, "candidate": moved,
            "blur_control": blur,
            "holdout_corr_gain": float(moved["corr"] - base["corr"]),
            "holdout_rmse_reduction": float(
                (base["rmse_after_affine"] - moved["rmse_after_affine"])
                / max(base["rmse_after_affine"], 1e-12)
            ),
            "shift_minus_blur_corr": float(
                moved["corr"] - blur["corr"]
            ),
            "shift_minus_blur_rmse_reduction": float(
                (blur["rmse_after_affine"] - moved["rmse_after_affine"])
                / max(blur["rmse_after_affine"], 1e-12)
            ),
        }
        band_report.append(record)
        print(
            f"SRF_LAG_HOLD {name} SHIFT_30M=({best['dy']:.2f},{best['dx']:.2f}) "
            f"CORR_0={base['corr']:.6f} CORR_SHIFT={moved['corr']:.6f} "
            f"CORR_GAIN={record['holdout_corr_gain']:.6f} "
            f"RMSE_0={base['rmse_after_affine']:.8f} "
            f"RMSE_SHIFT={moved['rmse_after_affine']:.8f} "
            f"RMSE_REDUCTION={record['holdout_rmse_reduction']:.5f}"
        )
        print(
            f"SRF_BLUR_CONTROL {name} "
            f"SHIFT_30M=({best['dy']:.2f},{best['dx']:.2f}) "
            f"BLUR_SIGMA_30M=({best_blur['sigma_y']:.2f},"
            f"{best_blur['sigma_x']:.2f}) "
            f"CORR_BLUR={blur['corr']:.6f} "
            f"CORR_SHIFT={moved['corr']:.6f} "
            f"SHIFT_VS_BLUR_CORR_GAIN={record['shift_minus_blur_corr']:.6f} "
            f"RMSE_BLUR={blur['rmse_after_affine']:.8f} "
            f"RMSE_SHIFT={moved['rmse_after_affine']:.8f} "
            f"SHIFT_VS_BLUR_RMSE_REDUCTION="
            f"{record['shift_minus_blur_rmse_reduction']:.5f}"
        )
    result = {
        "status": "ok", "selected_on": "stable_train_pixels",
        "evaluated_on": "disjoint_spatial_holdout_same_eroded_mask",
        "best_global_shift_lr_pixels": best,
        "best_zero_lag_blur_lr_pixels": best_blur,
        "train_pixels": int(tr.sum()), "holdout_pixels": int(ho.sum()),
        "bands": band_report,
    }
    return result


def _warped_srf(original_csv, column, wavelengths, widths, center, shift_nm, width_factor):
    # Physical, smooth shift/stretch of measured instrument response, never
    # independently adjust 242 individual SRF coefficients.
    sampled_at = center + (wavelengths - center - shift_nm) / width_factor
    response = interp_srf_to_hsi_wavelengths(
        original_csv["WL(nm)"].to_numpy(np.float32),
        original_csv[column].to_numpy(np.float32),
        sampled_at.astype(np.float32),
        interp_kind="pchip",
    )
    row = np.maximum(response * widths, 0.).astype(np.float64)
    if row.sum() <= 1e-12:
        raise RuntimeError(f"No SRF support at offset={shift_nm} width={width_factor}")
    return (row / row.sum()).astype(np.float32)


def main():
    args = parse_args()
    if not 0.2 < args.train_fraction < 0.8:
        raise ValueError("--train_fraction must be between 0.2 and 0.8")
    if not 0.2 < args.stable_fraction <= 1.0:
        raise ValueError("--stable_fraction must be in (0.2,1.0]")
    if args.max_shift_nm < 0 or args.shift_step_nm <= 0:
        raise ValueError("Invalid shift search settings")
    if not 0 <= args.width_fraction <= .1:
        raise ValueError("--width_fraction must be between 0 and 0.1")

    full = os.path.join(args.wald_root, "full")
    hsi = np.asarray(np.load(os.path.join(full, "lr_hsi.npy")), dtype=np.float32)
    msi_hr = np.asarray(np.load(os.path.join(full, "hr_msi.npy")), dtype=np.float32)
    hr_valid = np.asarray(np.load(os.path.join(full, "valid_mask.npy")), dtype=bool)
    old_srf = np.asarray(
        np.load(os.path.join(args.wald_root, "srf_weights.npy")), dtype=np.float32
    )
    wavelengths = np.asarray(
        np.load(os.path.join(args.wald_root, "hsi_wavelengths.npy")), dtype=np.float32
    )
    h, w, c = hsi.shape
    if c != 242 or old_srf.shape != (4, c) or wavelengths.size != c:
        raise ValueError("Expected 242 HSI and (4,242) SRF")
    if msi_hr.shape != (3 * h, 3 * w, 4) or hr_valid.shape != (3 * h, 3 * w):
        raise ValueError("Full-resolution S2 and 30m HSI must have exact x3 sizes")
    if not np.isfinite(old_srf).all() or np.min(old_srf) < 0:
        raise ValueError("Initial SRF has invalid negative or non-finite weights")
    if not np.allclose(old_srf.sum(axis=1), 1, rtol=0, atol=1e-4):
        raise ValueError(f"Initial SRF not normalized: {old_srf.sum(axis=1)}")

    msi = msi_hr.reshape(h, 3, w, 3, 4).mean(axis=(1, 3))
    valid = (hr_valid.reshape(h, 3, w, 3).all(axis=(1, 3))
             & np.isfinite(hsi).all(axis=-1) & np.isfinite(msi).all(axis=-1))
    valid &= ((hsi >= -0.05).all(axis=-1) & (hsi <= 1.5).all(axis=-1)
              & (msi >= -0.05).all(axis=-1) & (msi <= 1.5).all(axis=-1))
    # In case nan data is outside mask, prevent NaNs from propagating into
    # gradients and BLAS products.
    hsi = np.nan_to_num(hsi, nan=0., posinf=0., neginf=0.)
    msi = np.nan_to_num(msi, nan=0., posinf=0., neginf=0.)
    ref = hsi @ old_srf.T

    train, val, boundary, gradient_cut = _stable_masks(
        hsi, msi, valid, args.train_fraction, args.stable_fraction
    )
    lag = _geometry_lag_diagnostics(ref, msi, valid)
    heldout_lag = _heldout_fractional_lag_test(
        ref, msi, train, val, valid, step=0.5
    )
    csv_path, columns, protocol = _resolve_official_csv(args)
    with open(os.path.join(full, "meta.json"), "r", encoding="utf-8") as f:
        full_meta = json.load(f)
    s2_source = full_meta.get("s2_source", "")
    region_platform = "UNKNOWN"
    if s2_source and os.path.isfile(s2_source):
        rasterio, _, _ = _require_rasterio()
        with rasterio.open(s2_source) as source_ds:
            region_platform = infer_s2_platform(source_ds)
    cached_platform = protocol.get("s2_platform", "UNKNOWN")
    if region_platform in ("S2A", "S2B", "S2C") and cached_platform != region_platform:
        warnings.warn(
            f"Region-2 source is {region_platform} but cache SRF is {cached_platform}. "
            "Resolve platform SRF mismatch before considering an empirical SRF fit.",
            stacklevel=1,
        )
    print(
        f"SRF_SENSOR_CHECK REGION_PLATFORM={region_platform} "
        f"CACHE_PLATFORM={cached_platform} "
        f"BAND_NAMES={full_meta.get('s2_band_names', [])} "
        f"S2_SOURCE={s2_source}"
    )
    if full_meta.get("s2_band_names") != list(NAMES):
        raise ValueError(
            f"Region-2 S2 band order must be B2/B3/B4/B8, got "
            f"{full_meta.get('s2_band_names')}"
        )
    report = {
        "source": "Augsburg-2 30m HSI vs area-aggregated real 10m S2",
        "uses_enmap10": False,
        "msi_source": "real_Sentinel_2",
        "bands": list(NAMES), "shape30": list(hsi.shape),
        "valid_pixels": int(valid.sum()),
        "fit_pixels": int(train.sum()), "holdout_pixels": int(val.sum()),
        "spatial_holdout_boundary_column": boundary,
        "stable_quantile_cut": gradient_cut,
        "initial_srf_row_sums": [float(v) for v in old_srf.sum(axis=1)],
        "srf_csv": csv_path, "srf_columns": columns,
        "cache_platform": protocol.get("s2_platform", "UNKNOWN"),
        "cache_platform_detected": protocol.get("s2_platform_detected", "UNKNOWN"),
        "region2_s2_platform_detected": region_platform,
        "region2_s2_source": s2_source,
        "region2_s2_band_indexes": full_meta.get("s2_indexes"),
        "registration_shift_diagnostic": lag,
        "registration_fractional_holdout_diagnostic": heldout_lag,
        "original": [],
        "candidates": [],
    }

    original_affine = []
    for b in range(4):
        g, z = _fit_affine(msi[..., b][train], ref[..., b][train])
        original_affine.append((g, z))
        record = {
            "band": NAMES[b], "gain": g, "bias": z,
            "fit": _metrics(msi[..., b][train], ref[..., b][train], g, z),
            "holdout": _metrics(msi[..., b][val], ref[..., b][val], g, z),
        }
        report["original"].append(record)
        print(
            f"SRF_BASE {NAMES[b]} HOLD_CORR={record['holdout']['corr']:.6f} "
            f"HOLD_L1={record['holdout']['l1_after_affine']:.8f} "
            f"HOLD_RMSE={record['holdout']['rmse_after_affine']:.8f} "
            f"GAIN={g:.6f} BIAS={z:.6f} "
            f"BEST_LAG={lag[b]['best_lag']['dy']},{lag[b]['best_lag']['dx']} "
            f"LAG_GAIN={lag[b]['lag_corr_gain']:.6f}"
        )

    if csv_path:
        table = pd.read_csv(csv_path)
        widths = estimate_band_widths(wavelengths).astype(np.float64)
        # Verify exact correspondence of cached row with measured SRF source.
        repro_err = []
        for b in range(4):
            resp = interp_srf_to_hsi_wavelengths(
                table["WL(nm)"].values.astype(np.float32),
                table[columns[b]].values.astype(np.float32),
                wavelengths,
                interp_kind="pchip",
            )
            rr = resp.astype(np.float64) * widths
            rr /= rr.sum()
            repro_err.append(float(np.max(np.abs(rr - old_srf[b]))))
        report["source_reprojection_max_abs"] = repro_err
        print("SRF_SOURCE_CHECK " + " ".join(
            f"{name}={err:.9g}" for name, err in zip(NAMES, repro_err)
        ))
        if max(repro_err) > 1e-4:
            warnings.warn(
                "Cached SRF does not reproduce from the specified CSV; "
                "resolve SRF source mismatch before fitting", stacklevel=1,
            )
            if args.mode == "fit":
                raise ValueError("Refusing to fit from an SRF CSV different from the cached baseline")
    else:
        print("SRF_SOURCE_CHECK CSV_UNAVAILABLE; baseline diagnostic still valid")
        if args.mode == "fit":
            raise FileNotFoundError(
                "Fitting needs the original measured S2 SRF CSV. Supply --srf_csv and --srf_columns."
            )

    candidate = old_srf.copy()
    candidate_gains = [r["gain"] for r in report["original"]]
    candidate_biases = [r["bias"] for r in report["original"]]
    if args.mode == "fit":
        if heldout_lag.get("status") == "ok":
            geometry_warning_bands = [
                row["band"] for row in heldout_lag["bands"]
                if row["holdout_corr_gain"] > 0.005
                and row["holdout_rmse_reduction"] > 0.005
            ]
            if len(geometry_warning_bands) >= 2:
                raise ValueError(
                    "Holdout test supports a common spatial offset in "
                    f"{geometry_warning_bands}. Resolve geometry/radiometry "
                    "before fitting SRF; active SRF has not been changed."
                )
        if region_platform in ("S2A", "S2B", "S2C") and cached_platform != region_platform:
            raise ValueError(
                "Cannot fit: Region-2 platform does not match the cached SRF satellite"
            )
        # A small grid is intentional. An unrestricted 4x242 SRF is
        # non-identifiable in the presence of geometry/radiometry mismatch.
        shifts = np.arange(
            -args.max_shift_nm,
            args.max_shift_nm + args.shift_step_nm * 0.5,
            args.shift_step_nm,
        )
        width_factors = (1.,) if args.width_fraction == 0 else (
            1. - args.width_fraction, 1., 1. + args.width_fraction
        )
        for b in range(4):
            raw = table[columns[b]].to_numpy(np.float64)
            wl_raw = table["WL(nm)"].to_numpy(np.float64)
            ctr = float(np.sum(wl_raw * np.maximum(raw, 0.))
                        / max(1e-12, np.sum(np.maximum(raw, 0.))))
            orig = report["original"][b]
            best_train = -float("inf")
            best = None
            for shift in shifts:
                for width in width_factors:
                    row = _warped_srf(
                        table, columns[b], wavelengths, widths,
                        center=ctr, shift_nm=float(shift), width_factor=float(width),
                    )
                    synth = hsi @ row
                    xx = msi[..., b]
                    g, z = _fit_affine(xx[train], synth[train])
                    fit = _metrics(xx[train], synth[train], g, z)
                    hold = _metrics(xx[val], synth[val], g, z)
                    # Corr^2 is affine-invariant; small quadratic prior
                    # discourages fitting geometry/radiometry as SRF changes.
                    score = fit["corr"] ** 2 - (
                        .005 * (float(shift) / max(args.max_shift_nm, 1.)) ** 2
                        + .005 * ((float(width) - 1.) / max(args.width_fraction, 0.01)) ** 2
                    )
                    if not np.isfinite(score):
                        continue
                    if score > best_train:
                        best_train = score
                        best = (row, g, z, float(shift), float(width), fit, hold)
            if best is None:
                raise RuntimeError(f"No finite SRF candidates for {NAMES[b]}")
            row, g, z, shift, width, fit, hold = best
            corr_gain = hold["corr"] - orig["holdout"]["corr"]
            rmse_reduction = (
                (orig["holdout"]["rmse_after_affine"] - hold["rmse_after_affine"])
                / max(orig["holdout"]["rmse_after_affine"], 1e-12)
            )
            reliable_geometry = lag[b]["lag_corr_gain"] <= args.max_registration_corr_gain
            accepted = bool(
                reliable_geometry
                and corr_gain >= args.min_corr_gain
                and rmse_reduction >= args.min_rmse_reduction
                and abs(shift) <= args.max_shift_nm
            )
            if accepted:
                candidate[b] = row
                candidate_gains[b], candidate_biases[b] = g, z
            record = {
                "band": NAMES[b], "shift_nm": shift, "width_factor": width,
                "fit": fit, "holdout": hold, "corr_gain": float(corr_gain),
                "relative_rmse_reduction": float(rmse_reduction),
                "registration_safe": bool(reliable_geometry),
                "accepted": accepted,
            }
            report["candidates"].append(record)
            print(
                f"SRF_CANDIDATE {NAMES[b]} SHIFT_NM={shift:.2f} WIDTH={width:.3f} "
                f"HOLD_CORR={hold['corr']:.6f} CORR_GAIN={corr_gain:.6f} "
                f"RMSE_REDUCTION={rmse_reduction:.4f} ACCEPTED={int(accepted)}"
            )
        report["candidate_accepted_bands"] = [
            x["band"] for x in report["candidates"] if x["accepted"]
        ]
        # Never write/overwrite the active SRF in the Wald or Real cache.
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, "srf_weights_candidate.npy"), candidate)
        with open(os.path.join(args.output_dir, "radiometry_candidate.json"), "w", encoding="utf-8") as f:
            json.dump({
                "gain": candidate_gains, "bias": candidate_biases,
                "bands": list(NAMES),
                "protocol": "Augsburg-2_Wald_30m_stable_pixels_train_spatial",
                "warning": "candidate only; requires independent confirmation before use",
            }, f, indent=2)
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, "srf_diagnostic.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"SRF_DIAGNOSTIC_REPORT={output} mode={args.mode} "
          "ACTIVE_SRF_UNMODIFIED=1 ACTIVE_RADIOMETRY_UNMODIFIED=1")


if __name__ == "__main__":
    main()
