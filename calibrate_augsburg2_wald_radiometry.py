"""Calibrate Augsburg Region-2 radiometry without 10m HSI supervision.

The official Sentinel-2 SRF remains fixed. Fit 4 gains/biases between the
observed 30m EnMAP-like HSI projected through that SRF and the REAL S2
10m MSI aggregated to the 30m grid. A small GLOBAL lag (diagnostic only)
is permitted during the fit to reduce misregistration contamination.
No observations are warped/saved, no EnMAP10 data is accessed, and no
SRF or earlier radiometry file is overwritten.

The fitted affine is applied to RAW, unregistered S2 at training/inference;
the lag is for calibration only. This does not spatially register the input.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
from scipy.ndimage import map_coordinates, minimum_filter

from augsburg_real import _require_rasterio, infer_s2_platform
from diagnose_augsburg2_srf import (
    NAMES, _fit_affine, _metrics, _stable_masks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="30m domain Augsburg-2 Wald radiometry (fixed measured SRF)"
    )
    parser.add_argument("--wald_root", default="./data/augsburg2_wald")
    parser.add_argument("--real_cache_root", default="./data/augsburg_real_cache")
    parser.add_argument(
        "--output",
        default="./data/calibration/Augsburg2_Wald_radiometry.json",
    )
    parser.add_argument("--shift_y_30m", type=float, default=0.0)
    parser.add_argument("--shift_x_30m", type=float, default=-0.5)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--stable_fraction", type=float, default=0.75)
    parser.add_argument("--min_valid_pixels", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_shift(image, dy, dx):
    """Sample from image[y+dy, x+dx]; keep output frame unaltered."""
    h, w, bands = image.shape
    ys, xs = np.indices((h, w), dtype=np.float64)
    return np.stack([
        map_coordinates(
            image[..., band],
            (ys + dy, xs + dx),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        for band in range(bands)
    ], axis=-1).astype(np.float32)


def _platform_check(full_meta, real_cache_root):
    protocol_path = os.path.join(real_cache_root, "protocol.json")
    if not os.path.isfile(protocol_path):
        raise FileNotFoundError(protocol_path)
    with open(protocol_path, "r", encoding="utf-8") as file:
        protocol = json.load(file)
    source = full_meta["s2_source"]
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    rasterio, _, _ = _require_rasterio()
    with rasterio.open(source) as dataset:
        region_platform = infer_s2_platform(dataset)
    srf_platform = protocol.get("s2_platform", "UNKNOWN")
    if (
        region_platform in ("S2A", "S2B", "S2C")
        and region_platform != srf_platform
    ):
        raise ValueError(
            f"Region-2 actual platform={region_platform} differs from SRF "
            f"platform={srf_platform}. Fix this before calibration."
        )
    if region_platform == "UNKNOWN":
        warnings.warn(
            "Region-2 Sentinel-2 platform could not be determined from GeoTIFF. "
            "Check the original satellite product metadata before training.",
            stacklevel=2,
        )
    print(
        f"WALD_RADIOMETRY_SENSOR region_platform={region_platform} "
        f"srf_platform={srf_platform} S2_SOURCE={source}"
    )
    return region_platform, srf_platform


def main():
    args = parse_args()
    for value in (args.shift_y_30m, args.shift_x_30m):
        if not np.isfinite(value) or abs(value) > 1.0:
            raise ValueError("Fit shift must be within +/-1 pixel on the 30m grid")
    if not 0.2 < args.train_fraction < 0.8:
        raise ValueError("--train_fraction must be between .2 and .8")
    if not 0.2 < args.stable_fraction <= 1.0:
        raise ValueError("--stable_fraction must be in (.2, 1]")

    full = os.path.join(args.wald_root, "full")
    with open(os.path.join(full, "meta.json"), "r", encoding="utf-8") as f:
        full_meta = json.load(f)
    if full_meta.get("region") != "sub_area_2":
        raise ValueError(
            f"Only Augsburg Region-2 is supported, got {full_meta.get('region')}"
        )
    if full_meta.get("s2_band_names") != list(NAMES):
        raise ValueError("Real S2 bands must be B2,B3,B4,B8 in that order")
    platform, srf_platform = _platform_check(full_meta, args.real_cache_root)

    lr = np.load(os.path.join(full, "lr_hsi.npy"), mmap_mode="r")
    s2 = np.load(os.path.join(full, "hr_msi.npy"), mmap_mode="r")
    hr_valid = np.asarray(
        np.load(os.path.join(full, "valid_mask.npy")), dtype=bool
    )
    srf = np.asarray(
        np.load(os.path.join(args.wald_root, "srf_weights.npy")),
        dtype=np.float64,
    )
    h, w, channels = lr.shape
    if channels != 242 or srf.shape != (4, 242):
        raise ValueError(f"HSI/SRF spectral shape mismatch: {lr.shape}, {srf.shape}")
    if s2.shape != (h * 3, w * 3, 4) or hr_valid.shape != (h * 3, w * 3):
        raise ValueError(f"Expected S2 {(h*3, w*3, 4)}, found {s2.shape}")
    if np.any(srf < 0) or not np.all(np.isfinite(srf)):
        raise ValueError("SRF must contain finite, nonnegative weights")
    if not np.allclose(srf.sum(axis=1), 1.0, rtol=0, atol=1e-4):
        raise ValueError("SRF rows must sum to 1")

    # Every input here is an observable HSI30 or real-S2 product. In
    # particular, no EnMAP10 GT, pseudo-HR labels, or old calibrated MSI.
    lr_array = np.asarray(lr, dtype=np.float32)
    s2_array = np.asarray(s2, dtype=np.float32)
    raw_s2_30 = s2_array.reshape(h, 3, w, 3, 4).mean(axis=(1, 3))
    valid = (
        hr_valid.reshape(h, 3, w, 3).all(axis=(1, 3))
        & np.isfinite(lr_array).all(axis=-1)
        & np.isfinite(raw_s2_30).all(axis=-1)
        & (lr_array >= -0.05).all(axis=-1)
        & (lr_array <= 1.5).all(axis=-1)
        & (raw_s2_30 >= -0.05).all(axis=-1)
        & (raw_s2_30 <= 1.5).all(axis=-1)
    )
    lr_array = np.nan_to_num(lr_array, nan=0., posinf=0., neginf=0.)
    raw_s2_30 = np.nan_to_num(raw_s2_30, nan=0., posinf=0., neginf=0.)
    projection = lr_array @ srf.T

    fit, hold, boundary, cut = _stable_masks(
        lr_array, raw_s2_30, valid,
        args.train_fraction, args.stable_fraction,
    )
    safe = minimum_filter(
        valid.astype(np.uint8), size=3, mode="constant", cval=0
    ).astype(bool)
    fit &= safe
    hold &= safe
    if int(fit.sum()) < args.min_valid_pixels or int(hold.sum()) < args.min_valid_pixels:
        raise ValueError(
            f"Insufficient safe train/holdout pixels: "
            f"train={int(fit.sum())} holdout={int(hold.sum())}"
        )
    aligned_for_fit = sample_shift(
        raw_s2_30, args.shift_y_30m, args.shift_x_30m
    )
    gains = []
    biases = []
    comparisons = []
    for b, band in enumerate(NAMES):
        target = projection[..., b]
        base = raw_s2_30[..., b]
        moved = aligned_for_fit[..., b]

        g0, z0 = _fit_affine(base[fit], target[fit])
        g1, z1 = _fit_affine(moved[fit], target[fit])
        baseline = _metrics(base[hold], target[hold], g0, z0)
        shifted = _metrics(moved[hold], target[hold], g1, z1)
        if not np.isfinite([g1, z1]).all() or g1 <= 0:
            raise ValueError(f"Invalid fitted radiometry for {band}: {g1},{z1}")
        gains.append(float(g1))
        biases.append(float(z1))
        row = {
            "band": band,
            "raw_fit_gain": float(g0), "raw_fit_bias": float(z0),
            "aligned_fit_gain": float(g1), "aligned_fit_bias": float(z1),
            "holdout_unshifted": baseline,
            "holdout_fit_shift": shifted,
            "holdout_corr_gain": float(shifted["corr"] - baseline["corr"]),
            "holdout_rmse_reduction": float(
                (baseline["rmse_after_affine"] - shifted["rmse_after_affine"])
                / max(baseline["rmse_after_affine"], 1e-12)
            ),
        }
        comparisons.append(row)
        print(
            f"WALD_RADIOMETRY {band} "
            f"GAIN={g1:.6f} BIAS={z1:.6f} "
            f"CORR_0={baseline['corr']:.6f} "
            f"CORR_FIT_SHIFT={shifted['corr']:.6f} "
            f"RMSE_0={baseline['rmse_after_affine']:.8f} "
            f"RMSE_FIT_SHIFT={shifted['rmse_after_affine']:.8f} "
            f"RMSE_REDUCTION={row['holdout_rmse_reduction']:.5f}"
        )

    result = {
        "dataset": "Augsburg-2-Wald",
        "split": "region2_30m_spatial_train_only",
        "domain": "projected_EnMAP30_vs_real_S2_area30",
        "bands": list(NAMES),
        "gain": gains,
        "bias": biases,
        "model": "R(EnMAP30) ~= gain * mean_pool3(S2_10m) + bias",
        "calibration_lag_only_30m_pixels": {
            "dy": args.shift_y_30m, "dx": args.shift_x_30m,
        },
        "alignment_applied_to_network_input": False,
        "SRF_modified": False,
        "uses_EnMAP10_reference": False,
        "sensor_platform_detected": platform,
        "sensor_platform_SRF": srf_platform,
        "fit_pixels": int(fit.sum()), "holdout_pixels": int(hold.sum()),
        "split_column": boundary, "stable_gradient_cut": cut,
        "holdout_results": comparisons,
        "warning": (
            "Calibration-only offset reduces geometric contamination of "
            "radiometry fit. Never apply it as a presumed ground-truth "
            "nonregistration warp."
        ),
    }
    target_path = os.path.abspath(args.output)
    if os.path.exists(target_path) and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {target_path}. Pass --overwrite to replace it."
        )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(
        f"WALD_RADIOMETRY_SAVED={target_path} "
        "ACTIVE_SRF_UNMODIFIED=1 INPUT_GEOMETRY_UNMODIFIED=1"
    )


if __name__ == "__main__":
    main()
