"""Prepare Augsburg-2 Wald reduced-resolution training and full-resolution input.

TRAIN/VAL reduced-resolution targets come exclusively from EeteS_EnMAP_30m
and original Sentinel-2. EnMAP10 is never loaded or needed.

Wald training:
  reference 30 m HSI -> synthetic 90 m LR-HSI (factor 3)
  real 10 m S2 -> 30 m MSI (factor 3, spatial area integration)
  30 m HSI is the reference target.

Full inference:
  sub_area_2 30 m EnMAP-like HSI + 10 m real Sentinel-2
  -> 10 m HSI without a reference or full-resolution validation score.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from augsburg_real import (
    _reproject_multiband,
    _require_rasterio,
    _target_profile,
    find_augsburg_root,
    resolve_s2_band_indexes,
)
from degradations.effective_gaussian import EffectiveGaussianDegradation


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Augsburg-2 Wald no-HR-HSI-label protocol")
    p.add_argument("--real_cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--output_root", default="./data/augsburg2_wald")
    p.add_argument("--subarea", default="sub_area_2")
    p.add_argument("--sigma", type=float, default=1.2,
                   help="Fixed RR degradation sigma in 30m pixels, not fitted using EnMAP10")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def downsample_hsi_wald(hr_hsi, sigma):
    """Use the exact network's effective-Gaussian operator, on CPU in bands."""
    h, w, c = hr_hsi.shape
    if h % 3 or w % 3:
        raise ValueError("HSI reference must be divisible by 3")
    op = EffectiveGaussianDegradation(scale_ratio=3, terminal_sigma=sigma).cpu()
    result = np.empty((h // 3, w // 3, c), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, c, 12):
            end = min(c, start + 12)
            x = torch.from_numpy(
                np.ascontiguousarray(hr_hsi[:, :, start:end])
            ).permute(2, 0, 1).unsqueeze(0).float()
            pred = op.degrade(x)[0].permute(1, 2, 0).cpu().numpy()
            result[:, :, start:end] = pred
    return result


def mean_downsample_msi(hr_msi, factor=3):
    h, w, c = hr_msi.shape
    if h % factor or w % factor:
        raise ValueError("MSI spatial dimensions must be divisible by downsample factor")
    return np.ascontiguousarray(hr_msi).reshape(
        h // factor, factor, w // factor, factor, c
    ).mean(axis=(1, 3)).astype(np.float32)


def _write_array(path, arr, overwrite):
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError(path + " already exists; use --overwrite")
    np.save(path, arr)


def prepare_rr(args):
    for name in ("srf_weights.npy", "hsi_wavelengths.npy"):
        source = os.path.join(args.real_cache_root, name)
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        _write_array(
            os.path.join(args.output_root, name),
            np.load(source),
            args.overwrite,
        )

    # Use published deep_train / deep_valid geographic partitions; never
    # inspect their original EnMAP10 GT arrays.
    for split in ("train", "validation", "test"):
        source_dir = os.path.join(args.real_cache_root, split)
        source_meta = os.path.join(source_dir, "meta.json")
        with open(source_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        hsi30 = np.load(os.path.join(source_dir, "lr_hsi.npy"), mmap_mode="r")
        s2_10 = np.load(os.path.join(source_dir, "hr_msi.npy"), mmap_mode="r")
        validity10 = np.load(
            os.path.join(source_dir, "valid_mask.npy"), mmap_mode="r"
        )
        # Original mask includes EnMAP10 validity. Use only finite-range checks
        # on directly observed EnMAP30 and real S2 to avoid accessing EnMAP10.
        h, w = (int(hsi30.shape[0]) // 3 * 3,
                int(hsi30.shape[1]) // 3 * 3)
        if h < 6 or w < 6:
            raise ValueError(f"Wald reduced-resolution region is too small: {split}")
        gt = np.asarray(hsi30[:h, :w]).copy().astype(np.float32)
        msi30 = mean_downsample_msi(np.asarray(s2_10[:h * 3, :w * 3]))
        lr90 = downsample_hsi_wald(gt, args.sigma)
        valid_hsi = np.isfinite(gt).all(axis=-1) & (gt >= -0.05).all(axis=-1) & (gt <= 1.5).all(axis=-1)
        real_patch = np.asarray(s2_10[:h * 3, :w * 3])
        valid_msi10 = np.isfinite(real_patch).all(axis=-1) & (real_patch >= -0.05).all(axis=-1) & (real_patch <= 1.5).all(axis=-1)
        valid_msi30 = valid_msi10.reshape(h, 3, w, 3).all(axis=(1, 3))
        valid = (valid_hsi & valid_msi30).astype(np.uint8)
        out = os.path.join(args.output_root, split)
        os.makedirs(out, exist_ok=True)
        for name, value in (
            ("gt.npy", gt), ("lr_hsi.npy", lr90),
            ("hr_msi.npy", msi30), ("valid_mask.npy", valid),
        ):
            _write_array(os.path.join(out, name), value, args.overwrite)
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "split": split,
                "protocol": "Wald reduced-resolution; no 10m HSI labels",
                "supervision_source": meta["lr_source"],
                "msi_source": "real_Sentinel_2_Wald_30m",
                "target": "30m_EnMAP_like",
                "gt_source": "observed_30m_HSI_only",
                "hr_shape": list(gt.shape),
                "lr_shape": list(lr90.shape),
                "msi_shape": list(msi30.shape),
                "scale_ratio": 3,
                "valid_fraction": float(valid.mean()),
                "wald_sigma": args.sigma,
                "scl_mask_used": False,
            }, f, indent=2)
        print(f"WALD_{split.upper()} TARGET={gt.shape} LR={lr90.shape} MSI={msi30.shape} valid={valid.mean():.5f}")


def prepare_full(args):
    # sub_area_2 must not be silently substituted with sub_area_1.
    root = find_augsburg_root(args.data_root)
    region = os.path.join(root, args.subarea)
    suffix = args.subarea.replace("sub_area_", "sub_area")
    hsi_path = os.path.join(region, f"EeteS_EnMAP_30m_{suffix}.tif")
    s2_candidates = [os.path.join(region, "Sentinel-2.tif"),
                     os.path.join(region, "Sentinel_2.tif")]
    s2_path = next((p for p in s2_candidates if os.path.isfile(p)), None)
    if not os.path.isfile(hsi_path):
        raise FileNotFoundError(
            f"Region-2 observed LR-HSI missing: {hsi_path}; "
            "check actual MDAS sub_area_2 filename, do not substitute sub_area_1"
        )
    if s2_path is None:
        raise FileNotFoundError(
            "Region-2 real MSI missing: " + ", ".join(s2_candidates)
        )

    rasterio, _, _ = _require_rasterio()
    from affine import Affine
    crs, transform, width, height = _target_profile(s2_path)
    h_hr, w_hr = (height // 3 * 3, width // 3 * 3)
    with rasterio.open(s2_path) as src:
        s2_indexes, band_names = resolve_s2_band_indexes(src)
    msi = _reproject_multiband(
        s2_path, target_crs=crs, target_transform=transform,
        target_width=w_hr, target_height=h_hr,
        indexes=s2_indexes, scale=10000., resampling="bilinear",
    )
    lr = _reproject_multiband(
        hsi_path, target_crs=crs,
        target_transform=transform * Affine.scale(3, 3),
        target_width=w_hr // 3, target_height=h_hr // 3,
        scale=10000., resampling="bilinear",
    )
    if lr.shape[-1] != 242 or msi.shape[-1] != 4:
        raise ValueError(f"Unexpected Region-2 spectral channels: LR {lr.shape}, MSI {msi.shape}")
    hr_valid = (
        np.isfinite(msi).all(axis=-1) & (msi >= -0.05).all(axis=-1)
        & (msi <= 1.5).all(axis=-1)
    )
    lr_valid = (
        np.isfinite(lr).all(axis=-1) & (lr >= -0.05).all(axis=-1)
        & (lr <= 1.5).all(axis=-1)
    )
    mask = hr_valid & np.repeat(np.repeat(lr_valid, 3, axis=0), 3, axis=1)
    out = os.path.join(args.output_root, "full")
    os.makedirs(out, exist_ok=True)
    for name, value in (
        ("lr_hsi.npy", lr.astype(np.float32)),
        ("hr_msi.npy", msi.astype(np.float32)),
        ("valid_mask.npy", mask.astype(np.uint8)),
    ):
        _write_array(os.path.join(out, name), value, args.overwrite)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "protocol": "Full-resolution Augsburg Region-2, no HR-HSI reference",
            "hsi30_source": hsi_path, "s2_source": s2_path,
            "region": args.subarea, "lr_shape": list(lr.shape),
            "msi_shape": list(msi.shape), "s2_indexes": s2_indexes,
            "s2_band_names": band_names, "valid_fraction": float(mask.mean()),
            "transform_6": list(tuple(transform)[:6]), "crs": str(crs),
            "radiometry": "train_only_calibration_may_be_applied_at_inference",
        }, f, indent=2)
    print(f"WALD_FULL region={args.subarea} LR_HSI={lr.shape} HR_MSI={msi.shape} valid={mask.mean():.5f}")


def main():
    args = parse_args()
    if args.sigma < 0:
        raise ValueError("--sigma must be nonnegative")
    os.makedirs(args.output_root, exist_ok=True)
    prepare_rr(args)
    prepare_full(args)
    path = os.path.join(args.output_root, "wald_psf.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "Augsburg-2-Wald",
            "terminal_sigma_hr_pixels": args.sigma,
            "scale_ratio": 3,
            "stages": [1, 2, 3],
            "sigma_source": "fixed_before_10m_inference_no_EnMAP10_supervision",
        }, f, indent=2)
    print("WALD_PSF " + path)


if __name__ == "__main__":
    main()
