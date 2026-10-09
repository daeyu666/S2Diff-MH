"""Prepare a leakage-controlled *central holdout* on Augsburg-2 real Wald x3.

DRT-Net-inspired, not a claim to reproduce undocumented exact Fig.13 pixels.
Default central test block is 48x48 in the observed 30m HSI grid, therefore
144x144 on the real Sentinel-2 10m grid. Test 30m bbox = [24:72,36:84],
native 10m bbox = [72:216,108:252]. Region2 original 30m is 100x120.
A 6-pixel (30m) train exclusion margin exceeds Gaussian PSF truncation.

Training uses only tiles completely OUTSIDE the dilated holdout rectangle;
both a hard sample rejection and an invalid training mask are written.
Validation is the same disjoint deep_valid source as the original strict Wald.
The full scene is retained solely to create a red-box overview and native
raw observations; FULL REGION QNR must NOT be reported as held-out QNR.
Existing cache and checkpoints remain untouched.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from prepare_augsburg2_wald import downsample_hsi_wald, mean_downsample_msi


def _rects_intersect(a, b):
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    return ay0 < by1 and by0 < ay1 and ax0 < bx1 and bx0 < ax1


def heldout_train_tile_candidates(shape, bbox, *, patch=24, stride=6, guard=6):
    """Enumerate spatially independent training tiles, never with 80%-mask loopholes."""
    h, w = shape
    y0, x0, y1, x1 = bbox
    forbidden = (
        max(0, y0 - guard), max(0, x0 - guard),
        min(h, y1 + guard), min(w, x1 + guard)
    )
    return [
        (top, left, patch, patch)
        for top in range(0, h - patch + 1, stride)
        for left in range(0, w - patch + 1, stride)
        if not _rects_intersect((top, left, top + patch, left + patch), forbidden)
    ]


def prepare(src_root, dest_root, *, test_height=48, test_width=48,
            guard=6, train_patch=24, train_stride=6):
    src = Path(src_root)
    dest = Path(dest_root)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing center holdout: {dest}. "
            "Use a new output path for every protocol version."
        )
    for key in ("lr_hsi.npy", "hr_msi.npy", "valid_mask.npy", "meta.json"):
        if not (src / "full" / key).exists():
            raise FileNotFoundError(src / "full" / key)
    with (src / "full" / "meta.json").open(encoding="utf-8") as f:
        full_meta = json.load(f)
    if full_meta.get("region") != "sub_area_2":
        raise ValueError("Only MDAS Region 2 original observations allowed")
    with (src / "wald_psf.json").open(encoding="utf-8") as f:
        psf = json.load(f)
    if int(psf.get("scale_ratio", -1)) != 3:
        raise ValueError("Wald operator must have scale_ratio=3")
    sigma = float(psf["terminal_sigma_hr_pixels"])
    if guard < 4 or any(z % 3 for z in (test_height, test_width, train_patch, train_stride)):
        raise ValueError("Test sizes, train patch/stride must be divisible by 3; guard >=4")
    if train_patch % 24:
        raise ValueError("UAFL train patch must be divisible by 24")

    hsi_full = np.load(src / "full" / "lr_hsi.npy", mmap_mode="r")
    msi_full = np.load(src / "full" / "hr_msi.npy", mmap_mode="r")
    valid_full = np.load(src / "full" / "valid_mask.npy", mmap_mode="r")
    if hsi_full.ndim != 3 or hsi_full.shape[-1] != 242:
        raise ValueError("Observed EnMAP-like HSI must contain 242 bands")
    h, w = hsi_full.shape[:2]
    if msi_full.shape != (h * 3, w * 3, 4) or valid_full.shape != (h * 3, w * 3):
        raise ValueError("Actual MSI10/HSI30 grids or mask do not match")
    # Align ROI edges to the 30m and 90m sampling grid, and to 6-pixel
    # progressive-process divisibility, without inventing author Fig.13 coords.
    y0 = ((h - test_height) // 2 // 6) * 6
    x0 = ((w - test_width) // 2 // 6) * 6
    y1, x1 = y0 + test_height, x0 + test_width
    bbox = (y0, x0, y1, x1)
    # Gaussian PSF acts on original 30m HSI; crop only AFTER degradation
    # to preserve the same sampling phase, sensor blur and LR native data.
    hh, ww = (h // 3 * 3), (w // 3 * 3)
    if y1 > hh or x1 > ww:
        raise ValueError("Holdout ROI cannot fit the phase-aligned Wald source")
    gt = np.asarray(hsi_full[:hh, :ww], dtype=np.float32).copy()
    hr_msi30 = mean_downsample_msi(np.asarray(msi_full[:hh*3, :ww*3]))
    hr_valid = np.asarray(valid_full[:hh*3, :ww*3], dtype=bool)
    valid = hr_valid.reshape(hh, 3, ww, 3).all(axis=(1, 3))
    valid &= np.isfinite(gt).all(axis=2)
    lr90 = downsample_hsi_wald(gt, sigma)

    train_tiles = heldout_train_tile_candidates(
        (hh, ww), bbox, patch=train_patch, stride=train_stride, guard=guard
    )
    if not train_tiles:
        raise RuntimeError("No training tiles outside holdout and PSF guard")
    forbidden = (
        max(0, y0 - guard), max(0, x0 - guard),
        min(hh, y1 + guard), min(ww, x1 + guard),
    )
    # Make excluded gt/MSI/LR90 pixels inaccessible even to accidental code
    # which neglects hard tile exclusion. The active train loader also MUST
    # filter full tile extents against forbidden_bbox_30m.
    train_valid = valid.copy()
    train_valid[forbidden[0]:forbidden[2], forbidden[1]:forbidden[3]] = False
    train_gt = gt.copy()
    train_gt[~train_valid] = 0.
    train_msi = hr_msi30.copy()
    train_msi[~train_valid] = 0.
    train_lr = lr90.copy()
    bad_lr = ~train_valid.reshape(hh//3, 3, ww//3, 3).all(axis=(1, 3))
    train_lr[bad_lr] = 0.
    n_eligible = sum(
        float(train_valid[t:t+ph, l:l+pw].mean()) >= .8
        for t, l, ph, pw in train_tiles
    )
    if not n_eligible:
        raise RuntimeError("No valid training patches outside center/guard at 80% valid threshold")
    test_gt = gt[y0:y1, x0:x1].copy()
    test_lr = lr90[y0//3:y1//3, x0//3:x1//3].copy()
    test_msi = hr_msi30[y0:y1, x0:x1].copy()
    test_valid = valid[y0:y1, x0:x1].copy()
    if float(test_valid.mean()) < .8:
        raise ValueError("Central test area has insufficient valid observations")

    manifest = {
        "protocol_id": "Augsburg2-Wald-center-holdout-v1",
        "protocol": "DRT-Net-inspired central spatial holdout; exact paper ROI not published",
        "source_region": "sub_area_2",
        "spatial_reference": "observed_30m_HSI",
        "train_gt": "observed_30m_HSI_outside_guard_only",
        "test_gt": "observed_30m_HSI_center_only",
        "test_bbox_30m": list(bbox),
        "test_bbox_10m": [3*y0, 3*x0, 3*y1, 3*x1],
        "forbidden_bbox_30m": list(forbidden),
        "guard_pixels_30m": int(guard),
        "guard_exceeds_3sigma_psf": bool(guard >= np.ceil(3*sigma)),
        "train_patch_30m": int(train_patch),
        "train_stride_30m": int(train_stride),
        "validation": "disjoint_official_deep_valid",
        "test_roi_shape_30m": list(test_gt.shape),
        "test_roi_shape_10m": [3*test_height, 3*test_width],
        "origin_hsi30_shape": list(hsi_full.shape),
        "eligible_train_tiles_at_80pct": int(n_eligible),
        "sigma_30m": sigma,
        "no_10m_HSI_GT": True,
        "full_region_used_as_train_GT": False,
        "full_region_QNR_not_heldout": True,
    }

    for name in ("srf_weights.npy", "hsi_wavelengths.npy", "wald_psf.json"):
        if not (src/name).exists():
            raise FileNotFoundError(src/name)
    dest.mkdir(parents=True, exist_ok=False) if not dest.exists() else None
    for name in ("srf_weights.npy", "hsi_wavelengths.npy", "wald_psf.json"):
        shutil.copy2(src / name, dest / name)
    for name in ("meta.json", "lr_hsi.npy", "hr_msi.npy", "valid_mask.npy"):
        output = dest / "full" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "full" / name, output)
    with (dest / "roi.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    train_meta = {
        "split": "train", "msi_source": "real_Sentinel_2_Wald_30m",
        "target": "30m_EnMAP_like", "gt_source": "observed_30m_HSI_only",
        "scale_ratio": 3, "wald_sigma": sigma,
        "protocol_id": manifest["protocol_id"],
        "forbidden_bbox_30m": list(forbidden),
        "test_bbox_30m": list(bbox),
        "geometry_full_scene": False,
    }
    test_meta = {
        "split": "test", "msi_source": "real_Sentinel_2_Wald_30m",
        "target": "30m_EnMAP_like", "gt_source": "observed_30m_HSI_only",
        "scale_ratio": 3, "wald_sigma": sigma,
        "protocol_id": manifest["protocol_id"],
        "test_bbox_30m": list(bbox),
        "geometry_full_scene": False,
    }
    for split, arrays, meta in (
        ("train", (train_gt, train_lr, train_msi, train_valid), train_meta),
        ("test", (test_gt, test_lr, test_msi, test_valid), test_meta),
    ):
        out = dest / split
        out.mkdir(parents=True, exist_ok=False)
        for name, value in zip(("gt", "lr_hsi", "hr_msi", "valid_mask"), arrays):
            np.save(out / (name + ".npy"), value.astype(np.uint8 if name=="valid_mask" else np.float32))
        with (out / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    shutil.copytree(src / "validation", dest / "validation")
    # The validation geography remains disjoint, but explicitly annotate
    # protocol ID so checkpoints/tests reject mixing old vs new splits.
    v_meta = dest / "validation" / "meta.json"
    with v_meta.open(encoding="utf-8") as f:
        data = json.load(f)
    data["protocol_id"] = manifest["protocol_id"]
    with v_meta.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    full_path = dest / "full" / "meta.json"
    with full_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    meta["protocol_id"] = manifest["protocol_id"]
    meta["roi_manifest"] = "roi.json"
    with full_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"HOLDOUT_CREATED={dest.resolve()} REGION=sub_area_2 "
        f"TEST_30M={list(bbox)} TEST_10M={manifest['test_bbox_10m']} "
        f"TRAIN_VALID_PATCHES={n_eligible} GUARD_30M={guard}"
    )
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_wald_root", default="./data/augsburg2_wald")
    p.add_argument("--output_root", default="./data/augsburg2_wald_center_holdout")
    p.add_argument("--test_size_30m", type=int, default=48)
    p.add_argument("--guard_30m", type=int, default=6)
    p.add_argument("--train_patch_30m", type=int, default=24)
    p.add_argument("--train_stride_30m", type=int, default=6)
    a = p.parse_args()
    prepare(
        a.source_wald_root, a.output_root,
        test_height=a.test_size_30m, test_width=a.test_size_30m,
        guard=a.guard_30m, train_patch=a.train_patch_30m,
        train_stride=a.train_stride_30m,
    )


if __name__ == "__main__":
    main()
