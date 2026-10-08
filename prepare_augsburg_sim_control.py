"""Build the official MDAS simulated-MSI control from Augsburg-Real caches.

This creates a SEPARATE cache for a controlled ablation:
    identical EnMAP10/EnMAP30/valid mask/SRF,
    but replaces real Sentinel-2 B2/B3/B4/B8 with the official EeteS
    simulated Sentinel-2 MSI from the same geographic split.

No image-content registration, fitting, or artificial degradation is done.
Validation/test are prepared but never used to fit any parameter.
Large identical HSI .npy arrays are hard-linked by default, not duplicated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from augsburg_real import _reproject_multiband, _require_rasterio, _target_profile


NAMES = ("B2", "B3", "B4", "B8")
SPLITS = ("train", "validation", "test")
SHARED_NAMES = ("gt.npy", "lr_hsi.npy", "valid_mask.npy")
ROOT_NAMES = ("srf_weights.npy", "hsi_wavelengths.npy")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build separate MDAS official simulated-MSI control cache"
    )
    p.add_argument("--real_cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--output_cache_root", default="./data/augsburg_sim_control_cache")
    p.add_argument("--link_mode", choices=("hardlink", "copy"), default="hardlink")
    p.add_argument("--sim_scale", type=float, default=10000.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _select_sim_indexes(count):
    """Official EeteS MSI can be provided as four selected bands or full 12."""
    if count == 4:
        return [1, 2, 3, 4]
    if count == 12:
        return [2, 3, 4, 8]
    raise ValueError(f"Expected 4- or 12-band simulated MSI GeoTIFF, got count={count}")


def _simulated_path(gt_source):
    # Verified against the official MDAS deep-super-resolution dataset loader.
    name = os.path.basename(gt_source)
    if "EeteS_EnMAP_10m_" not in name:
        raise ValueError(f"Unrecognized MDAS EnMAP10 reference: {gt_source}")
    return os.path.join(
        os.path.dirname(gt_source),
        name.replace("EeteS_EnMAP_10m_", "EeteS_Sentinel_2_10m_", 1),
    )


def _shared(src, dst, mode, overwrite):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if src == dst:
        raise ValueError("Output cannot overwrite the Augsburg-Real source cache")
    if os.path.exists(dst):
        if not overwrite:
            raise FileExistsError(f"Output exists: {dst}; pass --overwrite to recreate")
        os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError as exc:
            raise OSError(
                f"Cannot hard-link {src} to {dst}. "
                "Use --link_mode copy if these paths are on different filesystems."
            ) from exc
    else:
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    real_root = os.path.abspath(args.real_cache_root)
    output_root = os.path.abspath(args.output_cache_root)
    if real_root == output_root:
        raise ValueError("output_cache_root must differ from real_cache_root")
    if args.sim_scale <= 0:
        raise ValueError("--sim_scale must be positive")

    # Check preconditions before creating any output.
    descriptions = {}
    rasterio, _, _ = _require_rasterio()
    for split in SPLITS:
        source_dir = os.path.join(real_root, split)
        meta_path = os.path.join(source_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        sim_path = _simulated_path(meta["gt_source"])
        if not os.path.isfile(sim_path):
            raise FileNotFoundError(
                f"Official MDAS simulated MSI for {split} not found: {sim_path}"
            )
        with rasterio.open(sim_path) as src:
            indexes = _select_sim_indexes(int(src.count))
        gt = np.load(os.path.join(source_dir, "gt.npy"), mmap_mode="r")
        lr = np.load(os.path.join(source_dir, "lr_hsi.npy"), mmap_mode="r")
        valid = np.load(os.path.join(source_dir, "valid_mask.npy"), mmap_mode="r")
        if gt.shape[:2] != valid.shape or gt.shape[:2] != (
            lr.shape[0] * 3, lr.shape[1] * 3
        ):
            raise ValueError(f"Invalid Augsburg-Real x3 cache geometry: {split}")
        descriptions[split] = (meta, sim_path, indexes, tuple(gt.shape[:2]))

    for name in ROOT_NAMES:
        source = os.path.join(real_root, name)
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        dest = os.path.join(output_root, name)
        if os.path.exists(dest) and not args.overwrite:
            raise FileExistsError(f"Control-cache output already exists: {dest}")
    for split in SPLITS:
        dest_dir = os.path.join(output_root, split)
        for name in SHARED_NAMES + ("hr_msi.npy", "meta.json"):
            dest = os.path.join(dest_dir, name)
            if os.path.exists(dest) and not args.overwrite:
                raise FileExistsError(f"Control-cache output already exists: {dest}")

    os.makedirs(output_root, exist_ok=True)
    for name in ROOT_NAMES:
        _shared(
            os.path.join(real_root, name),
            os.path.join(output_root, name),
            args.link_mode,
            args.overwrite,
        )

    summary = {
        "dataset": "Augsburg-Simulated-MSI-Control",
        "real_cache_root": real_root,
        "output_cache_root": output_root,
        "msi_source": "official_EeteS_simulated_Sentinel_2",
        "geometric_registration": "metadata_only",
        "gt_lr_and_mask": "shared_with_Augsburg_Real",
        "radiometry_json": "MUST_BE_EMPTY_for_control",
        "splits": {},
    }
    for split in SPLITS:
        meta, sim_path, indexes, (height, width) = descriptions[split]
        source_dir = os.path.join(real_root, split)
        out_dir = os.path.join(output_root, split)
        os.makedirs(out_dir, exist_ok=True)
        for name in SHARED_NAMES:
            _shared(
                os.path.join(source_dir, name),
                os.path.join(out_dir, name),
                args.link_mode,
                args.overwrite,
            )

        crs, transform, _, _ = _target_profile(meta["gt_source"])
        sim = _reproject_multiband(
            sim_path,
            target_crs=crs,
            target_transform=transform,
            target_width=width,
            target_height=height,
            indexes=indexes,
            scale=args.sim_scale,
            resampling="bilinear",
        )
        if sim.shape != (height, width, 4):
            raise AssertionError(f"Unexpected simulated MSI shape: {sim.shape}")

        # Keep exactly the same valid-pixel mask as the real-MSI baseline.
        mask = np.load(
            os.path.join(source_dir, "valid_mask.npy"), mmap_mode="r"
        ).astype(bool)
        if not np.isfinite(sim[mask]).all():
            raise ValueError(
                f"Simulated MSI is non-finite inside existing real valid_mask: {split}"
            )

        sim_dst = os.path.join(out_dir, "hr_msi.npy")
        np.save(sim_dst, sim.astype(np.float32))
        updated = dict(meta)
        updated["s2_source"] = sim_path
        updated["msi_source"] = "official_EeteS_simulated_Sentinel_2"
        updated["s2_platform"] = "simulated_product"
        updated["s2_band_indexes_1based"] = indexes
        updated["s2_band_names"] = list(NAMES)
        updated["msi_shape"] = list(sim.shape)
        updated["valid_fraction"] = float(mask.mean())
        updated["radiometry_correction"] = "none"
        updated["metadata_alignment_only"] = True
        updated["content_registration"] = False

        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(updated, handle, indent=2)
        summary["splits"][split] = {
            "simulated_source": sim_path,
            "selected_band_indexes": indexes,
            "hr_shape": updated["hr_shape"],
            "lr_shape": updated["lr_shape"],
            "msi_shape": updated["msi_shape"],
            "valid_fraction": updated["valid_fraction"],
        }
        print(
            f"SIM_CONTROL_SPLIT split={split} "
            f"msi_shape={sim.shape} source_indexes={indexes} "
            f"valid_fraction={float(mask.mean()):.6f}"
        )
    print("AUGSBURG_SIM_CONTROL " + json.dumps(summary))


if __name__ == "__main__":
    main()
