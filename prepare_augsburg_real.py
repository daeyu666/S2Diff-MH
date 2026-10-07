"""Prepare metadata-aligned Augsburg-Real caches.

This script performs only CRS/grid harmonisation.  It does not estimate any
image-content shift/rotation/non-rigid registration.
"""

from __future__ import annotations

import argparse
import json
import os

from augsburg_real import (
    S2_NATIVE10_BANDS,
    _require_rasterio,
    find_augsburg_root,
    infer_s2_platform,
    prepare_augsburg_real_cache,
    resolve_real_s2_path,
)


def _default_srf(platform: str) -> tuple[str, list[str]]:
    here = os.path.dirname(os.path.abspath(__file__))
    platform = platform.upper()
    suffix = platform[-1].lower()
    version = "v4" if platform == "S2A" else "v3"
    path = os.path.join(
        here,
        "data",
        "srf",
        f"sentinel2{suffix}_srf_{version}_B2_B3_B4_B8.csv",
    )
    cols = [f"{platform} {band}" for band in S2_NATIVE10_BANDS]
    return path, cols


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Augsburg-Real train/val/test cache")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--real_s2_path", default="")
    p.add_argument("--scl_path", default="")
    p.add_argument("--s2_platform", choices=["auto", "S2A", "S2B"], default="auto")
    p.add_argument("--srf_path", default="")
    p.add_argument("--srf_band_columns", default="")
    p.add_argument("--s2_scale", type=float, default=10000.0)
    p.add_argument("--enmap_scale", type=float, default=10000.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = find_augsburg_root(args.data_root)
    s2_path = resolve_real_s2_path(root, args.real_s2_path)
    rasterio, _, _ = _require_rasterio()
    with rasterio.open(s2_path) as src:
        detected = infer_s2_platform(src)

    platform = detected if args.s2_platform == "auto" else args.s2_platform
    if platform == "UNKNOWN":
        raise RuntimeError(
            "Sentinel-2 platform could not be inferred from GeoTIFF metadata. "
            "Pass --s2_platform S2A or S2B after checking the original product metadata."
        )

    default_path, default_cols = _default_srf(platform)
    srf_path = os.path.abspath(args.srf_path or default_path)
    if not os.path.exists(srf_path):
        raise FileNotFoundError(
            f"No SRF resource for detected platform={platform}: {srf_path}. "
            "Provide the official platform-specific B2/B3/B4/B8 SRF with --srf_path."
        )
    if args.srf_band_columns:
        columns = [x.strip() for x in args.srf_band_columns.split(",") if x.strip()]
    else:
        columns = default_cols
    if len(columns) != 4:
        raise ValueError("--srf_band_columns must contain exactly four columns")

    print(
        f"AUGSBURG_REAL_PREP root={root} real_s2={s2_path} "
        f"platform_detected={detected} platform_used={platform}"
    )
    print(f"SRF path={srf_path} columns={columns}")
    print(
        "ALIGNMENT_POLICY metadata_only=True content_registration=False "
        "real_S2_bands=B2,B3,B4,B8"
    )
    summary = prepare_augsburg_real_cache(
        data_root=args.data_root,
        cache_root=args.cache_root,
        real_s2_path=s2_path,
        srf_path=srf_path,
        srf_band_columns=columns,
        scl_path=args.scl_path,
        s2_reflectance_scale=args.s2_scale,
        enmap_reflectance_scale=args.enmap_scale,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
