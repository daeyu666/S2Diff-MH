"""Visualize the EXACT Augsburg synthetic-x4 held-out test image and footprint.

This is NOT the Augsburg-2 Wald sub_area_2 image. It reads the same official
10m EnMAP-like GT TIFF selected by data_loader._build_augsburg(), then
visualizes the exact nonoverlapping evaluation grid from _nonoverlap_coords:
128x128 tiles at (0,0), (0,128), (128,0), (128,128).

No reconstruction, source substitution, rotation, resampling, or synthetic
data generation is performed. Visualization bands and display stretch are
never used as metric inputs.

Usage (S2Diff-MH repository root):
python visualize_augsburg_synthetic_test.py \
  --data_root ./data/raw \
  --output_dir ./outputs/augsburg_synthetic_test_visualization

Optional side-by-side with a previously produced UAFL full-region inference:
  --uafl_npy ../comparison_experiments/comparison/UAFL/outputs/augsburg2_wald/Augsburg2_Wald_UAFL_full_HSI.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_loader import _find_augsburg_root, _read_tiff_cube, _nonoverlap_coords


DEFAULT_RGB = (43, 28, 10)  # TorchGeo MDAS published 0-based EnMAP RGB indices
TEST_TILE = 128


def prepare_rgb(cube, indices=DEFAULT_RGB):
    """Use fixed band indices and physical reflectance, not latent features."""
    x = np.asarray(cube)
    if x.ndim != 3 or x.shape[-1] != 242:
        raise ValueError(f"Expected HxWx242 EnMAP cube, got {x.shape}")
    if len(indices) != 3 or any(k < 0 or k >= 242 for k in indices):
        raise ValueError("RGB indices must be three valid zero-based EnMAP indices")
    return np.nan_to_num(x[:, :, list(indices)].astype(np.float32), nan=0)


def shared_stretch(images):
    """Determine shared per-channel limits for honest cross-area rendering."""
    lo, hi = [], []
    for i in range(3):
        values = np.concatenate([
            im[:, :, i][np.isfinite(im[:, :, i])].reshape(-1)
            for im in images
        ])
        lo.append(float(np.percentile(values, 1.0)))
        hi.append(float(np.percentile(values, 99.0)))
    return np.array(lo, dtype=np.float32), np.array(hi, dtype=np.float32)


def render_rgb(rgb, low, high):
    stretched = (rgb - low.reshape(1, 1, 3)) / np.maximum(
        high - low, 1e-6
    ).reshape(1, 1, 3)
    return Image.fromarray((255.0 * np.clip(stretched, 0, 1)).round().astype(np.uint8), "RGB")


def overlay_exact_tiles(image, coords, tile=TEST_TILE):
    out = image.convert("RGBA")
    covered = np.zeros((image.height, image.width), dtype=bool)
    for top, left in coords:
        covered[top:top + tile, left:left + tile] = True

    # Reduce brightness OUTSIDE the official evaluated pixels, leaving all
    # in-test pixels and image geometry unmodified.
    mask = Image.fromarray((~covered).astype(np.uint8) * 120, "L")
    shade = Image.new("RGBA", out.size, (0, 0, 0, 0))
    shade.paste(Image.new("RGBA", out.size, (0, 0, 0, 190)), (0, 0), mask)
    out = Image.alpha_composite(out, shade)
    draw = ImageDraw.Draw(out)
    for i, (top, left) in enumerate(coords):
        draw.rectangle(
            [left, top, left + tile - 1, top + tile - 1],
            outline=(0, 255, 255, 255), width=2,
        )
        draw.text((left + 5, top + 5), f"T{i + 1}", fill=(255, 255, 255, 255),
                  stroke_width=1, stroke_fill=(0, 0, 0, 255))
    return out.convert("RGB")


def make_comparison(left, right, *, title_left, title_right):
    if left.size != right.size:
        # Scale ONLY the rendered preview panel, never the original arrays.
        right = right.resize(left.size, Image.Resampling.NEAREST)
    w, h = left.size
    margin, header = 22, 42
    canvas = Image.new("RGB", (w * 2 + 3 * margin, h + header + 2 * margin), "#141a21")
    canvas.paste(left, (margin, header + margin))
    canvas.paste(right, (w + 2 * margin, header + margin))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), title_left, fill="#ffffff")
    draw.text((w + 2 * margin, 20), title_right, fill="#ffffff")
    return canvas


def visualize(data_root, output_dir, *, indices=DEFAULT_RGB, tile=128, uafl_npy=None):
    root = Path(_find_augsburg_root(str(data_root)))
    synthetic_tif = root / "sub_area_1" / "EeteS_EnMAP_10m_sub_area1.tif"
    if not synthetic_tif.is_file():
        raise FileNotFoundError(
            f"Exact synthetic Augsburg test GT not found: {synthetic_tif}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # This matches data_loader.py's preprocessing & band ordering.
    synthetic_hsi = _read_tiff_cube(str(synthetic_tif), 242)
    height, width, bands = synthetic_hsi.shape
    if (height, width, bands) != (300, 360, 242):
        raise ValueError(
            f"Unexpected Augsburg sub_area_1 shape {synthetic_hsi.shape}, "
            "expected 300x360x242. Check the original MDAS directory."
        )
    coords = _nonoverlap_coords(height, width, tile)
    if tile == 128 and coords != [(0, 0), (0, 128), (128, 0), (128, 128)]:
        raise AssertionError("Unexpected synthetic test tiling")
    full_rgb = prepare_rgb(synthetic_hsi, indices)
    other_rgb = None
    if uafl_npy:
        predicted = np.load(uafl_npy, mmap_mode="r")
        other_rgb = prepare_rgb(predicted, indices)
        if other_rgb.shape[:2] != full_rgb.shape[:2]:
            raise ValueError(
                f"UAFL comparison must be HxW={full_rgb.shape[:2]}, got {other_rgb.shape[:2]}"
            )
    lo, hi = shared_stretch([full_rgb] + ([other_rgb] if other_rgb is not None else []))
    full = render_rgb(full_rgb, lo, hi)
    footprint = overlay_exact_tiles(full, coords, tile)
    crop_height = max(t + tile for t, _ in coords)
    crop_width = max(l + tile for _, l in coords)
    mosaic = full.crop((0, 0, crop_width, crop_height))

    paths = {
        "full": output / "Augsburg_synthetic_subarea1_full_RGB.png",
        "footprint": output / "Augsburg_synthetic_subarea1_TEST_footprint.png",
        "evaluated_mosaic": output / "Augsburg_synthetic_subarea1_TEST_mosaic_256x256.png",
    }
    full.save(paths["full"])
    footprint.save(paths["footprint"])
    mosaic.save(paths["evaluated_mosaic"])
    if other_rgb is not None:
        comparison = make_comparison(
            full, render_rgb(other_rgb, lo, hi),
            title_left="Synthetic test: sub_area_1 / original HSI10",
            title_right="UAFL full: sub_area_2 / reconstructed HSI10",
        )
        paths["comparison"] = output / "Augsburg_subarea1_vs_UAFL_subarea2_RGB.png"
        comparison.save(paths["comparison"])

    report = {
        "synthetic_test_source": str(synthetic_tif.resolve()),
        "synthetic_test_shape": [height, width, bands],
        "evaluation_patch": [tile, tile],
        "evaluation_tile_top_left_coordinates": [list(pair) for pair in coords],
        "evaluated_pixels": len(coords) * tile * tile,
        "full_pixels": height * width,
        "evaluated_fraction": len(coords) * tile * tile / (height * width),
        "test_bbox_y0_y1_x0_x1": [0, crop_height, 0, crop_width],
        "rgb_band_indices_zero_based": list(indices),
        "display_percentiles_shared": [1, 99],
        "display_only": True,
        "uafl_comparison_source": str(Path(uafl_npy).resolve()) if uafl_npy else None,
        "outputs": {k: str(v.resolve()) for k, v in paths.items()},
    }
    with (output / "Augsburg_synthetic_TEST_visualization_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Visualize true sub_area_1 synthetic Augsburg test region and four 128x128 test tiles"
    )
    parser.add_argument("--data_root", default="./data/raw")
    parser.add_argument(
        "--output_dir",
        default="./outputs/augsburg_synthetic_test_visualization",
    )
    parser.add_argument("--rgb_indices", default="43,28,10",
                        help="Three 0-based EnMAP band indices for the display only")
    parser.add_argument("--uafl_npy", default="",
                        help="Optional aligned sub_area_2 UAFL prediction for two-region comparison")
    args = parser.parse_args()
    indices = tuple(int(x) for x in args.rgb_indices.split(","))
    visualize(args.data_root, args.output_dir, indices=indices,
              uafl_npy=args.uafl_npy or None)


if __name__ == "__main__":
    main()
