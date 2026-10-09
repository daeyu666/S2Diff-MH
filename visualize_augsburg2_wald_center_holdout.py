"""Generate Fig.13-style red-box overview and held-out reconstructed ROI panels.

Unlike the old full-region displays, this visualizer reads the central
untrained test ROI defined by roi.json. The left overview uses ORIGINAL
real Sentinel-2 full-scene MSI; comparison panels use saved 10m HSI
predictions on that exact ROI (no reconstruction or re-training).
Colors are for visualization only; all QNR computations use raw spectra.

Example:
python visualize_augsburg2_wald_center_holdout.py \
  --wald_root ./data/augsburg2_wald_center_holdout \
  --method UAFL ../comparison_experiments/comparison/UAFL/outputs/augsburg2_wald_center_holdout/Augsburg2_Wald_UAFL_heldout_HSI.npy \
  --method S2Diff ./outputs/augsburg2_wald_center_holdout/Augsburg2_Wald_heldout_HSI.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from augsburg2_wald_center_roi import read_roi


def stretch(x, bounds=None):
    if bounds is None:
        lo = np.percentile(x.reshape(-1, 3), 1, axis=0)
        hi = np.percentile(x.reshape(-1, 3), 99, axis=0)
    else:
        lo, hi = bounds
    z = (x - lo[None, None]) / np.maximum(hi - lo, 1e-6)[None, None]
    return Image.fromarray(np.asarray(np.rint(np.clip(z, 0, 1) * 255), dtype=np.uint8), "RGB")


def visualize(wald_root, output_dir, methods, rgb=(43,28,10), savefig=None):
    root = Path(wald_root)
    roi = read_roi(root)
    if roi is None:
        raise ValueError("Center-holdout roi.json required for Fig.13-style visualization")
    if roi.get("source_region") != "sub_area_2":
        raise ValueError("Expected Augsburg Region-2 source")
    yy0, xx0, yy1, xx1 = map(int, roi["test_bbox_10m"])
    full_msi = np.load(root/"full"/"hr_msi.npy", mmap_mode="r")
    if full_msi.shape[-1] != 4:
        raise ValueError("Expected real S2 four-band MSI")
    full_rgb = np.asarray(full_msi[..., [2, 1, 0]], dtype=np.float32)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Display only; do not use calibration fitted on the held-out area.
    overview = stretch(full_rgb)
    pen = ImageDraw.Draw(overview)
    pen.rectangle((xx0, yy0, xx1-1, yy1-1), outline=(255,0,0), width=3)
    overview.save(out/"Augsburg2_original_MSI_redbox.png")
    raw_roi = stretch(full_rgb[yy0:yy1, xx0:xx1])
    raw_roi.save(out/"Augsburg2_original_MSI_heldout_ROI.png")

    model_rgbs = {}
    for name, path in methods:
        if not name.strip():
            raise ValueError("Empty model name")
        cube = np.load(path, mmap_mode="r")
        if cube.shape != (yy1-yy0, xx1-xx0, 242):
            raise ValueError(
                f"{name}: expected exact center ROI {(yy1-yy0,xx1-xx0,242)} "
                f"but got {cube.shape}. Old full-region predictions are not heldout models."
            )
        model_rgbs[name] = np.asarray(cube[:,:,list(rgb)], dtype=np.float32)
    if model_rgbs:
        vals = np.concatenate([x.reshape(-1,3) for x in model_rgbs.values()])
        limits = (np.percentile(vals,1,axis=0), np.percentile(vals,99,axis=0))
    else:
        limits = None
    panels = [("Original S2 MSI (red box)", overview), ("Observed 10m MSI ROI", raw_roi)]
    for name, x in model_rgbs.items():
        im = stretch(x, limits)
        im.save(out/(name.replace(" ","_")+"_heldout_ROI_RGB.png"))
        panels.append((name+" (held-out reconstruction)", im))
    # Paper-style comparison: left overview, then ROI panels all at same scale.
    # Resize only the displayed preview; do not resample source arrays/metrics.
    tile_w, tile_h, top = 290, 290, 37
    canvas = Image.new("RGB", (tile_w*len(panels), tile_h+top), "#10151e")
    draw = ImageDraw.Draw(canvas)
    for i, (label, panel) in enumerate(panels):
        img = panel.copy()
        img.thumbnail((tile_w-10, tile_h-10), Image.Resampling.LANCZOS)
        canvas.paste(img, (i*tile_w+5+(tile_w-10-img.width)//2,
                           top+(tile_h-10-img.height)//2))
        draw.text((i*tile_w+8, 11), label, fill="white")
    fig = out/"Augsburg2_center_heldout_Fig13_style.png"
    canvas.save(fig)
    requested_fig = None
    if savefig:
        requested_fig = Path(savefig).expanduser()
        if requested_fig.suffix.lower() != ".png":
            raise ValueError("--savefig requires a .png filename")
        requested_fig.parent.mkdir(parents=True, exist_ok=True)
        if requested_fig.resolve() != fig.resolve():
            canvas.save(requested_fig)
    with (out/"visualization_provenance.json").open("w",encoding="utf-8") as f:
        json.dump({
            "protocol_id": roi["protocol_id"],
            "source_region": roi["source_region"],
            "raw_source": str(root/"full"/"hr_msi.npy"),
            "bbox_10m": roi["test_bbox_10m"],
            "bbox_30m": roi["test_bbox_30m"],
            "rgb_indices_0based_HSI": list(rgb),
            "comparison_rgb_per_method_shared_percentiles": [1,99],
            "model_paths": {name:str(Path(path).resolve()) for name,path in methods},
            "metrics_unchanged_by_visualization": True,
            "figure": str((requested_fig or fig).resolve()),
            "default_figure": str(fig.resolve()),
            "requested_savefig": str(requested_fig.resolve()) if requested_fig else None,
        },f,indent=2)
    print(
        f"FIG13_STYLE={(requested_fig or fig).resolve()} "
        f"REDBOX_BBOX_10M={roi['test_bbox_10m']}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wald_root", default="./data/augsburg2_wald_center_holdout")
    p.add_argument("--output_dir", default="./outputs/augsburg2_wald_center_holdout/fig13")
    p.add_argument("--method", nargs=2, action="append", default=[], metavar=("NAME","FUSED_NPY"))
    p.add_argument("--hsi_rgb", default="43,28,10")
    p.add_argument("--savefig", default=None,
                   help="Optional exact path for the final combined PNG figure")
    args = p.parse_args()
    bands = tuple(int(x) for x in args.hsi_rgb.split(","))
    if len(bands) != 3 or any(z<0 or z>=242 for z in bands):
        raise ValueError("3 zero-based HSI RGB band indices required")
    visualize(args.wald_root, args.output_dir, args.method, rgb=bands,
              savefig=args.savefig)


if __name__=="__main__":
    main()
