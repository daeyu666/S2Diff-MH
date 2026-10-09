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


def resolve_existing_outputs(methods, *, wald_root):
    """Validate model-owned reconstructed HSI inputs; visualization NEVER infers.

    Each model's reconstruction should already have been generated with
    its own infer_augsburg2_wald.py --center_holdout. Read only those files.
    """
    roi = read_roi(wald_root)
    if roi is None:
        raise FileNotFoundError(
            f"Center-holdout roi.json missing: {Path(wald_root) / 'roi.json'}"
        )
    y0, x0, y1, x1 = map(int, roi["test_bbox_10m"])
    expected_shape = (y1 - y0, x1 - x0, 242)
    resolved = []
    missing = []
    for method_name, file_path in methods:
        name = method_name.strip()
        if not name:
            raise ValueError("--method requires a nonempty model name")
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            missing.append((name, path))
            continue
        loaded = np.load(path, mmap_mode="r")
        if loaded.shape != expected_shape:
            raise ValueError(
                f"{name} output has shape {loaded.shape}, expected held-out "
                f"{expected_shape}: {path}. Do not use old full-scene predictions."
            )
        resolved.append((name, str(path)))
    if missing:
        descriptions = [
            f"{name}: {path}" for name, path in missing
        ]
        commands = [
            "S2Diff-MH repository: python infer_augsburg2_wald.py --center_holdout --write_tif",
            "comparison_experiments repository: python comparison/UAFL/infer_augsburg2_wald.py --center_holdout --write_tif",
        ]
        raise FileNotFoundError(
            "Visualization reads existing reconstruction files only. "
            "The following 10m held-out HSI .npy file(s) are missing:\n"
            + "\n".join(descriptions)
            + "\nFirst train each model, then run its OWN inference command:\n"
            + "\n".join(commands)
            + "\nAfter both inference scripts report QNR/Dlambda/Ds and save .npy, rerun visualization."
        )
    return resolved


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
        cube = np.load(path, mmap_mode="r")
        if cube.shape != (yy1-yy0, xx1-xx0, 242):
            raise ValueError(
                f"{name}: expected exact center ROI {(yy1-yy0,xx1-xx0,242)} "
                f"but got {cube.shape} at {path}"
            )
        model_rgbs[name] = np.asarray(cube[:, :, list(rgb)], dtype=np.float32)
    if model_rgbs:
        vals = np.concatenate([x.reshape(-1, 3) for x in model_rgbs.values()])
        limits = (np.percentile(vals, 1, axis=0), np.percentile(vals, 99, axis=0))
    else:
        limits = None
    panels = [("Original S2 MSI (red box)", overview), ("Observed 10m MSI ROI", raw_roi)]
    model_rgb_files = {}
    for name, path in methods:
        im = stretch(model_rgbs[name], limits)
        # Save each model's RGB preview BESIDE its own inference .npy,
        # never in another model's outputs folder.
        model_input = Path(path)
        rgb_path = model_input.with_name(model_input.stem.replace("_HSI", "") + "_RGB.png")
        im.save(rgb_path)
        model_rgb_files[name] = str(rgb_path.resolve())
        panels.append((name + " (held-out reconstruction)", im))
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
            "model_rgb_files_in_own_result_folders": model_rgb_files,
            "inference_triggered_by_visualization": False,
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
    p = argparse.ArgumentParser(
        description="Visualize existing Augsburg center-heldout reconstructions; NEVER trains or infers"
    )
    p.add_argument("--wald_root", default="./data/augsburg2_wald_center_holdout")
    p.add_argument("--output_dir", default="./figures/augsburg2_center_holdout")
    p.add_argument("--method", nargs=2, action="append", default=[],
                   metavar=("NAME", "FUSED_NPY"),
                   help="Existing 144x144x242 reconstruction .npy; repeat for multiple models")
    p.add_argument("--hsi_rgb", default="43,28,10")
    p.add_argument("--savefig", default="./figures/Augsburg_holdout_S2Diff_vs_UAFL_RGB.png")
    args = p.parse_args()
    bands = tuple(int(x) for x in args.hsi_rgb.split(","))
    if len(bands) != 3 or any(z < 0 or z >= 242 for z in bands):
        raise ValueError("3 zero-based HSI RGB band indices required")
    methods = args.method or [
        ("S2Diff", "./outputs/augsburg2_wald_center_holdout/Augsburg2_Wald_heldout_HSI.npy"),
        ("UAFL", "../comparison_experiments/comparison/UAFL/outputs/augsburg2_wald_center_holdout/Augsburg2_Wald_UAFL_heldout_HSI.npy"),
    ]
    # Fail early before opening overview imagery or creating output folders.
    methods = resolve_existing_outputs(methods, wald_root=args.wald_root)
    visualize(args.wald_root, args.output_dir, methods, rgb=bands,
              savefig=args.savefig)


if __name__ == "__main__":
    main()
