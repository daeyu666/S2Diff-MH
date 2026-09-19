"""Cross-dataset validation of MSI-observable local spectral heterogeneity.

This diagnostic is intentionally model-free.  It tests only the premise needed
by Innovation 3:

    local spectral heterogeneity observed from HR-MSI
        ~ local spectral heterogeneity in GT HR-HSI.

No diffusion checkpoint, prediction, SAM, or training is involved.

For every selected HSI patch, the script synthesizes HR-MSI with the repository's
fixed sensor protocol, computes the same 3x3 local spectral-heterogeneity map on
unit-normalized spectra for HSI and MSI, converts both maps to within-patch ranks,
and reports:

  * Spearman rho(H_M, H_GT)
  * top-q recall / Jaccard of MSI-high heterogeneity vs GT-high heterogeneity
  * low/high-quartile separation of GT heterogeneity induced by MSI ranking

Multiple deterministic spatial patches are used so the conclusion is not tied
to the single center test crop used by the normal loader.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from data_loader import (
    DATASET_SPECS,
    crop_to_scale,
    normalize_hsi,
    read_hsi_mat,
)
from diagnose_observable_heterogeneity import _rank01_map, observable_heterogeneity
from diagnose_spectral_shape import _spearman
from srf_utils import (
    build_srf_weights,
    hsi_to_msi_numpy,
    load_hsi_wavelengths,
    sensor_protocol,
)
from utils import ensure_dir, get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-dataset MSI-vs-GT local spectral heterogeneity diagnostic"
    )
    p.add_argument(
        "--datasets",
        default="Chikusei,Houston13",
        help="comma-separated datasets; e.g. Chikusei,Houston13",
    )
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument(
        "--patch_grid",
        type=int,
        default=3,
        help="deterministic NxN spatial grid of patches over each scene",
    )
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument(
        "--top_fraction",
        type=float,
        default=0.25,
        help="fraction used for high-heterogeneity overlap",
    )
    p.add_argument(
        "--output_json",
        default="./results/msi_gt_heterogeneity_transfer.json",
    )
    return p.parse_args()


def _parse_datasets(text: str) -> Tuple[str, ...]:
    values = tuple(x.strip() for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("--datasets is empty")
    for name in values:
        if name not in DATASET_SPECS:
            raise ValueError(f"unsupported dataset={name!r}")
    return values


def _grid_starts(length: int, patch: int, grid: int) -> List[int]:
    if patch > length:
        raise ValueError(f"patch_size={patch} exceeds scene dimension={length}")
    if grid <= 1:
        return [(length - patch) // 2]
    max_start = length - patch
    starts = np.linspace(0, max_start, grid)
    starts = np.rint(starts).astype(np.int64).tolist()
    return sorted(set(int(v) for v in starts))


def _patch_coords(h: int, w: int, patch: int, grid: int) -> List[Tuple[int, int]]:
    ys = _grid_starts(h, patch, grid)
    xs = _grid_starts(w, patch, grid)
    return [(y, x) for y in ys for x in xs]


def _finite_pair(a: torch.Tensor, b: torch.Tensor):
    mask = torch.isfinite(a) & torch.isfinite(b)
    return a[mask], b[mask]


def _top_overlap(
    pred_rank: torch.Tensor,
    gt_rank: torch.Tensor,
    fraction: float,
) -> Dict[str, float]:
    p, g = _finite_pair(pred_rank, gt_rank)
    qp = torch.quantile(p, 1.0 - fraction)
    qg = torch.quantile(g, 1.0 - fraction)
    hp = p >= qp
    hg = g >= qg
    inter = (hp & hg).sum().float()
    recall = inter / hg.sum().clamp_min(1)
    precision = inter / hp.sum().clamp_min(1)
    union = (hp | hg).sum().float()
    jaccard = inter / union.clamp_min(1)
    return {
        "recall": float(recall.item()),
        "precision": float(precision.item()),
        "jaccard": float(jaccard.item()),
    }


def _quantile_separation(
    msi_rank: torch.Tensor,
    gt_rank: torch.Tensor,
    fraction: float,
) -> Dict[str, float]:
    m, g = _finite_pair(msi_rank, gt_rank)
    lo = m <= torch.quantile(m, fraction)
    hi = m >= torch.quantile(m, 1.0 - fraction)
    low_mean = float(g[lo].mean().item())
    high_mean = float(g[hi].mean().item())
    return {
        "gt_rank_high_msi": high_mean,
        "gt_rank_low_msi": low_mean,
        "delta_gt_rank": high_mean - low_mean,
        "ratio_gt_rank": high_mean / max(low_mean, 1e-12),
    }


def analyze_patch(
    gt_hsi_np: np.ndarray,
    srf_weights: np.ndarray,
    *,
    device,
    top_fraction: float,
) -> Dict[str, float]:
    hr_msi_np = hsi_to_msi_numpy(gt_hsi_np, srf_weights)

    gt = (
        torch.from_numpy(gt_hsi_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .float()
        .to(device)
    )
    msi = (
        torch.from_numpy(hr_msi_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .float()
        .to(device)
    )

    h_gt = observable_heterogeneity(gt)
    h_msi = observable_heterogeneity(msi)
    r_gt = _rank01_map(h_gt).reshape(-1)
    r_msi = _rank01_map(h_msi).reshape(-1)

    overlap = _top_overlap(r_msi, r_gt, top_fraction)
    separation = _quantile_separation(r_msi, r_gt, top_fraction)
    return {
        "rho_rank": _spearman(r_msi, r_gt),
        "top_recall": overlap["recall"],
        "top_precision": overlap["precision"],
        "top_jaccard": overlap["jaccard"],
        **separation,
    }


def _summarize(values: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = values[0].keys()
    result = {}
    for key in keys:
        arr = np.asarray([v[key] for v in values], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return result


def _load_dataset_and_srf(
    dataset: str,
    data_root: str,
    scale_ratio: int,
    srf_interp: str,
):
    spec = DATASET_SPECS[dataset]
    path = os.path.join(data_root, spec["file"])
    image = normalize_hsi(read_hsi_mat(path, spec["keys"]))
    image = crop_to_scale(image, scale_ratio)

    protocol = sensor_protocol(dataset)
    wavelengths = load_hsi_wavelengths(protocol["wavelength_path"], image.shape[2])
    weights, band_names = build_srf_weights(
        protocol["srf_path"],
        wavelengths,
        protocol["bands"],
        interp_kind=srf_interp,
    )
    return image, weights, band_names, protocol


def main():
    args = parse_args()
    if not 0.0 < args.top_fraction < 0.5:
        raise ValueError("--top_fraction must lie in (0,0.5)")
    if args.patch_grid < 1:
        raise ValueError("--patch_grid must be >=1")

    datasets = _parse_datasets(args.datasets)
    set_seed(args.seed)
    device = get_device(args.device)

    payload = {
        "conditions": {
            "datasets": list(datasets),
            "patch_size": args.patch_size,
            "patch_grid": args.patch_grid,
            "scale_ratio": args.scale_ratio,
            "srf_interp": args.srf_interp,
            "top_fraction": args.top_fraction,
            "seed": args.seed,
            "model_free": True,
            "heterogeneity": "3x3 local variance magnitude of unit-normalized spectra",
            "comparison": "within-patch rank maps",
        },
        "datasets": {},
    }

    for dataset in datasets:
        image, weights, band_names, protocol = _load_dataset_and_srf(
            dataset,
            args.data_root,
            args.scale_ratio,
            args.srf_interp,
        )
        h, w, c = image.shape
        coords = _patch_coords(h, w, args.patch_size, args.patch_grid)
        rows = []

        print("=" * 120)
        print(
            f"DATASET={dataset} scene={h}x{w}x{c} MSI_bands={len(band_names)} "
            f"sensor_bands={','.join(band_names)} patches={len(coords)}"
        )
        print(
            f"SRF={protocol['srf_path']} wavelengths={protocol['wavelength_path']} "
            f"interp={args.srf_interp}"
        )

        for idx, (top, left) in enumerate(coords):
            patch = image[
                top : top + args.patch_size,
                left : left + args.patch_size,
            ].copy()
            row = analyze_patch(
                patch,
                weights,
                device=device,
                top_fraction=args.top_fraction,
            )
            row.update({"index": idx, "top": top, "left": left})
            rows.append(row)
            print(
                f"{dataset} PATCH={idx:02d} xy=({top},{left}) "
                f"rho={row['rho_rank']:+.6f} "
                f"recall={row['top_recall']:.6f} "
                f"jaccard={row['top_jaccard']:.6f} "
                f"deltaGT={row['delta_gt_rank']:+.6f}"
            )

        numeric_rows = [
            {k: v for k, v in row.items() if k not in ("index", "top", "left")}
            for row in rows
        ]
        summary = _summarize(numeric_rows)
        print(
            f"{dataset} SUMMARY "
            f"rho={summary['rho_rank']['mean']:.6f}±{summary['rho_rank']['std']:.6f} "
            f"rho_min={summary['rho_rank']['min']:.6f} "
            f"recall={summary['top_recall']['mean']:.6f}±{summary['top_recall']['std']:.6f} "
            f"jaccard={summary['top_jaccard']['mean']:.6f}±{summary['top_jaccard']['std']:.6f} "
            f"deltaGT={summary['delta_gt_rank']['mean']:+.6f}"
        )

        payload["datasets"][dataset] = {
            "scene_shape": [h, w, c],
            "msi_band_names": list(band_names),
            "srf_path": protocol["srf_path"],
            "wavelength_path": protocol["wavelength_path"],
            "patches": rows,
            "summary": summary,
        }

    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print("=" * 120)
    print(f"SAVED_JSON {args.output_json}")
    print(
        "INTERPRETATION: strong transfer support requires high rho plus high top-region recall/Jaccard "
        "across multiple spatial patches, not only a single center crop.  This diagnostic validates only "
        "whether HR-MSI can locate GT-HSI spectral-heterogeneity regions; it does not by itself prove that "
        "heterogeneity causes reconstruction SAM."
    )


if __name__ == "__main__":
    main()
