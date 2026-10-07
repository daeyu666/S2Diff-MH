"""Calibrate the MDAS effective 10->30 m spatial transfer on TRAIN only."""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean

import numpy as np
import torch

from degradations.effective_gaussian import EffectiveGaussianDegradation
from utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate Augsburg-Real effective x3 Gaussian PSF")
    p.add_argument("--cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--output", default="./data/calibration/AugsburgReal_effective_psf.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--patch_size", type=int, default=192)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--band_stride", type=int, default=4)
    p.add_argument("--sigma_min", type=float, default=0.30)
    p.add_argument("--sigma_max", type=float, default=2.50)
    p.add_argument("--coarse_steps", type=int, default=45)
    p.add_argument("--fine_steps", type=int, default=31)
    p.add_argument("--truncate", type=float, default=3.0)
    return p.parse_args()


def _valid_lr_mask(mask_hr: torch.Tensor) -> torch.Tensor:
    pooled = torch.nn.functional.avg_pool2d(mask_hr, 3, 3)
    return pooled >= 0.999


def _sample_train_patches(gt, lr, valid, *, patch: int, samples: int, seed: int):
    if patch % 3:
        raise ValueError("patch_size must be divisible by 3")
    rng = np.random.default_rng(seed)
    h, w = gt.shape[:2]
    candidates = []
    stride = max(3, patch // 2)
    stride = stride // 3 * 3
    for top in range(0, h - patch + 1, stride):
        for left in range(0, w - patch + 1, stride):
            if float(valid[top:top+patch, left:left+patch].mean()) >= 0.95:
                candidates.append((top, left))
    if not candidates:
        raise RuntimeError("No valid train patches available for PSF calibration")
    rng.shuffle(candidates)
    for top, left in candidates[: min(samples, len(candidates))]:
        lp = patch // 3
        yield (
            np.asarray(gt[top:top+patch, left:left+patch]),
            np.asarray(lr[top//3:top//3+lp, left//3:left//3+lp]),
            np.asarray(valid[top:top+patch, left:left+patch]),
        )


def _score_sigma(sigma, rows, *, device, band_stride: int, truncate: float):
    op = EffectiveGaussianDegradation(
        scale_ratio=3,
        terminal_sigma=float(sigma),
        truncate=truncate,
    ).to(device)
    l1s, rmses = [], []
    with torch.no_grad():
        for gt_np, lr_np, mask_np in rows:
            bands = np.arange(0, gt_np.shape[2], max(1, int(band_stride)))
            gt = torch.from_numpy(gt_np[..., bands]).permute(2, 0, 1).unsqueeze(0).float().to(device)
            lr = torch.from_numpy(lr_np[..., bands]).permute(2, 0, 1).unsqueeze(0).float().to(device)
            mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            pred = op.degrade(gt)
            valid_lr = _valid_lr_mask(mask).expand(-1, pred.shape[1], -1, -1)
            selected = (pred - lr)[valid_lr]
            if selected.numel() == 0:
                continue
            l1s.append(float(selected.abs().mean().item()))
            rmses.append(float(torch.sqrt((selected * selected).mean()).item()))
    if not l1s:
        return float("inf"), float("inf")
    return mean(l1s), mean(rmses)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    train_dir = os.path.join(args.cache_root, "train")
    gt = np.load(os.path.join(train_dir, "gt.npy"), mmap_mode="r")
    lr = np.load(os.path.join(train_dir, "lr_hsi.npy"), mmap_mode="r")
    valid = np.load(os.path.join(train_dir, "valid_mask.npy"), mmap_mode="r")
    rows = list(
        _sample_train_patches(
            gt,
            lr,
            valid,
            patch=args.patch_size,
            samples=args.samples,
            seed=args.seed,
        )
    )

    coarse = np.linspace(args.sigma_min, args.sigma_max, args.coarse_steps)
    records = []
    for sigma in coarse:
        l1, rmse = _score_sigma(
            float(sigma),
            rows,
            device=device,
            band_stride=args.band_stride,
            truncate=args.truncate,
        )
        records.append((float(sigma), l1, rmse))
        print(f"COARSE sigma={sigma:.6f} L1={l1:.8f} RMSE={rmse:.8f}")
    best = min(records, key=lambda x: x[1])

    coarse_step = float(coarse[1] - coarse[0]) if len(coarse) > 1 else 0.1
    fine_lo = max(args.sigma_min, best[0] - coarse_step)
    fine_hi = min(args.sigma_max, best[0] + coarse_step)
    fine = np.linspace(fine_lo, fine_hi, args.fine_steps)
    fine_records = []
    for sigma in fine:
        l1, rmse = _score_sigma(
            float(sigma),
            rows,
            device=device,
            band_stride=args.band_stride,
            truncate=args.truncate,
        )
        fine_records.append((float(sigma), l1, rmse))
        print(f"FINE sigma={sigma:.6f} L1={l1:.8f} RMSE={rmse:.8f}")
    best = min(fine_records, key=lambda x: x[1])

    payload = {
        "dataset": "Augsburg-Real",
        "operator": "effective_inter_resolution_gaussian",
        "scale_ratio": 3,
        "stages": [1, 2, 3],
        "terminal_sigma_hr_pixels": best[0],
        "train_l1": best[1],
        "train_rmse": best[2],
        "truncate": args.truncate,
        "band_stride": args.band_stride,
        "calibration_split": "train_only",
        "interpretation": (
            "effective 10-to-30 m transfer response; not native EnMAP instrument PSF"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("BEST_EFFECTIVE_PSF " + json.dumps(payload))


if __name__ == "__main__":
    main()
