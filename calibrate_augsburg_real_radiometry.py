"""Train-only bandwise radiometric harmonisation for Augsburg-Real.

Fits y ~= gain*x + bias in the common 30 m S2-equivalent observation domain,
where x is real Sentinel-2 degraded to 30 m and y is EnMAP30 projected by the
fixed Sentinel-2 SRF. No validation/test pixels enter the fit.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from augsburg_real import build_augsburg_real_loaders
from cdrdi_geometry import spectral_project
from degradations.effective_gaussian import EffectiveGaussianDegradation
from utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate Augsburg-Real radiometry on train split")
    p.add_argument("--cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--psf_json", default="./data/calibration/AugsburgReal_effective_psf.json")
    p.add_argument("--output", default="./data/calibration/AugsburgReal_radiometry.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--patch_size", type=int, default=192)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--min_valid_fraction", type=float, default=0.90)
    return p.parse_args()


def _load_sigma(path: str) -> float:
    with open(path, "r", encoding="utf-8") as handle:
        return float(json.load(handle)["terminal_sigma_hr_pixels"])


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    sigma = _load_sigma(args.psf_json)
    train_loader, _, _, info = build_augsburg_real_loaders(
        args.cache_root,
        train_patch_size=args.patch_size,
        train_stride=args.stride,
        eval_patch_size=args.patch_size,
        min_valid_fraction=args.min_valid_fraction,
        batch_size=1,
        num_workers=0,
    )
    p0 = EffectiveGaussianDegradation(scale_ratio=3, terminal_sigma=sigma).to(device)
    srf = torch.as_tensor(info["srf_weights"], dtype=torch.float32, device=device)

    n = torch.zeros(4, dtype=torch.float64, device=device)
    sx = torch.zeros_like(n)
    sy = torch.zeros_like(n)
    sxx = torch.zeros_like(n)
    syy = torch.zeros_like(n)
    sxy = torch.zeros_like(n)

    with torch.no_grad():
        for batch in train_loader:
            lr = batch["lr_hsi"].to(device)
            msi = batch["hr_msi"].to(device)
            mask = batch["valid_mask"].to(device)
            target = spectral_project(lr, srf)
            pred = p0.degrade(msi)
            valid = F.avg_pool2d(mask.float(), 3, 3) >= 0.999
            valid = valid.expand(-1, 4, -1, -1)
            for band in range(4):
                keep = valid[:, band]
                x = pred[:, band][keep].double()
                y = target[:, band][keep].double()
                if x.numel() == 0:
                    continue
                n[band] += x.numel()
                sx[band] += x.sum()
                sy[band] += y.sum()
                sxx[band] += (x * x).sum()
                syy[band] += (y * y).sum()
                sxy[band] += (x * y).sum()

    eps = 1e-12
    safe_n = n.clamp_min(1)
    mx = sx / safe_n
    my = sy / safe_n
    varx = (sxx / safe_n - mx * mx).clamp_min(eps)
    vary = (syy / safe_n - my * my).clamp_min(eps)
    cov = sxy / safe_n - mx * my
    gain = cov / varx
    bias = my - gain * mx
    corr = cov / torch.sqrt(varx * vary)

    payload = {
        "dataset": "Augsburg-Real",
        "split": "train_only",
        "domain": "30m_S2_equivalent",
        "bands": ["B2", "B3", "B4", "B8"],
        "effective_sigma_hr_pixels": sigma,
        "gain": [float(v) for v in gain.cpu()],
        "bias": [float(v) for v in bias.cpu()],
        "correlation_before_geometry": [float(v) for v in corr.cpu()],
        "samples": [int(v) for v in n.cpu()],
        "model": "target_R_EnMAP30 ~= gain * D3(real_S2) + bias",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("AUGSBURG_REAL_RADIOMETRY " + json.dumps(payload))


if __name__ == "__main__":
    main()
