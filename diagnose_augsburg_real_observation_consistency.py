"""Measure reference/observation inconsistency on Augsburg-Real VALIDATION.

No parameters are fitted, no test imagery is used, and no models are changed.

This checks whether the reference HR EnMAP-like cube itself satisfies:
  D_eff(EnMAP10) ~= observed EnMAP30
  R_S2(EnMAP10) ~= calibrated real Sentinel-2 MSI

Optionally evaluate an identity-geometry Real-D2 checkpoint under the same
metrics. Keep the benchmark's full-region pixel-weighted evaluation mask.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from augsburg_real import (
    _reproject_multiband,
    _target_profile,
    build_augsburg_real_loaders,
)
from augsburg_real_process import build_augsburg_real_process
from cdrdi_geometry import spectral_project
from innovation1 import reconstruct_from_terminal_lr
from main import build_model
from train_augsburg_real_diffusion import (
    _apply_radiometry,
    _config,
    _load_json,
    _lr_mask,
    _masked_metric_sums,
    _metrics_from_sums,
    _radiometry,
)
from utils import get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Check Augsburg-Real GT physical and spectral observation consistency"
    )
    p.add_argument("--cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--psf_json", default="./data/calibration/AugsburgReal_effective_psf.json")
    p.add_argument("--radiometry_json", default="./data/calibration/AugsburgReal_radiometry.json")
    p.add_argument("--diffusion_checkpoint", default="")
    p.add_argument("--compare_simulated_msi", action="store_true",
                   help="Compare official same-source EeteS simulated 10m MSI to real S2; validation only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--eval_patch_size", type=int, default=192)
    p.add_argument("--min_valid_fraction", type=float, default=0.8)
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    return p.parse_args()


class ChannelStats:
    def __init__(self, channels):
        self.n = torch.zeros(channels, dtype=torch.float64)
        self.abs_sum = torch.zeros(channels, dtype=torch.float64)
        self.sq_sum = torch.zeros(channels, dtype=torch.float64)

    def update(self, pred, target, valid):
        diff = (pred - target).double()
        mask = valid[:, :1].expand_as(diff)
        # Mask by multiplication, avoiding indexed flattening of all spectral bands.
        self.n += mask.sum(dim=(0, 2, 3)).double().cpu()
        self.abs_sum += (diff.abs() * mask).sum(dim=(0, 2, 3)).cpu()
        self.sq_sum += (diff.square() * mask).sum(dim=(0, 2, 3)).cpu()

    def l1(self):
        return self.abs_sum / self.n.clamp_min(1)

    def report(self, prefix, names=None):
        l1 = self.l1()
        rmse = torch.sqrt(self.sq_sum / self.n.clamp_min(1))
        print(f"{prefix} L1={self.abs_sum.sum().item()/max(self.n.sum().item(),1):.8f} "
              f"RMSE={math.sqrt(self.sq_sum.sum().item()/max(self.n.sum().item(),1)):.8f} "
              f"PIXEL_SPECTRAL_VALUES={int(self.n.sum().item())}")
        if names is None:
            print(f"{prefix}_BANDS L1_MEDIAN={float(l1.median()):.8f} "
                  f"L1_P90={float(torch.quantile(l1,0.9)):.8f} "
                  f"L1_MAX={float(l1.max()):.8f}")
        else:
            for i, name in enumerate(names):
                print(f"{prefix}_{name} L1={float(l1[i]):.8f} "
                      f"RMSE={float(rmse[i]):.8f}")


class CorrStats:
    def __init__(self, channels):
        self.n = torch.zeros(channels, dtype=torch.float64)
        self.sx = torch.zeros(channels, dtype=torch.float64)
        self.sy = torch.zeros(channels, dtype=torch.float64)
        self.sxx = torch.zeros(channels, dtype=torch.float64)
        self.syy = torch.zeros(channels, dtype=torch.float64)
        self.sxy = torch.zeros(channels, dtype=torch.float64)

    def update(self, x, y, valid):
        v = valid[:, :1].double().expand_as(x)
        x = x.double()
        y = y.double()
        def sum_ch(t):
            return t.sum(dim=(0, 2, 3)).cpu()
        self.n += sum_ch(v)
        self.sx += sum_ch(v * x)
        self.sy += sum_ch(v * y)
        self.sxx += sum_ch(v * x.square())
        self.syy += sum_ch(v * y.square())
        self.sxy += sum_ch(v * x * y)

    def values(self):
        n = self.n.clamp_min(1)
        mx, my = self.sx / n, self.sy / n
        cov = self.sxy / n - mx * my
        vx = (self.sxx / n - mx.square()).clamp_min(1e-12)
        vy = (self.syy / n - my.square()).clamp_min(1e-12)
        return cov / torch.sqrt(vx * vy)


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    sigma = float(_load_json(args.psf_json)["terminal_sigma_hr_pixels"])
    radiometry = _radiometry(args.radiometry_json, device)
    _, val_loader, _, info = build_augsburg_real_loaders(
        args.cache_root,
        train_patch_size=96,
        train_stride=48,
        eval_patch_size=args.eval_patch_size,
        min_valid_fraction=args.min_valid_fraction,
        batch_size=1,
        num_workers=0,
    )
    srf = torch.as_tensor(info["srf_weights"], dtype=torch.float32, device=device)
    process = build_augsburg_real_process(
        effective_sigma=sigma, diffusion_steps=args.diffusion_steps
    )
    operator = process.operator.to(device)

    model = None
    if args.diffusion_checkpoint:
        model = build_model(_config(args), info, device)
        load_checkpoint(
            model, args.diffusion_checkpoint,
            map_location=str(device), load_optimizer=False,
        )
        model.eval()

    gt_phy = ChannelStats(info["n_bands"])
    gt_msi = ChannelStats(4)
    gt_corr = CorrStats(4)
    d2_phy = ChannelStats(info["n_bands"]) if model is not None else None
    d2_msi = ChannelStats(4) if model is not None else None
    d2_sse = 0.
    d2_n = 0
    d2_sam = 0.
    d2_sam_n = 0
    tiles = 0

    sim_msi = None
    sim_vs_gt = ChannelStats(4)
    sim_corr = CorrStats(4)
    real_vs_sim = ChannelStats(4)
    real_sim_corr = CorrStats(4)
    if args.compare_simulated_msi:
        metadata_path = os.path.join(args.cache_root, "validation", "meta.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        gt_path = metadata["gt_source"]
        sim_path = gt_path.replace("EnMAP_10m", "Sentinel_2_10m")
        if sim_path == gt_path or not os.path.isfile(sim_path):
            raise FileNotFoundError(
                "Official simulated MSI validation product not found: " + sim_path
            )
        import rasterio
        with rasterio.open(sim_path) as src:
            n_sim_bands = int(src.count)
        if n_sim_bands == 4:
            sim_indexes = [1, 2, 3, 4]
        elif n_sim_bands == 12:
            sim_indexes = [2, 3, 4, 8]
        else:
            raise ValueError(
                f"Expected official simulated MSI with 4 or 12 bands, got {n_sim_bands}"
            )
        crs, transform, width, height = _target_profile(gt_path)
        sim_msi = _reproject_multiband(
            sim_path,
            target_crs=crs,
            target_transform=transform,
            target_width=width,
            target_height=height,
            indexes=sim_indexes,
            scale=10000.,
            resampling="bilinear",
        )
        if tuple(sim_msi.shape[:2]) != tuple(val_loader.dataset.gt.shape[:2]):
            raise RuntimeError("Simulated MSI comparison grid differs from validation cache")
        print(f"SIMULATED_MSI_SOURCE={sim_path} source_bands={n_sim_bands} selected={sim_indexes}")
    for sample_index, batch in enumerate(val_loader):
        gt = batch["gt"].to(device)
        y_h = batch["lr_hsi"].to(device)
        y_m = _apply_radiometry(batch["hr_msi"].to(device), radiometry)
        hr_valid = batch["valid_mask"].to(device) > 0.5
        lr_valid = _lr_mask(hr_valid)

        gt_phy.update(operator.degrade(gt), y_h, lr_valid)
        gt_proj = spectral_project(gt, srf)
        gt_msi.update(gt_proj, y_m, hr_valid)
        gt_corr.update(gt_proj, y_m, hr_valid)

        if sim_msi is not None:
            top, left, ph, pw = val_loader.dataset.samples[sample_index]
            sim_patch = np.asarray(sim_msi[top:top+ph, left:left+pw, :])
            pad_h = gt.shape[-2] - ph
            pad_w = gt.shape[-1] - pw
            if pad_h or pad_w:
                sim_patch = np.pad(
                    sim_patch,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="edge",
                )
            sim_t = torch.from_numpy(
                np.ascontiguousarray(sim_patch)
            ).permute(2, 0, 1).unsqueeze(0).to(device).float()
            sim_valid = hr_valid & torch.isfinite(sim_t).all(dim=1, keepdim=True)
            sim_vs_gt.update(gt_proj, sim_t, sim_valid)
            sim_corr.update(gt_proj, sim_t, sim_valid)
            real_vs_sim.update(y_m, sim_t, sim_valid)
            real_sim_corr.update(y_m, sim_t, sim_valid)

        if model is not None:
            pred = reconstruct_from_terminal_lr(
                model, process, y_h,
                target_size=tuple(gt.shape[-2:]), hr_msi=y_m,
            )
            d2_phy.update(operator.degrade(pred), y_h, lr_valid)
            d2_msi.update(spectral_project(pred, srf), y_m, hr_valid)
            sse, n, sa, ns = _masked_metric_sums(pred, gt, hr_valid)
            d2_sse += sse
            d2_n += n
            d2_sam += sa
            d2_sam_n += ns
        tiles += 1

    print(f"OBSERVATION_CONSISTENCY split=validation tiles={tiles} sigma={sigma:.6f}")
    gt_phy.report("GT_PHY")
    gt_msi.report("GT_MSI", ["B2", "B3", "B4", "B8"])
    print("GT_S2_CORR " + " ".join(
        f"{name}={float(corr):.6f}"
        for name, corr in zip(["B2", "B3", "B4", "B8"], gt_corr.values())
    ))

    if sim_msi is not None:
        sim_vs_gt.report("SIM_GT_MSI", ["B2", "B3", "B4", "B8"])
        print("SIM_GT_S2_CORR " + " ".join(
            f"{name}={float(corr):.6f}"
            for name, corr in zip(["B2", "B3", "B4", "B8"], sim_corr.values())
        ))
        real_vs_sim.report("REAL_SIM_MSI", ["B2", "B3", "B4", "B8"])
        print("REAL_SIM_S2_CORR " + " ".join(
            f"{name}={float(corr):.6f}"
            for name, corr in zip(["B2", "B3", "B4", "B8"], real_sim_corr.values())
        ))
        print(
            "NOTE: SIM_GT_MSI uses the frozen S2B SRF projection of EnMAP10 "
            "for the GT side; exact agreement with the S2eteS simulator is "
            "not assumed. Per-band correlation is gain/bias-insensitive."
        )

    if model is not None:
        d2_phy.report("D2_PHY")
        d2_msi.report("D2_MSI", ["B2", "B3", "B4", "B8"])
        psnr, sam = _metrics_from_sums(d2_sse, d2_n, d2_sam, d2_sam_n)
        print(f"D2_IDENTITY REF_PSNR={psnr:.6f} REF_SAM={sam:.6f}")
    print("INTERPRETATION: Smaller D2 observation residuals than GT residuals "
          "can indicate a trade-off between observation fitting and reference "
          "accuracy; they do not by themselves prove which sensor model is wrong.")


if __name__ == "__main__":
    main()
