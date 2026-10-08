"""Augsburg-2 full-resolution fusion using a strictly Wald-trained D2.

Input: original EnMAP-like 30m HSI and real Sentinel-2 10m MSI, both from
MDAS sub_area_2. Does not load 10m HSI ground truth or run validation on it.
Output: 10m 242-band HSI in the original MSI georeferenced frame.

Overlap-add tiles avoid tiny edge tiles and fit a 12 GB GPU.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from augsburg2_wald_qnr import evaluate_cache
from augsburg_real_process import build_augsburg_real_process
from cdrdi_geometry import spectral_project
from config import TrainConfig
from innovation1 import reconstruct_from_terminal_lr
from main import build_model
from train_augsburg_real_diffusion import _apply_radiometry, _radiometry
from utils import get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Run full-resolution Augsburg-2 Wald diffusion fusion")
    p.add_argument("--wald_root", default="./data/augsburg2_wald")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--radiometry_json", default="./data/calibration/Augsburg2_Wald_radiometry.json")
    p.add_argument("--save_root", default="./outputs/augsburg2_wald")
    p.add_argument("--tile_size", type=int, default=96)
    p.add_argument("--tile_stride", type=int, default=48)
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--write_tif", action="store_true")
    p.add_argument("--skip_qnr", action="store_true",
                   help="Disable UAFL-identical MSI-projected modified QNR; default evaluates it")
    p.add_argument("--qnr_window_hr", type=int, default=48,
                   help="UIQI window on original 10m grid (matches UAFL)")
    p.add_argument("--qnr_min_valid_fraction", type=float, default=0.8,
                   help="Minimum valid fraction per non-overlapping UIQI window")
    return p.parse_args()


def positions(length, tile, stride):
    if length < tile:
        raise ValueError(f"Full-resolution side {length} is smaller than tile {tile}")
    coords = list(range(0, length - tile + 1, stride))
    last = length - tile
    if not coords or coords[-1] != last:
        coords.append(last)
    if any(v % 3 for v in coords):
        raise ValueError("Tile origins must be multiples of 3 for matched LR-HSI")
    return coords


@torch.no_grad()
def main():
    args = parse_args()
    if args.tile_size <= 0 or args.tile_size % 6:
        raise ValueError("--tile_size must be a positive multiple of 6")
    if args.tile_stride <= 0 or args.tile_stride % 6:
        raise ValueError("--tile_stride must be a positive multiple of 6")
    if args.tile_stride > args.tile_size:
        raise ValueError("--tile_stride must not exceed --tile_size")
    set_seed(args.seed)
    device = get_device(args.device)
    with open(os.path.join(args.wald_root, "wald_psf.json"), "r", encoding="utf-8") as f:
        sigma = float(json.load(f)["terminal_sigma_hr_pixels"])
    with open(os.path.join(args.wald_root, "full", "meta.json"), "r", encoding="utf-8") as f:
        full_meta = json.load(f)
    if full_meta.get("region") != "sub_area_2":
        raise ValueError("Strict Augsburg-2 full-resolution no-reference evaluation requires sub_area_2")

    lr = np.load(os.path.join(args.wald_root, "full", "lr_hsi.npy"), mmap_mode="r")
    msi = np.load(os.path.join(args.wald_root, "full", "hr_msi.npy"), mmap_mode="r")
    valid = np.load(os.path.join(args.wald_root, "full", "valid_mask.npy"), mmap_mode="r").astype(bool)
    h, w, channels = msi.shape
    if msi.shape[2] != 4 or lr.shape[2] != 242:
        raise ValueError(f"Expected LR HSI242 and MSI4, got LR={lr.shape}, MSI={msi.shape}")
    if (h, w) != (3 * lr.shape[0], 3 * lr.shape[1]):
        raise ValueError("Full-resolution LR-HSI/HR-MSI have different geographic sizes")
    if h % 6 or w % 6:
        raise ValueError(
            f"Full region {(h, w)} must be divisible by 6; "
            "use a full-resolution region with x3 and x2 compatibility"
        )

    info = {"n_bands": 242, "n_msi_bands": 4}
    cfg = TrainConfig(
        dataset="Augsburg",
        scale_ratio=3,
        diffusion_steps=args.diffusion_steps,
        predictor="raw_direct",
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        spectral_hidden=args.spectral_hidden,
        dropout=args.dropout,
        device=args.device,
    )
    model = build_model(cfg, info, device)
    # Strict Wald provenance check when metadata is present.
    try:
        checkpoint_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint_state = torch.load(args.checkpoint, map_location="cpu")
    extra = checkpoint_state.get("extra", {}) if isinstance(checkpoint_state, dict) else {}
    if (extra.get("stage") != "Augsburg2-Wald-D2"
        or extra.get("msi_source") != "real_Sentinel_2_Wald_30m"
        or extra.get("reference_supervision_source") != "observed_30m_HSI_only"):
        raise ValueError(
            "Full inference requires strictly Wald-D2 trained on observed 30m HSI, "
            "never legacy EnMAP10 supervision"
        )
    # Current full inference constructs an identity physics process. Reusing
    # B/C checkpoints here would silently ignore their nonzero geometry.
    # Future B/C full inference must explicitly transform 30m-pixel motion
    # into 10m-pixel units and use a geometry-aware process.
    if extra.get("geometry_mode") != "identity":
        raise ValueError(
            "Full inference currently supports Wald A/identity only. "
            "B/C checkpoints require geometrically consistent 30m->10m "
            "motion scaling; refusing to evaluate them with identity physics."
        )
    if os.path.normpath(extra.get("radiometry_json", "")) != os.path.normpath(args.radiometry_json):
        raise ValueError("Full inference radiometry differs from the Wald-D2 checkpoint")
    if abs(float(extra.get("effective_sigma", -1)) - sigma) > 1e-7:
        raise ValueError("Full inference PSF does not match the Wald-D2 checkpoint")
    load_checkpoint(model, args.checkpoint, map_location=str(device), load_optimizer=False)
    model.eval()

    process = build_augsburg_real_process(
        effective_sigma=sigma, diffusion_steps=args.diffusion_steps
    )
    radiometry = _radiometry(args.radiometry_json, device)
    sum_cube = np.zeros((h, w, 242), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    ys = positions(h, args.tile_size, args.tile_stride)
    xs = positions(w, args.tile_size, args.tile_stride)

    for i, top in enumerate(ys):
        for j, left in enumerate(xs):
            ts = args.tile_size
            lr_patch = np.ascontiguousarray(
                lr[top // 3:(top + ts) // 3, left // 3:(left + ts) // 3, :]
            )
            msi_patch = np.ascontiguousarray(
                msi[top:top + ts, left:left + ts, :]
            )
            y_h = torch.from_numpy(lr_patch).permute(2, 0, 1).unsqueeze(0).to(device)
            y_m = torch.from_numpy(msi_patch).permute(2, 0, 1).unsqueeze(0).to(device)
            y_m = _apply_radiometry(y_m, radiometry)
            pred = reconstruct_from_terminal_lr(
                model, process, y_h,
                target_size=(ts, ts), hr_msi=y_m,
            )
            array = pred[0].permute(1, 2, 0).cpu().numpy()
            sum_cube[top:top + ts, left:left + ts, :] += array
            weight[top:top + ts, left:left + ts] += 1.
            print(f"FULL_TILE {i + 1}/{len(ys)} {j + 1}/{len(xs)} at={top},{left}")

    if not np.all(weight > 0):
        raise AssertionError("Not all full-resolution pixels were reconstructed")
    fused = sum_cube / weight[..., None]
    fused[~valid] = 0.
    os.makedirs(args.save_root, exist_ok=True)
    output_path = os.path.join(args.save_root, "Augsburg2_Wald_full_HSI.npy")
    np.save(output_path, fused.astype(np.float32))

    # Diagnostic observation consistency only. No HR-HSI reference exists here.
    with torch.no_grad():
        srf = torch.as_tensor(
            np.load(os.path.join(args.wald_root, "srf_weights.npy")),
            dtype=torch.float32, device=device,
        )
        ph = torch.from_numpy(fused).permute(2, 0, 1).unsqueeze(0).to(device)
        lh = torch.from_numpy(np.asarray(lr).copy()).permute(2, 0, 1).unsqueeze(0).to(device)
        hm = torch.from_numpy(np.asarray(msi).copy()).permute(2, 0, 1).unsqueeze(0).to(device)
        hm = _apply_radiometry(hm, radiometry)
        valid_hr = torch.from_numpy(valid.copy()).unsqueeze(0).unsqueeze(0).to(device)
        valid_lr = F.avg_pool2d(valid_hr.float(), 3, 3) >= 0.999
        predicted_lr = process.terminal_observation(ph)
        predicted_msi = spectral_project(ph, srf)
        phy = (predicted_lr - lh).abs()[valid_lr.expand_as(predicted_lr)].mean().item()
        msi_l1 = (predicted_msi - hm).abs()[valid_hr.expand_as(predicted_msi)].mean().item()

    qnr = None
    if not args.skip_qnr:
        qnr = evaluate_cache(
            args.wald_root, output_path, args.radiometry_json,
            window_hr=args.qnr_window_hr,
            min_valid_fraction=args.qnr_min_valid_fraction,
        )
        qnr_json = os.path.join(args.save_root, "Augsburg2_Wald_full_QNR.json")
        with open(qnr_json, "w", encoding="utf-8") as f:
            json.dump(qnr, f, ensure_ascii=False, indent=2)
        print(
            f"S2DIFF_MH_WALD_ORIGINAL_MSI_QNR "
            f"QNR={qnr['QNR']:.6f} "
            f"Dlambda={qnr['Dlambda']:.6f} "
            f"Ds={qnr['Ds']:.6f} QNR_JSON={qnr_json} "
            "METRIC=MSI_projected_modified_QNR_NOT_full_242_band_quality"
        )

    if args.write_tif:
        from affine import Affine
        import rasterio
        raster_path = os.path.join(args.save_root, "Augsburg2_Wald_full_HSI.tif")
        with rasterio.open(
            raster_path, "w", driver="GTiff", height=h, width=w, count=242,
            dtype="float32", crs=full_meta["crs"],
            transform=Affine(*full_meta["transform_6"]),
            compress="deflate", tiled=True,
        ) as dst:
            for band in range(242):
                dst.write(fused[:, :, band], band + 1)
        print(f"FULL_OUTPUT_GEOTIFF={raster_path}")
    print(
        f"FULL_OUTPUT_NPY={output_path} shape={fused.shape} "
        f"PHY_L1={phy:.8f} MSI_L1={msi_l1:.8f} "
        "HR_HSI_REFERENCE=unavailable"
    )


if __name__ == "__main__":
    main()
