"""Complete Augsburg-2 Center-Holdout Wald C + Diffusion96 + HPGC original-scale inference.

Input: original EnMAP-like 30m HSI and real Sentinel-2 10m MSI, both from
MDAS sub_area_2. Does not load 10m HSI ground truth or run validation on it.
Output: 10m 242-band HSI in the original MSI georeferenced frame.

Overlap-add tiles avoid tiny edge tiles and fit a 12 GB GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from augsburg2_wald_qnr import evaluate_cache
from augsburg2_wald_center_roi import crop_heldout, geotiff_transform
from augsburg_real_process import build_augsburg_real_process
from degradations.deformation_aware import DeformationAwareProgressiveDegradation
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from models import ranked_msi_heterogeneity
from train_augsburg_real_cdrdi import _checkpoint_extra, _wald_checkpoint_guard
from train_augsburg_real_diffusion import _batch_geometry, _wald_d2_extra_guard
from train_augsburg_real_gigi import (
    _build_projector, _build_refiner, _load_training_refiner, _physical_residual,
)
from cdrdi_geometry import spectral_project
from config import TrainConfig
from innovation1 import reconstruct_from_terminal_lr
from main import build_model
from train_augsburg_real_diffusion import _apply_radiometry, _radiometry
from utils import get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Run full-resolution Augsburg-2 Wald diffusion fusion")
    p.add_argument("--center_holdout", action="store_true",
                   help="Use the trained center-test ROI protocol, own checkpoint, calibration and results folder")
    p.add_argument("--wald_root", default="./data/augsburg2_wald")
    p.add_argument("--checkpoint", required=True,
                   help="Wald-C 96-channel diffusion checkpoint")
    p.add_argument("--geometry_checkpoint", required=True)
    p.add_argument("--refiner_checkpoint", required=True)
    p.add_argument("--geometry_steps", type=int, default=9)
    p.add_argument("--fixed_dx_px", type=float, default=-0.5)
    p.add_argument("--fixed_dy_px", type=float, default=0.0)
    p.add_argument("--guard_window", type=int, default=5)
    p.add_argument("--guard_min_jac", type=float, default=0.5)
    p.add_argument("--guard_relative_gain", type=float, default=1e-4)
    p.add_argument("--refine_hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--variant", default="full", choices=("full",))
    p.add_argument("--disable_tangent", action="store_true")
    p.add_argument("--radiometry_json", default="./data/calibration/Augsburg2_Wald_radiometry.json")
    p.add_argument("--save_root", default="./outputs/augsburg2_wald")
    p.add_argument("--tile_size", type=int, default=96)
    p.add_argument("--tile_stride", type=int, default=48)
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--base_channels", type=int, default=96)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--write_tif", action="store_true")
    p.add_argument("--skip_qnr", action="store_true",
                   help="Skip native-scale HSI-MSI QNR (normally computed and saved on heldout test ROI)")
    p.add_argument("--qnr_window_hr", type=int, default=48,
                   help="UIQI window on original 10m grid (matches UAFL)")
    p.add_argument("--qnr_min_valid_fraction", type=float, default=0.8,
                   help="Minimum valid fraction per non-overlapping UIQI window")
    p.add_argument("--qnr_support_fraction", type=float, default=0.01,
                   help="SRF relative-to-peak threshold for MSI-covered HSI bands")
    args = p.parse_args()
    if args.center_holdout:
        if args.wald_root == "./data/augsburg2_wald":
            args.wald_root = "./data/augsburg2_wald_center_holdout"
        if args.radiometry_json == "./data/calibration/Augsburg2_Wald_radiometry.json":
            args.radiometry_json = "./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json"
        if args.save_root == "./outputs/augsburg2_wald":
            args.save_root = "./outputs/augsburg2_wald_center_holdout"
    else:
        p.error("This full96 entrypoint requires --center_holdout")
    if args.center_holdout and args.skip_qnr:
        p.error("--center_holdout requires QNR output; omit --skip_qnr")
    if args.base_channels != 96:
        p.error("Full96 inference requires --base_channels 96")
    return args


def _tile_rigid(global_rigid, top, left, tile_size, height, width):
    """Transfer a global-center rigid warp to a crop-centered sampling grid."""
    angle = global_rigid[:, 2] * (math.pi / 180.0)
    cos, sin = torch.cos(angle), torch.sin(angle)
    off_x = left + (tile_size - 1.0) / 2.0 - (width - 1.0) / 2.0
    off_y = top + (tile_size - 1.0) / 2.0 - (height - 1.0) / 2.0
    rigid = global_rigid.clone()
    rigid[:, 0] += (cos - 1.0) * off_x - sin * off_y
    rigid[:, 1] += sin * off_x + (cos - 1.0) * off_y
    return rigid


def _geometry_from_wald_30m(geometry_model, base_process, lr, msi, valid, radiometry, srf, args, device):
    """Solve geometry at the Wald-trained 30m scale, then scale motion to 10m."""
    hsi_30 = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1).unsqueeze(0).to(device)
    msi_10 = torch.from_numpy(np.ascontiguousarray(msi)).permute(2, 0, 1).unsqueeze(0).to(device)
    msi_10 = _apply_radiometry(msi_10, radiometry)
    hsi_90 = base_process.terminal_observation(hsi_30)
    msi_30 = F.avg_pool2d(msi_10, 3, 3)
    mask_10 = torch.from_numpy(valid.copy()).unsqueeze(0).unsqueeze(0).to(device).float()
    mask_30 = F.avg_pool2d(mask_10, 3, 3) >= 0.999
    rigid_30, local_30 = _batch_geometry(
        args, geometry_model, hsi_90, msi_30, mask_30,
        p0=base_process.operator, srf=srf,
    )
    if local_30.abs().amax().item() > 1e-7:
        raise ValueError("Wald C geometry must remain rigid-only")
    rigid_10 = rigid_30.clone()
    rigid_10[:, :2] *= 3.0
    return rigid_10, rigid_30


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
    if extra.get("geometry_mode") != "wald_cdrdi":
        raise ValueError("Requires a Wald C geometry-aware diffusion checkpoint")
    if extra.get("geometry_checkpoint") != args.geometry_checkpoint:
        raise ValueError("Diffusion checkpoint refers to another geometry checkpoint")
    if int(extra.get("geometry_steps", -1)) != args.geometry_steps:
        raise ValueError("Diffusion/geometry recursion steps mismatch")
    if int(extra.get("scale_ratio", -1)) != 3:
        raise ValueError("Expected Wald 3x reconstruction")
    checkpoint_rad = extra.get("radiometry_json", "")
    if (not checkpoint_rad or
        os.path.realpath(checkpoint_rad) != os.path.realpath(args.radiometry_json)):
        raise ValueError(
            "Full inference radiometry differs from Wald-D2 checkpoint. "
            f"checkpoint={checkpoint_rad!r}; supplied={args.radiometry_json!r}"
        )
    if abs(float(extra.get("effective_sigma", -1)) - sigma) > 1e-7:
        raise ValueError("Full inference PSF does not match the Wald-D2 checkpoint")
    lr, msi, valid, roi_y0, roi_x0, output_suffix, split_protocol_id = crop_heldout(
        args.wald_root, lr, msi, valid,
        ckpt_protocol_id=extra.get("split_protocol_id"),
        ckpt_bbox_30m=extra.get("test_bbox_30m"),
    )
    h, w = msi.shape[:2]
    with open(args.radiometry_json, encoding="utf-8") as handle:
        radiometry_meta = json.load(handle)
    if (radiometry_meta.get("spatial_protocol_id", "legacy_full_region_wald")
        != split_protocol_id):
        raise ValueError(
            "Full-resolution inference radiometry belongs to a different spatial split; "
            "re-fit outside center holdout"
        )
    if h % 6 or w % 6:
        raise ValueError("Held-out area must be divisible by 6 for the diffusion stages")
    print(
        f"WALD_INFERENCE spatial_protocol={split_protocol_id} "
        f"area={output_suffix} HR_shape={h}x{w} "
        f"origin_10m_rowcol=({roi_y0},{roi_x0})"
    )
    load_checkpoint(model, args.checkpoint, map_location=str(device), load_optimizer=False)
    model.eval().requires_grad_(False)

    geom_extra = _wald_checkpoint_guard(args.geometry_checkpoint, stage="Wald-Full96-Inference")
    if (geom_extra.get("protocol_id") != split_protocol_id
        or geom_extra.get("test_bbox_30m") != extra.get("test_bbox_30m")):
        raise ValueError("Geometry checkpoint was trained with a different center holdout")
    geom_config = geom_extra["geometry_config"]
    geometry_model = LearnedPhysicalResidualSolver(
        4, **{
            key: int(geom_config[key]) if key in ("base_channels", "control_grid")
            else float(geom_config[key])
            for key in (
                "base_channels", "control_grid", "max_translation",
                "max_rotation_deg", "max_local_px", "initial_dx_px", "initial_dy_px"
            )
        }
    ).to(device)
    load_checkpoint(
        geometry_model, args.geometry_checkpoint,
        map_location=str(device), load_optimizer=False,
    )
    geometry_model.eval().requires_grad_(False)
    ref_args = SimpleNamespace(
        refine_hidden=args.refine_hidden, heads=args.heads,
        variant=args.variant, disable_tangent=args.disable_tangent,
        diffusion_steps=args.diffusion_steps,
    )
    ref_extra = _checkpoint_extra(args.refiner_checkpoint)
    if (ref_extra.get("geometry_checkpoint") != args.geometry_checkpoint
        or ref_extra.get("diffusion_checkpoint") != args.checkpoint
        or ref_extra.get("wald_protocol_id") != split_protocol_id
        or int(ref_extra.get("diffusion_base_channels", -1)) != args.base_channels):
        raise ValueError("Refiner was not trained with this Wald C/96/split combination")
    refiner = _build_refiner(ref_args, info, device)
    _load_training_refiner(refiner, args.refiner_checkpoint, device)
    refiner.eval().requires_grad_(False)
    projector = _build_projector(info, ref_args, device)

    process = build_augsburg_real_process(
        effective_sigma=sigma, diffusion_steps=args.diffusion_steps
    )
    radiometry = _radiometry(args.radiometry_json, device)
    srf = torch.as_tensor(
        np.load(os.path.join(args.wald_root, "srf_weights.npy")),
        dtype=torch.float32, device=device,
    )
    geometry_args = SimpleNamespace(
        geometry_mode="wald_cdrdi", geometry_steps=args.geometry_steps,
        fixed_dx_px=args.fixed_dx_px, fixed_dy_px=args.fixed_dy_px,
        guard_window=args.guard_window, guard_min_jac=args.guard_min_jac,
        guard_relative_gain=args.guard_relative_gain,
    )
    _wald_d2_extra_guard(
        args.checkpoint,
        SimpleNamespace(**{
            **vars(geometry_args),
            "wald_protocol_id": split_protocol_id,
            "test_bbox_30m": extra.get("test_bbox_30m"),
            "radiometry_json": args.radiometry_json,
            "geometry_checkpoint": args.geometry_checkpoint,
        }),
        msi_source="real_Sentinel_2_Wald_30m", sigma=sigma,
    )
    rigid_10, rigid_30 = _geometry_from_wald_30m(
        geometry_model, process, lr, msi, valid, radiometry, srf, geometry_args, device
    )
    print(
        "FULL96_GEOMETRY rigid_30m=" + str(rigid_30[0].tolist())
        + " rigid_10m=" + str(rigid_10[0].tolist())
    )
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
            tile_rigid = _tile_rigid(
                rigid_10, top, left, ts, h, w
            )
            tile_local = y_m.new_zeros((1, 2, ts, ts))
            tile_process = DeformationAwareProgressiveDegradation(
                process, rigid=tile_rigid, local_field=tile_local
            )
            base = reconstruct_from_terminal_lr(
                model, tile_process, y_h,
                target_size=(ts, ts), hr_msi=y_m,
            )
            residual = _physical_residual(tile_process, projector, y_h, base)
            heterogeneity = ranked_msi_heterogeneity(y_m)
            pred = refiner(base, y_m, residual, heterogeneity)
            array = pred[0].permute(1, 2, 0).cpu().numpy()
            sum_cube[top:top + ts, left:left + ts, :] += array
            weight[top:top + ts, left:left + ts] += 1.
            print(f"FULL_TILE {i + 1}/{len(ys)} {j + 1}/{len(xs)} at={top},{left}")

    if not np.all(weight > 0):
        raise AssertionError("Not all full-resolution pixels were reconstructed")
    fused = sum_cube / weight[..., None]
    fused[~valid] = 0.
    os.makedirs(args.save_root, exist_ok=True)
    output_path = os.path.join(args.save_root, f"Augsburg2_Wald_{output_suffix}_HSI.npy")
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
        full_process = DeformationAwareProgressiveDegradation(
            process, rigid=rigid_10,
            local_field=ph.new_zeros((1, 2, h, w)),
        )
        predicted_lr = full_process.terminal_observation(ph)
        predicted_msi = spectral_project(ph, srf)
        phy = (predicted_lr - lh).abs()[valid_lr.expand_as(predicted_lr)].mean().item()
        msi_l1 = (predicted_msi - hm).abs()[valid_hr.expand_as(predicted_msi)].mean().item()

    qnr = None
    if not args.skip_qnr:
        qnr = evaluate_cache(
            args.wald_root, output_path, args.radiometry_json,
            window_hr=args.qnr_window_hr,
            min_valid_fraction=args.qnr_min_valid_fraction,
            support_fraction=args.qnr_support_fraction,
        )
        qnr_json = os.path.join(args.save_root, f"Augsburg2_Wald_{output_suffix}_QNR.json")
        with open(qnr_json, "w", encoding="utf-8") as f:
            json.dump(qnr, f, ensure_ascii=False, indent=2)
        print(
            f"S2DIFF_MH_WALD_HSI_MSI_QNR "
            f"QNR={qnr['QNR']:.6f} "
            f"Dlambda={qnr['Dlambda']:.6f} "
            f"Ds={qnr['Ds']:.6f} "
            f"SPECTRAL_REFERENCE=242_band_LR_HSI "
            f"SPATIAL_REFERENCE=original_HR_MSI QNR_JSON={qnr_json}"
        )

    if args.write_tif:
        from affine import Affine
        import rasterio
        raster_path = os.path.join(args.save_root, f"Augsburg2_Wald_{output_suffix}_HSI.tif")
        with rasterio.open(
            raster_path, "w", driver="GTiff", height=h, width=w, count=242,
            dtype="float32", crs=full_meta["crs"],
            transform=geotiff_transform(full_meta["transform_6"], roi_y0, roi_x0),
            compress="deflate", tiled=True,
        ) as dst:
            for band in range(242):
                dst.write(fused[:, :, band], band + 1)
        print(f"FULL_OUTPUT_GEOTIFF={raster_path}")
    report_path = os.path.join(args.save_root, f"Augsburg2_Wald_{output_suffix}_protocol.json")
    report = {
        "model": "S2Diff-MH-Full96-CDRDI-Diffusion-HPGC",
        "stage": "original_10m_inference",
        "spatial_protocol_id": split_protocol_id,
        "evaluation_area": output_suffix,
        "test_bbox_30m": extra.get("test_bbox_30m"),
        "checkpoint": os.path.realpath(args.checkpoint),
        "geometry_checkpoint": os.path.realpath(args.geometry_checkpoint),
        "refiner_checkpoint": os.path.realpath(args.refiner_checkpoint),
        "geometry_rigid_30m": rigid_30[0].tolist(),
        "geometry_rigid_10m": rigid_10[0].tolist(),
        "radiometry_json": os.path.realpath(args.radiometry_json),
        "wald_root": os.path.realpath(args.wald_root),
        "reconstruction": os.path.realpath(output_path),
        "shape": list(fused.shape),
        "ground_truth_10m_HSI": False,
        "qnr": qnr,
        "phy_l1": phy,
        "msi_l1": msi_l1,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"FULL_OUTPUT_NPY={output_path} shape={fused.shape} "
        f"PHY_L1={phy:.8f} MSI_L1={msi_l1:.8f} "
        f"PROTOCOL_JSON={report_path} HR_HSI_REFERENCE=unavailable"
    )


if __name__ == "__main__":
    main()
