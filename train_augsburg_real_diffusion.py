"""Real-D2: observation-anchored diffusion adaptation for Augsburg-Real.

The output latent remains in the real Sentinel-2 reference coordinate system.
In estimated mode, EnMAP10 supervision is mapped into the MSI latent frame via
the normalized adjoint of the Real-C estimated warp. Identity mode is an
explicit georeferenced no-geometry baseline: reference, diffusion latent and
real MSI remain on the metadata-harmonized 10 m grid. Neither mode artificially
warps observed LR-HSI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from types import SimpleNamespace
from statistics import mean

import torch
import torch.nn.functional as F

from augsburg_real import build_augsburg_real_loaders
from augsburg_real_process import build_augsburg_real_process
from cdrdi_geometry import forward_warp, spectral_project
from config import TrainConfig
from degradations.deformation_aware import (
    DeformationAwareProgressiveDegradation,
    bilinear_border_warp_adjoint,
)
from innovation1 import model_predict, reconstruct_from_terminal_lr
from main import build_model
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from train_cdrdi_diffusion_estimated import observation_anchored_batch_state
from train_augsburg_real_cdrdi import (
    _wald_metadata, _wald_checkpoint_guard,
    _assert_wald_checkpoint_settings, _checkpoint_extra,
)
from utils import CSVLogger, ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Augsburg-Real Real-D2 diffusion adaptation")
    p.add_argument("--stage", choices=["train", "test"], default="train")
    p.add_argument("--cache_root", default="./data/augsburg_real_cache")
    p.add_argument("--psf_json", default="./data/calibration/AugsburgReal_effective_psf.json")
    p.add_argument("--radiometry_json", default="./data/calibration/AugsburgReal_radiometry.json")
    p.add_argument("--geometry_mode", choices=["estimated", "identity", "wald_fixed", "wald_cdrdi"], default="estimated",
                   help="strict Wald A=identity, B=wald_fixed, C=wald_cdrdi (frozen rigid+closure guard)")
    p.add_argument("--geometry_checkpoint", default="",
                   help="Required for estimated and wald_cdrdi modes; NEVER synthetic/EnMAP10 for Wald")
    p.add_argument("--fixed_dx_px", type=float, default=-0.5,
                   help="Wald B fixed horizontal offset in native Wald 30m MSI pixels")
    p.add_argument("--fixed_dy_px", type=float, default=0.0,
                   help="Wald B fixed vertical offset in native Wald 30m MSI pixels")
    p.add_argument("--guard_min_jac", type=float, default=0.5)
    p.add_argument("--guard_relative_gain", type=float, default=1e-4)
    p.add_argument("--guard_window", type=int, default=5)
    p.add_argument("--init_checkpoint", default="", help="Synthetic Augsburg D2 or Stage-A checkpoint")
    p.add_argument("--from_scratch", action="store_true",
                   help="Initialize without any pretraining checkpoint; required for strict Augsburg-2 Wald protocol")
    p.add_argument("--diffusion_checkpoint", default="", help="Trained Augsburg-Real D2 checkpoint for --stage test")
    p.add_argument("--checkpoint_root", default="./checkpoints/augsburg_real")
    p.add_argument("--log_root", default="./logs/augsburg_real")
    p.add_argument("--save_name", default="AugsburgReal_D2_estimated_geometry")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)

    p.add_argument("--train_patch_size", type=int, default=96)
    p.add_argument("--train_stride", type=int, default=48)
    p.add_argument("--eval_patch_size", type=int, default=192)
    p.add_argument("--min_valid_fraction", type=float, default=0.80)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--geometry_steps", type=int, default=9)
    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)

    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--boundary_probability", type=float, default=0.2)
    p.add_argument("--boundary_radius", type=int, default=1)

    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_sam", type=float, default=0.1)
    p.add_argument("--lambda_ref", type=float, default=0.5)
    p.add_argument("--lambda_phy", type=float, default=0.1)
    p.add_argument("--lambda_msi", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--monitor", choices=["ref_sam", "ref_psnr"], default="ref_sam")
    p.add_argument("--resume", default="")
    return p.parse_args()


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _radiometry(path: str, device):
    if not path:
        return None
    payload = _load_json(path)
    gain = torch.tensor(payload["gain"], dtype=torch.float32, device=device)[None, :, None, None]
    bias = torch.tensor(payload["bias"], dtype=torch.float32, device=device)[None, :, None, None]
    return gain, bias


def _apply_radiometry(msi, params):
    if params is None:
        return msi
    gain, bias = params
    return gain * msi + bias


def _estimate_geometry(model, lr_hsi, hr_msi, *, p0, srf, steps):
    target = spectral_project(lr_hsi, srf)
    out = model(target, hr_msi, p0, steps=steps)
    return out["final_rigid"].detach(), out["final_local_field"].detach()


def _geometry_required(args) -> bool:
    return args.geometry_mode != "identity"


def _fixed_wald_geometry(hr_msi, *, dx, dy):
    """Constant physical lag in the MSI30 pixel grid; no input pre-warp."""
    batch, _, h, w = hr_msi.shape
    rigid = hr_msi.new_zeros((batch, 3))
    rigid[:, 0] = float(dx)
    rigid[:, 1] = float(dy)
    local = hr_msi.new_zeros((batch, 2, h, w))
    return rigid, local


def _batch_geometry(args, geometry_model, y_h, y_m, mask_ref, *, p0, srf):
    """Return B/legacy/C geometry on the same unaltered observed MSI input."""
    if args.geometry_mode == "identity":
        return None, None
    if args.geometry_mode == "wald_fixed":
        return _fixed_wald_geometry(
            y_m, dx=args.fixed_dx_px, dy=args.fixed_dy_px
        )
    if args.geometry_mode == "wald_cdrdi":
        if geometry_model is None:
            raise ValueError("wald_cdrdi requires a frozen Wald-only geometry checkpoint")
        target = spectral_project(y_h, srf)
        # Output latent is in the MSI30 frame; observed HSI90 is never warped.
        out = geometry_model(
            target, y_m, p0, steps=args.geometry_steps,
            update_mode="rigid_only",
            update_policy="closure_backtrack",
            acceptance_mask=_lr_mask(mask_ref),
            acceptance_window=args.guard_window,
            acceptance_min_jac=args.guard_min_jac,
            acceptance_relative_gain=args.guard_relative_gain,
        )
        local = out["final_local_field"].detach()
        if bool((local.abs().amax() > 1e-7).item()):
            raise RuntimeError("Wald C mode must not introduce a local warp")
        return out["final_rigid"].detach(), local
    if args.geometry_mode == "estimated":
        if geometry_model is None:
            raise ValueError("estimated geometry requires a checkpoint")
        return _estimate_geometry(
            geometry_model, y_h, y_m, p0=p0, srf=srf,
            steps=args.geometry_steps
        )
    raise ValueError(f"Unsupported geometry mode: {args.geometry_mode}")


def _wald_d2_provenance(args, *, sigma):
    """Guard all Wald split metadata and frozen C model provenance."""
    if not _wald_metadata(args.cache_root):
        raise ValueError("Strict Wald requires all three splits to be genuine Wald 30m observations")
    if args.geometry_mode not in ("identity", "wald_fixed", "wald_cdrdi"):
        raise ValueError("Wald A/B/C only supports identity, wald_fixed and wald_cdrdi")
    if args.init_checkpoint or (args.from_scratch and args.resume):
        raise ValueError("Wald-D2 must train from scratch or resume its own Wald-only checkpoint")
    if args.stage == "train" and not (args.from_scratch or args.resume):
        raise ValueError("Wald-D2 training requires --from_scratch or --resume")
    if args.geometry_mode == "wald_fixed":
        if abs(args.fixed_dx_px) > 1.5 or abs(args.fixed_dy_px) > 1.5:
            raise ValueError("Fixed Wald shift exceeds conservative 30m range")
    if args.geometry_mode == "wald_cdrdi":
        if not args.geometry_checkpoint:
            raise ValueError("C mode requires --geometry_checkpoint from strictly Wald-trained CDRDI")
        extra = _wald_checkpoint_guard(args.geometry_checkpoint, stage="Wald-D2")
        config = extra["geometry_config"]
        if (abs(float(config["initial_dx_px"]) - float(args.fixed_dx_px)) > 1e-7
            or abs(float(config["initial_dy_px"]) - float(args.fixed_dy_px)) > 1e-7):
            raise ValueError("A/B/C geometry seed must match the frozen CDRDI initial offset")
        _assert_wald_checkpoint_settings(extra, SimpleNamespace(
            base_channels=int(config["base_channels"]),
            control_grid=int(config["control_grid"]),
            max_translation=float(config["max_translation"]),
            max_rotation_deg=float(config["max_rotation_deg"]),
            max_local_px=float(config["max_local_px"]),
            initial_dx_px=float(config["initial_dx_px"]),
            initial_dy_px=float(config["initial_dy_px"]),
            radiometry_json=args.radiometry_json,
        ), sigma=sigma)
    elif args.geometry_checkpoint:
        raise ValueError("Wald A/B must not supply a geometry checkpoint")
    if args.guard_window != 5 or abs(args.guard_min_jac - 0.5) > 1e-6:
        raise ValueError("Wald C guard must reproduce validated window=5, min_jac=0.5")
    if args.guard_relative_gain < 0 or args.guard_relative_gain >= 1:
        raise ValueError("Invalid Wald C guard threshold")


def _wald_d2_extra_guard(checkpoint, args, *, msi_source, sigma):
    """Never silently resume or test an unrelated branch/checkpoint."""
    extra = _checkpoint_extra(checkpoint)
    if (extra.get("msi_source") != msi_source
        or extra.get("stage") != "Augsburg2-Wald-D2"
        or extra.get("geometry_mode") != args.geometry_mode
        or extra.get("geometry_checkpoint", "") != args.geometry_checkpoint
        or abs(float(extra.get("effective_sigma", -1)) - sigma) > 1e-7
        or extra.get("radiometry_json") != args.radiometry_json):
        raise ValueError("Wald-D2 checkpoint differs in dataset, branch, PSF or calibration")
    if args.geometry_mode != "identity" and (
        abs(float(extra.get("fixed_dx_px", 99)) - args.fixed_dx_px) > 1e-7
        or abs(float(extra.get("fixed_dy_px", 99)) - args.fixed_dy_px) > 1e-7
    ):
        raise ValueError("Wald-D2 checkpoint has inconsistent fixed geometry seed")
    if args.geometry_mode == "wald_cdrdi" and (
        int(extra.get("geometry_steps", -1)) != int(args.geometry_steps)
        or float(extra.get("guard_relative_gain", -1)) != float(args.guard_relative_gain)
    ):
        raise ValueError("Wald-D2 checkpoint has inconsistent guarded CDRDI settings")


def _estimated_process(base_process, rigid, local):
    return DeformationAwareProgressiveDegradation(
        base_process,
        rigid=rigid.detach(),
        local_field=local.detach(),
    )


def _warp_adjoint_normalized(values, rigid, local, eps=1e-8):
    num = bilinear_border_warp_adjoint(
        values,
        rigid[:, 0], rigid[:, 1], rigid[:, 2], local,
        target_size=tuple(values.shape[-2:]),
    )
    ones = torch.ones(
        (values.shape[0], 1, values.shape[-2], values.shape[-1]),
        device=values.device,
        dtype=values.dtype,
    )
    den = bilinear_border_warp_adjoint(
        ones,
        rigid[:, 0], rigid[:, 1], rigid[:, 2], local,
        target_size=tuple(values.shape[-2:]),
    )
    return num / den.clamp_min(eps)


def _mask_to_msi(mask_ref, rigid, local):
    soft = _warp_adjoint_normalized(mask_ref.float(), rigid, local)
    return soft >= 0.90


def _lr_mask(mask_ref):
    return F.avg_pool2d(mask_ref.float(), 3, 3) >= 0.999


def _masked_l1(pred, target, mask):
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    values = (pred - target).abs()[mask]
    return values.mean() if values.numel() else pred.new_zeros(())


def _masked_sam(pred, target, mask, eps=1e-12):
    p = pred.float()
    t = target.float()
    valid = mask[:, 0] > 0.5
    pn = torch.linalg.vector_norm(p, dim=1)
    tn = torch.linalg.vector_norm(t, dim=1)
    valid = valid & (pn > eps) & (tn > eps)
    if not valid.any():
        return pred.new_zeros(())
    dot = (p * t).sum(dim=1)
    cos = (dot[valid] / (pn[valid] * tn[valid]).clamp_min(eps)).clamp(-1.0, 1.0)
    return torch.acos(cos).mean()


def _masked_metric_sums(pred, target, mask, eps=1e-12):
    if mask.shape[1] == 1:
        mask_band = mask.expand(-1, pred.shape[1], -1, -1)
    else:
        mask_band = mask
    diff = (pred - target)[mask_band]
    sse = float((diff.double() * diff.double()).sum().item())
    n_values = int(diff.numel())

    p = pred.float()
    t = target.float()
    valid = mask[:, 0] > 0.5
    pn = torch.linalg.vector_norm(p, dim=1)
    tn = torch.linalg.vector_norm(t, dim=1)
    valid = valid & (pn > eps) & (tn > eps)
    if valid.any():
        dot = (p * t).sum(dim=1)
        cos = (
            dot[valid] / (pn[valid] * tn[valid]).clamp_min(eps)
        ).clamp(-1.0, 1.0)
        angles = torch.acos(cos)
        sam_sum = float(angles.double().sum().item())
        sam_count = int(angles.numel())
    else:
        sam_sum = 0.0
        sam_count = 0
    return sse, n_values, sam_sum, sam_count


def _metrics_from_sums(sse, n_values, sam_sum, sam_count, eps=1e-12):
    if n_values <= 0:
        return float("nan"), float("nan")
    mse = sse / float(n_values)
    psnr = -10.0 * math.log10(max(mse, eps))
    sam = (
        sam_sum / float(sam_count) * 180.0 / math.pi
        if sam_count > 0
        else float("nan")
    )
    return psnr, sam


def _masked_metrics(pred, target, mask, eps=1e-12):
    return _metrics_from_sums(
        *_masked_metric_sums(pred, target, mask, eps=eps),
        eps=eps,
    )


def _config(args):
    return TrainConfig(
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


def train_one_epoch(
    model,
    geometry_model,
    loader,
    optimizer,
    *,
    base_process,
    p0,
    srf,
    radiometry,
    args,
    device,
):
    model.train()
    if geometry_model is not None:
        geometry_model.eval()
    sums = {k: 0.0 for k in ("loss", "l1", "sam", "ref", "phy", "msi")}
    count = 0
    for batch in loader:
        gt_ref = batch["gt"].to(device, non_blocking=True)
        y_h = batch["lr_hsi"].to(device, non_blocking=True)
        y_m = _apply_radiometry(
            batch["hr_msi"].to(device, non_blocking=True), radiometry
        )
        mask_ref = batch["valid_mask"].to(device, non_blocking=True) > 0.5

        with torch.no_grad():
            if _geometry_required(args):
                rigid, local = _batch_geometry(
                    args, geometry_model, y_h, y_m, mask_ref, p0=p0, srf=srf
                )
                process = _estimated_process(base_process, rigid, local)
                gt_msi = _warp_adjoint_normalized(gt_ref, rigid, local)
                mask_msi = _mask_to_msi(mask_ref, rigid, local)
            else:
                process = base_process
                gt_msi = gt_ref
                mask_msi = mask_ref
                rigid = None
                local = None
            timesteps = base_process.sample_timesteps(
                gt_ref.shape[0],
                boundary_probability=args.boundary_probability,
                boundary_radius=args.boundary_radius,
                device=device,
            )
            x_t = observation_anchored_batch_state(
                process,
                gt_msi,
                y_h,
                timesteps,
            )

        pred = model_predict(model, x_t, timesteps, y_m)
        l1 = _masked_l1(pred, gt_msi, mask_msi)
        sam = _masked_sam(pred, gt_msi, mask_msi)
        pred_ref = (
            forward_warp(
                pred, rigid[:, 0], rigid[:, 1], rigid[:, 2], local
            ) if _geometry_required(args) else pred
        )
        ref = _masked_l1(pred_ref, gt_ref, mask_ref)
        phy = _masked_l1(
            process.terminal_observation(pred), y_h, _lr_mask(mask_ref)
        )
        msi = _masked_l1(spectral_project(pred, srf), y_m, mask_msi)
        loss = (
            args.lambda_l1 * l1
            + args.lambda_sam * sam
            + args.lambda_ref * ref
            + args.lambda_phy * phy
            + args.lambda_msi * msi
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Augsburg-Real D2 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()

        n = gt_ref.shape[0]
        for key, value in (
            ("loss", loss),
            ("l1", l1),
            ("sam", sam),
            ("ref", ref),
            ("phy", phy),
            ("msi", msi),
        ):
            sums[key] += float(value.detach().item()) * n
        count += n
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(
    model,
    geometry_model,
    loader,
    *,
    base_process,
    p0,
    srf,
    radiometry,
    args,
    device,
):
    model.eval()
    if geometry_model is not None:
        geometry_model.eval()
    ref_sse = 0.0
    ref_values = 0
    ref_sam_sum = 0.0
    ref_sam_count = 0
    phy_sum = 0.0
    phy_weight = 0.0
    msi_sum = 0.0
    msi_weight = 0.0

    for batch in loader:
        gt_ref = batch["gt"].to(device)
        y_h = batch["lr_hsi"].to(device)
        y_m = _apply_radiometry(batch["hr_msi"].to(device), radiometry)
        mask_ref = batch["valid_mask"].to(device) > 0.5
        if _geometry_required(args):
            rigid, local = _batch_geometry(
                args, geometry_model, y_h, y_m, mask_ref, p0=p0, srf=srf
            )
            process = _estimated_process(base_process, rigid, local)
        else:
            rigid = None
            local = None
            process = base_process
        pred = reconstruct_from_terminal_lr(
            model,
            process,
            y_h,
            target_size=tuple(gt_ref.shape[-2:]),
            hr_msi=y_m,
        )
        pred_ref = (
            forward_warp(
                pred, rigid[:, 0], rigid[:, 1], rigid[:, 2], local
            ) if _geometry_required(args) else pred
        )
        sse, n_values, sam_sum, sam_count = _masked_metric_sums(
            pred_ref, gt_ref, mask_ref
        )
        ref_sse += sse
        ref_values += n_values
        ref_sam_sum += sam_sum
        ref_sam_count += sam_count

        lr_valid = _lr_mask(mask_ref)
        phy_value = float(
            _masked_l1(
                process.terminal_observation(pred), y_h, lr_valid
            ).item()
        )
        lr_weight = float(lr_valid[:, 0].sum().item())
        phy_sum += phy_value * lr_weight
        phy_weight += lr_weight

        mask_msi = (
            _mask_to_msi(mask_ref, rigid, local)
            if _geometry_required(args) else mask_ref
        )
        msi_value = float(
            _masked_l1(spectral_project(pred, srf), y_m, mask_msi).item()
        )
        hr_weight = float(mask_msi[:, 0].sum().item())
        msi_sum += msi_value * hr_weight
        msi_weight += hr_weight

    if ref_values <= 0:
        raise ValueError("empty/invalid validation loader")
    psnr, sam = _metrics_from_sums(
        ref_sse, ref_values, ref_sam_sum, ref_sam_count
    )
    return {
        "ref_psnr": psnr,
        "ref_sam": sam,
        "phy": phy_sum / max(phy_weight, 1.0),
        "msi": msi_sum / max(msi_weight, 1.0),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    train_meta = _load_json(os.path.join(args.cache_root, "train", "meta.json"))
    if train_meta.get("msi_source") == "real_Sentinel_2_Wald_30m":
        if args.psf_json == "./data/calibration/AugsburgReal_effective_psf.json":
            args.psf_json = os.path.join(args.cache_root, "wald_psf.json")
        if args.radiometry_json == "./data/calibration/AugsburgReal_radiometry.json":
            args.radiometry_json = "./data/calibration/Augsburg2_Wald_radiometry.json"
    sigma = float(_load_json(args.psf_json)["terminal_sigma_hr_pixels"])
    msi_source = train_meta.get("msi_source", "real_Sentinel_2")
    is_wald = msi_source == "real_Sentinel_2_Wald_30m"
    if is_wald:
        _wald_d2_provenance(args, sigma=sigma)
        if args.resume:
            _wald_d2_extra_guard(args.resume, args, msi_source=msi_source, sigma=sigma)
        if args.stage == "test":
            if not args.diffusion_checkpoint:
                raise ValueError("Wald --stage test requires --diffusion_checkpoint")
            _wald_d2_extra_guard(args.diffusion_checkpoint, args, msi_source=msi_source, sigma=sigma)
    elif args.geometry_mode in ("wald_fixed", "wald_cdrdi"):
        raise ValueError("Wald-only geometry modes cannot be used for synthetic/legacy Real-D2")
    if msi_source == "official_EeteS_simulated_Sentinel_2":
        if args.radiometry_json:
            raise ValueError(
                "Simulated-MSI control requires --radiometry_json '' "
                "(do not apply real-S2 radiometry to simulated MSI)"
            )
        if args.geometry_mode != "identity":
            raise ValueError(
                "Simulated-MSI control requires --geometry_mode identity "
                "(Real-C was trained against a different real MSI source)"
            )
    radiometry = _radiometry(args.radiometry_json, device)

    train_loader, val_loader, test_loader, info = build_augsburg_real_loaders(
        args.cache_root,
        train_patch_size=args.train_patch_size,
        train_stride=args.train_stride,
        eval_patch_size=args.eval_patch_size,
        min_valid_fraction=args.min_valid_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_augment=not is_wald,
    )
    srf = torch.as_tensor(info["srf_weights"], dtype=torch.float32, device=device)
    base_process = build_augsburg_real_process(
        effective_sigma=sigma,
        diffusion_steps=args.diffusion_steps,
    )
    p0 = base_process.operator.to(device)

    geometry_model = None
    if args.geometry_mode in ("estimated", "wald_cdrdi"):
        if not args.geometry_checkpoint:
            raise ValueError("--geometry_mode estimated/wald_cdrdi requires --geometry_checkpoint")
        if args.geometry_mode == "wald_cdrdi":
            cc = _wald_checkpoint_guard(args.geometry_checkpoint, stage="Wald-D2")["geometry_config"]
            geom_cfg = dict(
                base_channels=int(cc["base_channels"]),
                control_grid=int(cc["control_grid"]),
                max_translation=float(cc["max_translation"]),
                max_rotation_deg=float(cc["max_rotation_deg"]),
                max_local_px=float(cc["max_local_px"]),
                initial_dx_px=float(cc["initial_dx_px"]),
                initial_dy_px=float(cc["initial_dy_px"]),
            )
        else:
            geom_cfg = dict(
                base_channels=args.geometry_base_channels,
                control_grid=args.control_grid,
                max_translation=args.max_translation,
                max_rotation_deg=args.max_rotation_deg,
                max_local_px=args.max_local_px,
            )
        geometry_model = LearnedPhysicalResidualSolver(4, **geom_cfg).to(device)
        load_checkpoint(
            geometry_model,
            args.geometry_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        geometry_model.eval()
        for parameter in geometry_model.parameters():
            parameter.requires_grad_(False)

    model = build_model(_config(args), info, device)
    if args.stage == "test":
        if not args.diffusion_checkpoint:
            raise ValueError("--stage test requires --diffusion_checkpoint")
        load_checkpoint(
            model,
            args.diffusion_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        metrics = evaluate(
            model,
            geometry_model,
            test_loader,
            base_process=base_process,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            args=args,
            device=device,
        )
        print(
            f"FINAL_REAL_D2 REF_PSNR={metrics['ref_psnr']:.6f} "
            f"REF_SAM={metrics['ref_sam']:.6f} "
            f"PHY_L1={metrics['phy']:.8f} MSI_L1={metrics['msi']:.8f}"
        )
        return

    if not args.init_checkpoint and not args.resume and not args.from_scratch:
        raise ValueError("Real-D2 training requires --init_checkpoint, --resume, or --from_scratch")
    if args.init_checkpoint and args.from_scratch:
        raise ValueError("--from_scratch is incompatible with --init_checkpoint")
    if args.init_checkpoint and not args.resume:
        load_checkpoint(
            model,
            args.init_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ensure_dir(args.checkpoint_root)
    ensure_dir(args.log_root)
    checkpoint = os.path.join(args.checkpoint_root, args.save_name + ".pth")
    last_checkpoint = os.path.join(
        args.checkpoint_root, args.save_name + "_last.pth"
    )
    logger = CSVLogger(
        os.path.join(args.log_root, args.save_name + ".csv"),
        [
            "epoch", "loss", "l1", "sam_rad", "ref_l1", "phy_l1", "msi_l1",
            "val_ref_psnr", "val_ref_sam", "val_phy", "val_msi", "best",
        ],
    )
    start_epoch = 0
    best = float("inf") if args.monitor == "ref_sam" else float("-inf")
    if args.resume:
        start_epoch, stored = load_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
        best = stored
        # Older _last checkpoints were saved before validation. Recover the
        # latest validated best from this run's CSV before training resumes.
        if os.path.isfile(logger.csv_path):
            with open(logger.csv_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not row.get("epoch") or not row.get("best"):
                        continue
                    if int(row["epoch"]) > start_epoch:
                        continue
                    logged_best = float(row["best"])
                    if math.isfinite(logged_best):
                        best = (min(best, logged_best) if args.monitor == "ref_sam"
                                else max(best, logged_best))
        print(f"RESUMED epoch={start_epoch} validated_best_{args.monitor}={best:.6f}")

    reference_frame = (
        "Wald_EnMAP30"
        if msi_source == "real_Sentinel_2_Wald_30m"
        else "metadata_harmonized_EnMAP10"
    )
    print(
        f"AUGSBURG_REAL_D2 msi_source={msi_source} geometry_mode={args.geometry_mode} "
        f"reference_frame={reference_frame} "
        "output_frame=metadata_harmonized_S2_grid "
        f"reference_supervision={'normalized_warp_adjoint' if _geometry_required(args) else 'direct_georeferenced'} "
        f"scale=3 stages={base_process.stages} sigma={sigma:.6f} "
        f"geometry_steps={args.geometry_steps if args.geometry_mode in ('estimated', 'wald_cdrdi') else 0} "
        f"fixed_shift=({args.fixed_dx_px},{args.fixed_dy_px}) wald_augmentation={not is_wald}"
    )
    print(
        "OBSERVED_LR_HSI inverse_warp=False terminal_observation=real_EnMAP30"
    )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr = train_one_epoch(
            model,
            geometry_model,
            train_loader,
            optimizer,
            base_process=base_process,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            args=args,
            device=device,
        )
        print(
            f"epoch={epoch:03d}/{args.epochs} loss={tr['loss']:.7f} "
            f"l1={tr['l1']:.7f} sam={tr['sam']:.7f} ref={tr['ref']:.7f} "
            f"phy={tr['phy']:.7f} msi={tr['msi']:.7f}"
        )
        last_metadata = {
            "stage": "AugsburgReal-D2",
            "monitor": args.monitor,
            "effective_sigma": sigma,
            "scale_ratio": 3,
            "stages": [1, 2, 3],
            "geometry_checkpoint": args.geometry_checkpoint,
            "geometry_mode": args.geometry_mode,
            "msi_source": msi_source,
            "output_frame": "metadata_harmonized_S2_grid",
            "reference_metric_frame": (
                "forward_warp_to_EnMAP10"
                if args.geometry_mode == "estimated"
                else reference_frame
            ),
            "kind": "last",
        }
        save_checkpoint(
            model, optimizer, epoch, best, last_checkpoint,
            extra=last_metadata,
        )
        if epoch % args.eval_interval != 0 and epoch != args.epochs:
            continue

        va = evaluate(
            model,
            geometry_model,
            val_loader,
            base_process=base_process,
            p0=p0,
            srf=srf,
            radiometry=radiometry,
            args=args,
            device=device,
        )
        print(
            f"REAL_D2_VAL REF_PSNR={va['ref_psnr']:.6f} "
            f"REF_SAM={va['ref_sam']:.6f} PHY_L1={va['phy']:.8f} "
            f"MSI_L1={va['msi']:.8f}"
        )
        value = va[args.monitor]
        better = value < best if args.monitor == "ref_sam" else value > best
        if better:
            best = value
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best,
                checkpoint,
                extra={
                    "stage": "AugsburgReal-D2",
                    "monitor": args.monitor,
                    "effective_sigma": sigma,
                    "scale_ratio": 3,
                    "stages": [1, 2, 3],
                    "geometry_checkpoint": args.geometry_checkpoint,
                    "geometry_mode": args.geometry_mode,
                    "msi_source": msi_source,
                    "output_frame": "metadata_harmonized_S2_grid",
                    "reference_metric_frame": (
                        "forward_warp_to_EnMAP10"
                        if args.geometry_mode == "estimated"
                        else reference_frame
                    ),
                },
            )
            print(
                f"SAVED_BEST {checkpoint} {args.monitor}={best:.6f}"
            )
            # Synchronize best_metric in _last after successful validation.
            # Retain pre-validation save above to survive eval failures.
            save_checkpoint(
                model, optimizer, epoch, best, last_checkpoint,
                extra=last_metadata,
            )
        logger.write(
            {
                "epoch": epoch,
                "loss": tr["loss"],
                "l1": tr["l1"],
                "sam_rad": tr["sam"],
                "ref_l1": tr["ref"],
                "phy_l1": tr["phy"],
                "msi_l1": tr["msi"],
                "val_ref_psnr": va["ref_psnr"],
                "val_ref_sam": va["ref_sam"],
                "val_phy": va["phy"],
                "val_msi": va["msi"],
                "best": best,
            }
        )


if __name__ == "__main__":
    main()
