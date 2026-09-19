"""Innovation-3 Stage-C0: held-out capacity diagnostic for physical-residual direction calibration.

Purpose
-------
The current terminal physical residual is observable and causally useful, but its
spectral direction is only weakly aligned with the oracle SAM-descent direction.
Before building another neural refiner, this script tests whether a *small,
global spectral transform* can calibrate that direction on training patches and
generalize to the held-out center test patch.

The baseline final reconstruction is X_hat.  The observable physical direction is

    e       = Y_H - D_T(X_hat)
    r_raw   = P_{T_u} P_{C4:L} U_T(e)

The GT-only diagnostic target on calibration patches is

    d*      = v - <u,v>u
    r_star  = P_{C4:L cap T_u}(d*)

where u is the unit X_hat spectrum and v is the unit GT spectrum.

Only the active C4:L DCT coefficients are calibrated.  The tested mappings are:

    raw          : identity
    diagonal     : independent coefficient scaling
    banded       : local coefficient mixing within +/- band_radius
    lowrank-r    : I + rank-r approximation of the learned full correction
    full_linear  : unconstrained KxK linear map (diagnostic ceiling)

The default fit is direction-only least squares: each per-pixel active input and
oracle target coefficient vector is normalized before accumulating the normal
Equations.  This makes the diagnostic test direction calibration rather than
simply fitting residual magnitude.

Calibration patches come only from the repository training split, which excludes
the center test rectangle.  Data augmentation is disabled for this diagnostic.
GT is used to fit the diagnostic calibrators on the training split and to score
the held-out test patch; GT is never used by the physical residual itself.

For each held-out method the script reports:
  - ALIGN_ALLOWED: cosine with the oracle-allowed correction direction
  - PSNR / SAM after an inference-visible self-calibrated scalar closure step
  - alpha_phys and terminal closure reduction

This is a capacity diagnostic, not yet the final deployable Innovation-3 module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from config import TrainConfig
from data_loader import build_loaders
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from metrics import MetricAverager, calc_metrics
from models import (
    HeterogeneityGuidedSpectralRefiner,
    RawMSIDirectPredictor,
    load_legacy_raw_direct_checkpoint,
)
from utils import ensure_dir, get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 held-out physical-residual direction-calibration capacity diagnostic"
    )
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)

    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)

    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--baseline_checkpoint",
        default="./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth",
    )
    p.add_argument(
        "--baseline_checkpoint_type",
        choices=["legacy", "standard"],
        default="legacy",
    )

    p.add_argument(
        "--calib_patches",
        type=int,
        default=64,
        help="Maximum number of non-overlapping-with-test training patches used to fit the diagnostic maps; <=0 uses all.",
    )
    p.add_argument(
        "--fit_mode",
        choices=["direction", "raw"],
        default="direction",
        help="direction normalizes per-pixel active coefficient vectors before least-squares fitting.",
    )
    p.add_argument("--band_radius", type=int, default=1)
    p.add_argument(
        "--lowrank_ranks",
        default="2,4",
        help="Comma-separated ranks for I+low-rank calibration.",
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-4,
        help="Relative ridge multiplier; scaled by mean diagonal energy of each normal matrix.",
    )
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument(
        "--fixed_alpha_reference",
        type=float,
        default=4.0,
        help="Optional prior GT-sweep reference printed only for raw physical residual.",
    )
    p.add_argument(
        "--output_json",
        default="./results/innovation3_direction_calibration_capacity_PaviaU.json",
    )
    return p.parse_args()


def _parse_ranks(text: str) -> Tuple[int, ...]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            value = int(token)
            if value < 1:
                raise ValueError("low-rank values must be >=1")
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError("--lowrank_ranks is empty")
    return tuple(values)


def _config(args) -> TrainConfig:
    return TrainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        test_size=args.test_size,
        scale_ratio=args.scale_ratio,
        srf_interp=args.srf_interp,
        degradation_mode="physical",
        diffusion_steps=args.diffusion_steps,
        lift_mode="normalized_adjoint",
        mtf_nyquist=args.mtf_nyquist,
        psf_truncate=args.psf_truncate,
        predictor="raw_direct",
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        spectral_hidden=args.spectral_hidden,
        dropout=args.dropout,
        batch_size=1,
        num_workers=0,
        seed=args.seed,
        device=args.device,
    )


def _build_baseline(args, info, device):
    model = RawMSIDirectPredictor(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_msi_bands"],
        total_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        residual_prediction=True,
        spectral_hidden=args.spectral_hidden,
    ).to(device)

    if args.baseline_checkpoint_type == "legacy":
        report = load_legacy_raw_direct_checkpoint(
            model,
            args.baseline_checkpoint,
            map_location=str(device),
        )
        print("BASELINE_LOAD legacy", report)
    else:
        epoch, best = load_checkpoint(
            model,
            args.baseline_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        print(f"BASELINE_LOAD standard epoch={epoch} best={best}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _ideal_allowed(refiner, base: torch.Tensor, gt: torch.Tensor, eps: float):
    base_f = base.float()
    gt_f = gt.float()
    u = F.normalize(base_f, dim=1, eps=eps)
    v = F.normalize(gt_f, dim=1, eps=eps)
    uv = (u * v).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    ideal = v - uv * u
    broad = refiner.project_broadshape(ideal)
    allowed = refiner.project_tangent(
        broad,
        base_f,
        preserve_broadshape=True,
    )
    return ideal, allowed


def _physical_raw(process, refiner, terminal_lr, base):
    pred_terminal = process.terminal_observation(base)
    e = terminal_lr - pred_terminal
    lift = process.terminal_state(e, target_size=tuple(base.shape[-2:]))
    broad = refiner.project_broadshape(lift)
    raw = refiner.project_tangent(
        broad,
        base,
        preserve_broadshape=True,
    )
    return e, raw


def _active_coeff(refiner, x: torch.Tensor) -> torch.Tensor:
    coeff = refiner.dct(x)
    return coeff[:, 4:refiner.low_end]


def _from_active_coeff(refiner, coeff_active: torch.Tensor) -> torch.Tensor:
    b, k, h, w = coeff_active.shape
    full = coeff_active.new_zeros((b, refiner.n_bands, h, w))
    full[:, 4:refiner.low_end] = coeff_active
    return refiner.idct(full)


def _samples(coeff: torch.Tensor, *, interior: bool = True) -> torch.Tensor:
    if interior:
        coeff = coeff[:, :, 1:-1, 1:-1]
    return coeff.permute(0, 2, 3, 1).reshape(-1, coeff.shape[1]).float()


def _normalize_rows(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    valid = norm.squeeze(1) > eps
    out = x / norm.clamp_min(eps)
    return out, valid


def _prepare_fit_samples(
    refiner,
    raw: torch.Tensor,
    allowed: torch.Tensor,
    *,
    fit_mode: str,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = _samples(_active_coeff(refiner, raw), interior=True)
    y = _samples(_active_coeff(refiner, allowed), interior=True)

    x_valid = torch.isfinite(x).all(dim=1)
    y_valid = torch.isfinite(y).all(dim=1)
    valid = x_valid & y_valid

    if fit_mode == "direction":
        x, vx = _normalize_rows(x, eps)
        y, vy = _normalize_rows(y, eps)
        valid = valid & vx & vy

    return x[valid], y[valid]


class NormalEquationAccumulator:
    def __init__(self, k: int, device):
        self.xtx = torch.zeros((k, k), dtype=torch.float64, device=device)
        self.xty = torch.zeros((k, k), dtype=torch.float64, device=device)
        self.n = 0

    def update(self, x: torch.Tensor, y: torch.Tensor):
        if x.numel() == 0:
            return
        xd = x.double()
        yd = y.double()
        self.xtx += xd.T @ xd
        self.xty += xd.T @ yd
        self.n += int(x.shape[0])


def _ridge_scale(xtx: torch.Tensor, ridge: float, eps: float) -> float:
    mean_diag = float(torch.diagonal(xtx).mean().item())
    return max(float(ridge) * max(mean_diag, eps), eps)


def _solve_full(acc: NormalEquationAccumulator, ridge: float, eps: float) -> torch.Tensor:
    k = acc.xtx.shape[0]
    lam = _ridge_scale(acc.xtx, ridge, eps)
    reg = acc.xtx + lam * torch.eye(k, dtype=acc.xtx.dtype, device=acc.xtx.device)
    return torch.linalg.solve(reg, acc.xty).float()


def _solve_diagonal(acc: NormalEquationAccumulator, ridge: float, eps: float) -> torch.Tensor:
    k = acc.xtx.shape[0]
    out = torch.zeros((k, k), dtype=torch.float64, device=acc.xtx.device)
    for j in range(k):
        xx = acc.xtx[j, j]
        xy = acc.xty[j, j]
        lam = max(float(ridge) * float(xx.item()), eps)
        out[j, j] = xy / (xx + lam)
    return out.float()


def _solve_banded(
    acc: NormalEquationAccumulator,
    radius: int,
    ridge: float,
    eps: float,
) -> torch.Tensor:
    if radius < 0:
        raise ValueError("band_radius must be >=0")
    k = acc.xtx.shape[0]
    out = torch.zeros((k, k), dtype=torch.float64, device=acc.xtx.device)
    # Row-vector convention: Y_hat = X @ B.  For output j solve using input
    # indices i in [j-radius, j+radius].
    for j in range(k):
        lo = max(0, j - radius)
        hi = min(k, j + radius + 1)
        idx = torch.arange(lo, hi, device=acc.xtx.device)
        local_xx = acc.xtx.index_select(0, idx).index_select(1, idx)
        local_xy = acc.xty.index_select(0, idx)[:, j]
        lam = _ridge_scale(local_xx, ridge, eps)
        reg = local_xx + lam * torch.eye(
            len(idx), dtype=local_xx.dtype, device=local_xx.device
        )
        weights = torch.linalg.solve(reg, local_xy)
        out[idx, j] = weights
    return out.float()


def _lowrank_identity(full_b: torch.Tensor, rank: int) -> torch.Tensor:
    k = full_b.shape[0]
    identity = torch.eye(k, dtype=full_b.dtype, device=full_b.device)
    delta = full_b - identity
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    r = min(int(rank), k)
    approx = (u[:, :r] * s[:r].unsqueeze(0)) @ vh[:r]
    return identity + approx


def _apply_map(
    refiner,
    base: torch.Tensor,
    raw: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    active = _active_coeff(refiner, raw)
    # BxKxHxW -> BxHxWxK -> right-multiply the row-vector coefficient map.
    bhwk = active.permute(0, 2, 3, 1)
    mapped = torch.einsum("bhwk,kl->bhwl", bhwk, matrix.to(bhwk.dtype))
    mapped = mapped.permute(0, 3, 1, 2).contiguous()
    broad = _from_active_coeff(refiner, mapped)
    return refiner.project_tangent(
        broad,
        base,
        preserve_broadshape=True,
    )


def _safe_alignment(a: torch.Tensor, b: torch.Tensor, eps: float) -> Tuple[float, float]:
    aa = _samples(a, interior=True)
    bb = _samples(b, interior=True)
    an = torch.linalg.vector_norm(aa, dim=1)
    bn = torch.linalg.vector_norm(bb, dim=1)
    valid = (
        torch.isfinite(aa).all(dim=1)
        & torch.isfinite(bb).all(dim=1)
        & (an > eps)
        & (bn > eps)
    )
    if not bool(valid.any()):
        return float("nan"), float("nan")
    cos = (aa[valid] * bb[valid]).sum(dim=1) / (an[valid] * bn[valid]).clamp_min(eps)
    return float(cos.mean().item()), float((cos > 0).float().mean().item())


def _selfcal_alpha(process, e: torch.Tensor, r: torch.Tensor, eps: float) -> Tuple[float, torch.Tensor]:
    q = process.terminal_observation(r)
    numerator = (e.float() * q.float()).sum()
    denominator = q.float().square().sum().clamp_min(eps)
    alpha = (numerator / denominator).clamp_min(0.0)
    return float(alpha.item()), q


def _closure_reduction(e: torch.Tensor, q: torch.Tensor, alpha: float, eps: float) -> float:
    before = torch.linalg.vector_norm(e.float())
    after = torch.linalg.vector_norm(e.float() - float(alpha) * q.float())
    return float((1.0 - after / before.clamp_min(eps)).item())


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


@torch.no_grad()
def main():
    args = parse_args()
    ranks = _parse_ranks(args.lowrank_ranks)
    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)

    train_loader, test_loader, info = build_loaders(cfg)
    # This is a diagnostic fit, not normal training augmentation.  Keeping the
    # raw training patches makes the train/test spatial split explicit.
    if hasattr(train_loader.dataset, "augment"):
        train_loader.dataset.augment = False

    process = build_progressive_process(cfg)
    baseline = _build_baseline(args, info, device)
    refiner = HeterogeneityGuidedSpectralRefiner(
        n_bands=info["n_bands"],
        total_steps=args.diffusion_steps,
        hidden_channels=1,
        variant="full",
        eps=args.eps,
    ).to(device)
    refiner.eval()
    for parameter in refiner.parameters():
        parameter.requires_grad_(False)

    k = refiner.low_end - 4
    if k < 1:
        raise RuntimeError("empty C4:L active interval")
    acc = NormalEquationAccumulator(k, device)

    print(
        "DIAGNOSTIC innovation3_stage_C0 direction_calibration_capacity "
        f"dataset={args.dataset} C4L=[4,{refiner.low_end-1}] K={k} "
        f"fit_mode={args.fit_mode} calib_patches={args.calib_patches}"
    )
    print(
        "CALIBRATION_SPLIT training patches exclude center test rectangle; "
        "augmentation=off; GT used only to fit diagnostic mapping."
    )

    used_patches = 0
    for batch_index, batch in enumerate(train_loader):
        if args.calib_patches > 0 and used_patches >= args.calib_patches:
            break

        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        terminal_lr = process.terminal_observation(gt)
        base = reconstruct_from_terminal_lr(
            baseline,
            process,
            terminal_lr,
            target_size=tuple(gt.shape[-2:]),
            hr_msi=hr_msi,
        )
        _, raw = _physical_raw(process, refiner, terminal_lr, base)
        _, allowed = _ideal_allowed(refiner, base, gt, args.eps)
        x, y = _prepare_fit_samples(
            refiner,
            raw,
            allowed,
            fit_mode=args.fit_mode,
            eps=args.eps,
        )
        acc.update(x, y)
        used_patches += 1
        if used_patches == 1 or used_patches % 8 == 0:
            print(
                f"CALIB patch={used_patches} valid_pixels_total={acc.n}"
            )

    if acc.n < max(100, 10 * k):
        raise RuntimeError(f"too few valid calibration pixels: {acc.n}")

    diagonal = _solve_diagonal(acc, args.ridge, args.eps)
    banded = _solve_banded(acc, args.band_radius, args.ridge, args.eps)
    full = _solve_full(acc, args.ridge, args.eps)

    matrices: Dict[str, torch.Tensor] = {
        "raw": torch.eye(k, dtype=torch.float32, device=device),
        "diagonal": diagonal.to(device),
        f"banded_r{args.band_radius}": banded.to(device),
    }
    for rank in ranks:
        matrices[f"lowrank_r{rank}"] = _lowrank_identity(full.to(device), rank)
    matrices["full_linear"] = full.to(device)

    print("=" * 138)
    print(
        f"CALIB_FIT patches={used_patches} pixels={acc.n} ridge={args.ridge} "
        f"band_radius={args.band_radius} ranks={ranks}"
    )
    for name, matrix in matrices.items():
        identity = torch.eye(k, device=matrix.device, dtype=matrix.dtype)
        print(
            f"MATRIX {name:<14} fro={torch.linalg.matrix_norm(matrix).item():.6f} "
            f"delta_identity={torch.linalg.matrix_norm(matrix-identity).item():.6f}"
        )

    # Held-out center test patch evaluation.
    baseline_meter = MetricAverager()
    method_meters = {name: MetricAverager() for name in matrices}
    fixed_raw_meter = MetricAverager()
    method_stats = {
        name: {"ALIGN_ALLOWED": [], "ALIGN_POS_FRAC": [], "ALPHA_PHYS": [], "CLOSURE_REDUCTION": []}
        for name in matrices
    }

    for test_index, batch in enumerate(test_loader):
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        terminal_lr = process.terminal_observation(gt)
        base = reconstruct_from_terminal_lr(
            baseline,
            process,
            terminal_lr,
            target_size=tuple(gt.shape[-2:]),
            hr_msi=hr_msi,
        )
        e, raw = _physical_raw(process, refiner, terminal_lr, base)
        _, allowed = _ideal_allowed(refiner, base, gt, args.eps)

        baseline_meter.update(calc_metrics(base, gt, args.scale_ratio))
        fixed_raw_meter.update(
            calc_metrics(base + float(args.fixed_alpha_reference) * raw, gt, args.scale_ratio)
        )

        for name, matrix in matrices.items():
            calibrated = _apply_map(refiner, base, raw, matrix)
            align, pos_frac = _safe_alignment(calibrated, allowed, args.eps)
            alpha, q = _selfcal_alpha(process, e, calibrated, args.eps)
            pred = base + alpha * calibrated

            method_meters[name].update(calc_metrics(pred, gt, args.scale_ratio))
            method_stats[name]["ALIGN_ALLOWED"].append(align)
            method_stats[name]["ALIGN_POS_FRAC"].append(pos_frac)
            method_stats[name]["ALPHA_PHYS"].append(alpha)
            method_stats[name]["CLOSURE_REDUCTION"].append(
                _closure_reduction(e, q, alpha, args.eps)
            )

        print(f"TEST case={test_index} done")

    base_metrics = baseline_meter.average()
    fixed_raw = fixed_raw_meter.average()

    print("=" * 138)
    print(
        "METHOD              PSNR          SAM      dPSNR     dSAM    "
        "ALIGN_ALLOWED   POS_FRAC   ALPHA_PHYS   CLOSURE_RED"
    )

    rows = {}
    for name in matrices:
        metrics = method_meters[name].average()
        row = {
            **metrics,
            "dPSNR": metrics["PSNR"] - base_metrics["PSNR"],
            "dSAM": metrics["SAM"] - base_metrics["SAM"],
            "ALIGN_ALLOWED": _mean(method_stats[name]["ALIGN_ALLOWED"]),
            "ALIGN_POS_FRAC": _mean(method_stats[name]["ALIGN_POS_FRAC"]),
            "ALPHA_PHYS": _mean(method_stats[name]["ALPHA_PHYS"]),
            "CLOSURE_REDUCTION": _mean(method_stats[name]["CLOSURE_REDUCTION"]),
        }
        rows[name] = row
        print(
            f"{name:<18} "
            f"{metrics['PSNR']:11.6f} {metrics['SAM']:12.6f} "
            f"{row['dPSNR']:+10.6f} {row['dSAM']:+9.6f} "
            f"{row['ALIGN_ALLOWED']:+13.6f} {row['ALIGN_POS_FRAC']:10.6f} "
            f"{row['ALPHA_PHYS']:12.6f} {100.0*row['CLOSURE_REDUCTION']:11.4f}%"
        )

    best_align = max(rows.items(), key=lambda item: item[1]["ALIGN_ALLOWED"])
    best_sam = min(rows.items(), key=lambda item: item[1]["SAM"])

    print("=" * 138)
    print(
        "BASELINE "
        f"PSNR={base_metrics['PSNR']:.6f} SAM={base_metrics['SAM']:.6f}"
    )
    print(
        "FIXED_RAW_REFERENCE "
        f"alpha={args.fixed_alpha_reference:.6f} "
        f"PSNR={fixed_raw['PSNR']:.6f} SAM={fixed_raw['SAM']:.6f} "
        f"dPSNR={fixed_raw['PSNR']-base_metrics['PSNR']:+.6f} "
        f"dSAM={fixed_raw['SAM']-base_metrics['SAM']:+.6f}"
    )
    print(
        "BEST_ALIGNMENT "
        f"method={best_align[0]} ALIGN_ALLOWED={best_align[1]['ALIGN_ALLOWED']:+.6f} "
        f"SAM={best_align[1]['SAM']:.6f} dSAM={best_align[1]['dSAM']:+.6f}"
    )
    print(
        "BEST_HELDOUT_SAM "
        f"method={best_sam[0]} PSNR={best_sam[1]['PSNR']:.6f} "
        f"SAM={best_sam[1]['SAM']:.6f} dPSNR={best_sam[1]['dPSNR']:+.6f} "
        f"dSAM={best_sam[1]['dSAM']:+.6f} "
        f"ALIGN_ALLOWED={best_sam[1]['ALIGN_ALLOWED']:+.6f}"
    )

    payload = {
        "conditions": {
            "stage": "Innovation-3 Stage-C0 direction-calibration capacity diagnostic",
            "dataset": args.dataset,
            "baseline_checkpoint": args.baseline_checkpoint,
            "baseline_checkpoint_type": args.baseline_checkpoint_type,
            "calibration_split": "training patches excluding center test rectangle",
            "calib_patches": used_patches,
            "calib_valid_pixels": acc.n,
            "augmentation": "disabled",
            "fit_mode": args.fit_mode,
            "band_radius": args.band_radius,
            "lowrank_ranks": list(ranks),
            "ridge": args.ridge,
            "c4l_range": [4, refiner.low_end - 1],
            "active_dim": k,
            "test_size": args.test_size,
            "scale_ratio": args.scale_ratio,
            "seed": args.seed,
            "gt_usage": "calibration-target fitting on training split and held-out diagnostic scoring only",
        },
        "definitions": {
            "raw_physical": "P_Tu P_C4L U_T[Y_H-D_T(X_hat)]",
            "oracle_allowed": "P_{C4L intersect T_u}[v-<u,v>u]",
            "diagonal": "independent active-DCT coefficient scaling",
            "banded": "local active-DCT coefficient mixing",
            "lowrank": "I + truncated-SVD low-rank approximation of the full learned correction",
            "full_linear": "unconstrained active-DCT linear map; diagnostic ceiling",
            "inference_gain": "nonnegative observable closure alpha=<e,D_T(r)>/||D_T(r)||^2",
        },
        "baseline": base_metrics,
        "fixed_raw_reference": {
            **fixed_raw,
            "alpha": args.fixed_alpha_reference,
        },
        "rows": rows,
        "matrices": {
            name: matrix.detach().cpu().tolist() for name, matrix in matrices.items()
        },
        "best_alignment": {"method": best_align[0], **best_align[1]},
        "best_heldout_sam": {"method": best_sam[0], **best_sam[1]},
    }

    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
