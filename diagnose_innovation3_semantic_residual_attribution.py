"""Innovation-3 Stage-C1: oracle semantic/material spectral-residual attribution.

Question: does the terminal LR-HSI residual contain useful spectral error but lose
its HR pixel/material ownership after PSF + detector integration + downsampling?

PaviaU label 0 is always treated as UNLABELED. Class prototypes are estimated
from labeled pixels outside the held-out center test rectangle; the center test
rectangle never contributes to prototypes. If a test class has no labeled
prototype support outside the test rectangle, that class is reported and
excluded from class-conditioned comparisons rather than leaking test spectra.
Test labels are oracle semantic information. Test GT spectra are used only for
scoring and for the explicit C1-5 oracle-direction ceiling.

Groups:
  C1-0 raw physical residual + observable global self-calibrated gain.
  C1-1 local prototype direction + LR physical magnitude inversion.
  C1-2 oracle class prototype direction + one observable closure gain.
  C1-3 oracle class direction + class-specific LR magnitude fields (core).
  C1-4 same masks/DOF as C1-3 but cyclically wrong class-prototype identity.
  C1-5 unit GT oracle-allowed direction + LR physical magnitude inversion.

C1-1..C1-5 optimize closure only on the forward projection of labeled support,
so sparse PaviaU labels do not make unlabeled pixels masquerade as a material.
This is a capacity diagnostic, not a deployable method.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import build_loaders
from diagnose_innovation3_direction_calibration_capacity import (
    _build_baseline,
    _config,
    _ideal_allowed,
    _physical_raw,
)
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from metrics import calc_metrics
from models import HeterogeneityGuidedSpectralRefiner, ranked_msi_heterogeneity
from utils import ensure_dir, get_device, set_seed

try:
    import scipy.io as scio
except ImportError:
    scio = None
try:
    import h5py
except ImportError:
    h5py = None


def parse_args():
    p = argparse.ArgumentParser(description="Innovation-3 Stage-C1 semantic residual attribution")
    p.add_argument("--dataset", choices=["PaviaU"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--label_file", default="./data/raw/PaviaU_gt.mat")
    p.add_argument("--label_key", default="paviaU_gt")
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
    p.add_argument("--baseline_checkpoint_type", choices=["legacy", "standard"], default="legacy")
    p.add_argument("--local_kernel", type=int, default=5)
    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument("--min_test_label_coverage", type=float, default=0.05)
    p.add_argument("--fail_on_low_coverage", action="store_true")
    p.add_argument("--solver_steps", type=int, default=400)
    p.add_argument("--solver_lr", type=float, default=5e-2)
    p.add_argument("--lambda_amp", type=float, default=1e-4)
    p.add_argument("--lambda_smooth", type=float, default=1e-4)
    p.add_argument("--amp_max", type=float, default=1.0)
    p.add_argument("--solver_log_interval", type=int, default=100)
    p.add_argument("--solver_patience", type=int, default=80)
    p.add_argument("--solver_tol", type=float, default=1e-7)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument(
        "--output_json",
        default="./results/innovation3_semantic_residual_attribution_PaviaU.json",
    )
    return p.parse_args()


def _load_labels(path: str, key: str, hw: Tuple[int, int]) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing PaviaU label file: {path}")
    arrays = []
    errors = []
    if scio is not None:
        try:
            m = scio.loadmat(path)
            keys = [key] + [k for k in m if not k.startswith("__") and k != key]
            arrays.extend(
                np.squeeze(m[k]) for k in keys
                if k in m
                and isinstance(m[k], np.ndarray)
                and np.squeeze(m[k]).ndim == 2
                and np.issubdtype(np.squeeze(m[k]).dtype, np.number)
            )
        except Exception as exc:
            errors.append(f"scipy.io: {exc}")
    if not arrays and h5py is not None:
        try:
            with h5py.File(path, "r") as f:
                if key in f:
                    arrays.append(np.squeeze(np.asarray(f[key])))
                else:
                    def visit(_name, obj):
                        if not arrays and isinstance(obj, h5py.Dataset):
                            a = np.squeeze(np.asarray(obj))
                            if a.ndim == 2:
                                arrays.append(a)
                    f.visititems(visit)
        except Exception as exc:
            errors.append(f"h5py: {exc}")
    h, w = hw
    for arr in arrays:
        for a in (np.asarray(arr), np.asarray(arr).T):
            if a.shape[0] < h or a.shape[1] < w:
                continue
            a = a[:h, :w]
            if not np.all(np.isfinite(a)):
                continue
            rounded = np.rint(a)
            if np.max(np.abs(a.astype(np.float64) - rounded)) <= 1e-4 and rounded.min() >= 0:
                return rounded.astype(np.int64)
    raise RuntimeError(f"Unable to align 2-D label map to HSI shape {hw}. " + " | ".join(errors))


def _outside_test_support(
    hw: Tuple[int, int],
    top: int,
    left: int,
    test_size: int,
) -> np.ndarray:
    """All pixels outside the held-out center test rectangle.

    This is intentionally broader than the union of sampled training patches:
    semantic prototypes are a diagnostic prior, and any labeled pixel outside
    the held-out rectangle is admissible without test leakage.
    """
    out = np.ones(hw, dtype=bool)
    bottom = min(top + int(test_size), hw[0])
    right = min(left + int(test_size), hw[1])
    out[top:bottom, left:right] = False
    return out


def _prototypes(image, labels, support, classes):
    proto, counts, missing = {}, {}, []
    for c in classes:
        m = support & (labels == c)
        counts[c] = int(m.sum())
        if counts[c] == 0:
            missing.append(c)
            continue
        proto[c] = image[m].mean(axis=0).astype(np.float32)
    return proto, counts, missing


def _target_from_prototypes(base, labels, prototypes, mapping=None):
    out = base.detach().clone().float()
    lab = labels[0]
    for c in sorted(prototypes):
        source = mapping[c] if mapping is not None else c
        p = torch.as_tensor(prototypes[source], device=out.device, dtype=out.dtype).view(-1, 1)
        m = lab == c
        if bool(m.any()):
            out[0, :, m] = p
    return out


def _class_masks(labels, classes):
    return torch.stack([(labels == c).float() for c in classes], dim=1)


def _perm_mapping(classes):
    if len(classes) < 2:
        raise RuntimeError("C1-4 requires >=2 labeled classes in the test patch")
    return {c: classes[(i + 1) % len(classes)] for i, c in enumerate(classes)}


def _unit_direction(x, support, eps):
    n = torch.linalg.vector_norm(x.float(), dim=1, keepdim=True)
    out = torch.where(n > eps, x.float() / n.clamp_min(eps), torch.zeros_like(x.float()))
    return out if support is None else out * support.float().unsqueeze(1)


def _project_target(projector, base, target, support, eps):
    broad = projector.project_broadshape(target.float() - base.float())
    tangent = projector.project_tangent(broad, base.float(), preserve_broadshape=True)
    return _unit_direction(tangent, support, eps)


def _local_target(base, kernel):
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError("--local_kernel must be odd and >=3")
    r = kernel // 2
    return F.avg_pool2d(F.pad(base.float(), (r, r, r, r), mode="reflect"), kernel, stride=1)


def _weighted_mse(x, weight, eps):
    sq = x.float().square()
    if weight is None:
        return sq.mean()
    return (sq * weight.float()).sum() / (weight.sum().clamp_min(eps) * x.shape[1])


def _selfcal(process, e, direction, eps, weight=None):
    q = process.terminal_observation(direction)
    if weight is None:
        num = (e.float() * q.float()).sum()
        den = q.float().square().sum()
    else:
        num = (e.float() * q.float() * weight.float()).sum()
        den = (q.float().square() * weight.float()).sum()
    alpha = (num / den.clamp_min(eps)).clamp_min(0.0)
    return float(alpha.item()), q


def _closure_reduction(e, q, weight, eps):
    before = torch.sqrt(_weighted_mse(e, weight, eps).clamp_min(eps))
    after = torch.sqrt(_weighted_mse(e - q, weight, eps).clamp_min(eps))
    return float((1.0 - after / before).item())


def _pixel_sam(pred, gt, eps):
    dot = (pred.float() * gt.float()).sum(1)
    pn = torch.linalg.vector_norm(pred.float(), dim=1)
    gn = torch.linalg.vector_norm(gt.float(), dim=1)
    cos = dot / (pn * gn).clamp_min(eps)
    return torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)) * (180.0 / math.pi)


def _semantic_sam(pred, gt, support, risk, fraction, eps):
    sam = _pixel_sam(pred, gt, eps)[:, 1:-1, 1:-1].reshape(-1)
    sup = support[:, 1:-1, 1:-1].reshape(-1).bool()
    r = risk[:, 1:-1, 1:-1].reshape(-1).float()
    v = sup & torch.isfinite(sam) & torch.isfinite(r)
    if not bool(v.any()):
        return {"SAM_LABELED": float("nan"), "SAM_HIGH_LABELED": float("nan"), "SAM_LOW_LABELED": float("nan"), "LABELED_N": 0}
    s, rv = sam[v], r[v]
    lo, hi = torch.quantile(rv, fraction), torch.quantile(rv, 1.0 - fraction)
    return {
        "SAM_LABELED": float(s.mean().item()),
        "SAM_HIGH_LABELED": float(s[rv >= hi].mean().item()),
        "SAM_LOW_LABELED": float(s[rv <= lo].mean().item()),
        "LABELED_N": int(s.numel()),
    }


def _alignment(update, allowed, support, eps):
    u = update[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1).reshape(-1, update.shape[1]).float()
    a = allowed[:, :, 1:-1, 1:-1].permute(0, 2, 3, 1).reshape(-1, allowed.shape[1]).float()
    s = support[:, 1:-1, 1:-1].reshape(-1).bool()
    un, an = torch.linalg.vector_norm(u, dim=1), torch.linalg.vector_norm(a, dim=1)
    v = s & torch.isfinite(u).all(1) & torch.isfinite(a).all(1) & (un > eps) & (an > eps)
    if not bool(v.any()):
        return float("nan"), float("nan")
    cos = (u[v] * a[v]).sum(1) / (un[v] * an[v]).clamp_min(eps)
    return float(cos.mean().item()), float((cos > 0).float().mean().item())


def _tv(x):
    terms = []
    if x.shape[-2] > 1:
        terms.append((x[:, :, 1:] - x[:, :, :-1]).square().mean())
    if x.shape[-1] > 1:
        terms.append((x[:, :, :, 1:] - x[:, :, :, :-1]).square().mean())
    return sum(terms) / max(len(terms), 1) if terms else x.new_zeros(())


def _logit(x):
    x = min(max(float(x), 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


def _solve_magnitude(process, e, direction, weight, args, name, class_masks=None):
    b, _, h, w = direction.shape
    g = class_masks.shape[1] if class_masks is not None else 1
    lh, lw = e.shape[-2:]
    alpha0, _ = _selfcal(process, e, direction, args.eps, weight)
    if args.amp_max > 0:
        init = min(max(alpha0, args.amp_max * 1e-2), args.amp_max * 0.95)
        raw = torch.nn.Parameter(torch.full((b, g, lh, lw), _logit(init / args.amp_max), device=direction.device))
    else:
        init = max(alpha0, 1e-2)
        raw = torch.nn.Parameter(torch.full((b, g, lh, lw), math.log(math.expm1(init)), device=direction.device))
    opt = torch.optim.Adam([raw], lr=args.solver_lr)
    base_mse = _weighted_mse(e.detach(), weight, args.eps).detach().clamp_min(args.eps)
    if args.solver_steps < 1 or args.solver_lr <= 0:
        raise ValueError("solver_steps must be >=1 and solver_lr must be >0")
    best_loss, best_amp, best_iter, stale = float("inf"), None, 0, 0

    for i in range(1, args.solver_steps + 1):
        amp_lr = args.amp_max * torch.sigmoid(raw) if args.amp_max > 0 else F.softplus(raw)
        amp_groups = process.terminal_state(amp_lr, target_size=(h, w))
        amp_hr = amp_groups if class_masks is None else (amp_groups * class_masks).sum(1, keepdim=True)
        update = direction * amp_hr
        q = process.terminal_observation(update)
        closure = _weighted_mse(e - q, weight, args.eps) / base_mse
        loss = closure + args.lambda_amp * amp_lr.square().mean() + args.lambda_smooth * _tv(amp_lr)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: non-finite solver loss at iter={i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss.detach().item())
        if lv < best_loss - args.solver_tol:
            best_loss, best_amp, best_iter, stale = lv, amp_lr.detach().clone(), i, 0
        else:
            stale += 1
        if i == 1 or i == args.solver_steps or (args.solver_log_interval > 0 and i % args.solver_log_interval == 0):
            print(f"SOLVER {name:<27} iter={i:04d}/{args.solver_steps} loss={lv:.8f} closure_rel={float(closure.detach().item()):.8f} amp_mean={float(amp_lr.detach().mean().item()):.8f} amp_max={float(amp_lr.detach().max().item()):.8f}")
        if args.solver_patience > 0 and stale >= args.solver_patience:
            print(f"SOLVER {name:<27} early_stop iter={i} best_iter={best_iter} best_loss={best_loss:.8f}")
            break

    if best_amp is None:
        raise RuntimeError(f"{name}: solver did not produce a valid amplitude field")
    with torch.no_grad():
        amp_groups = process.terminal_state(best_amp, target_size=(h, w))
        amp_hr = amp_groups if class_masks is None else (amp_groups * class_masks).sum(1, keepdim=True)
        update = direction * amp_hr
        q = process.terminal_observation(update)
        stats = {
            "SOLVER_BEST_ITER": best_iter,
            "SOLVER_BEST_LOSS": best_loss,
            "AMP_LR_MEAN": float(best_amp.mean().item()),
            "AMP_LR_STD": float(best_amp.std(unbiased=False).item()),
            "AMP_LR_MAX": float(best_amp.max().item()),
            "AMP_SAT_FRAC": float((best_amp >= 0.99 * args.amp_max).float().mean().item()) if args.amp_max > 0 else 0.0,
            "CLOSURE_REDUCTION_WEIGHTED": _closure_reduction(e, q, weight, args.eps),
            "CLOSURE_REDUCTION_GLOBAL": _closure_reduction(e, q, None, args.eps),
        }
    return update.detach(), stats


def _row(pred, base, gt, update, allowed, label_mask, risk, args, extra=None):
    base_m = calc_metrics(base, gt, args.scale_ratio)
    m = calc_metrics(pred, gt, args.scale_ratio)
    bs = _semantic_sam(base, gt, label_mask, risk, args.region_fraction, args.eps)
    s = _semantic_sam(pred, gt, label_mask, risk, args.region_fraction, args.eps)
    align, pos = _alignment(update, allowed, label_mask, args.eps)
    out = {
        **m, **s,
        "dPSNR": m["PSNR"] - base_m["PSNR"],
        "dSAM": m["SAM"] - base_m["SAM"],
        "dSAM_LABELED": s["SAM_LABELED"] - bs["SAM_LABELED"],
        "dSAM_HIGH_LABELED": s["SAM_HIGH_LABELED"] - bs["SAM_HIGH_LABELED"],
        "dSAM_LOW_LABELED": s["SAM_LOW_LABELED"] - bs["SAM_LOW_LABELED"],
        "ALIGN_ALLOWED_LABELED": align,
        "ALIGN_POS_FRAC_LABELED": pos,
    }
    if extra:
        out.update(extra)
    return out


def _print_row(name, r):
    print(f"{name:<30} PSNR={r['PSNR']:.6f} SAM={r['SAM']:.6f} SAM_L={r['SAM_LABELED']:.6f} dSAM_L={r['dSAM_LABELED']:+.6f} SAM_H={r['SAM_HIGH_LABELED']:.6f} dH={r['dSAM_HIGH_LABELED']:+.6f} SAM_LO={r['SAM_LOW_LABELED']:.6f} dLO={r['dSAM_LOW_LABELED']:+.6f} ALIGN_L={r['ALIGN_ALLOWED_LABELED']:+.6f} POS={r['ALIGN_POS_FRAC_LABELED']:.6f}")


def main():
    args = parse_args()
    if not 0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    train_loader, test_loader, info = build_loaders(cfg)
    process = build_progressive_process(cfg)
    baseline = _build_baseline(args, info, device)
    projector = HeterogeneityGuidedSpectralRefiner(
        n_bands=info["n_bands"], total_steps=args.diffusion_steps,
        hidden_channels=1, variant="full", eps=args.eps,
    ).to(device).eval()
    for p in projector.parameters():
        p.requires_grad_(False)

    image = np.asarray(train_loader.dataset.image, dtype=np.float32)
    labels = _load_labels(args.label_file, args.label_key, image.shape[:2])
    if len(test_loader.dataset.coords) != 1:
        raise RuntimeError("Stage-C1 expects the repository single center test patch")
    top, left = test_loader.dataset.coords[0]
    labels_np = labels[top:top + args.test_size, left:left + args.test_size]
    if labels_np.shape != (args.test_size, args.test_size):
        raise RuntimeError(f"test label crop shape={labels_np.shape}, expected={(args.test_size, args.test_size)}")

    all_test_classes = sorted(int(c) for c in np.unique(labels_np) if c > 0)
    coverage = float((labels_np > 0).mean())
    if not all_test_classes:
        raise RuntimeError("center test patch has no labeled pixels")
    if coverage < args.min_test_label_coverage:
        msg = f"label coverage {coverage:.6f} < {args.min_test_label_coverage:.6f}; conclusions apply only to sparse labeled support"
        if args.fail_on_low_coverage:
            raise RuntimeError(msg)
        print("WARNING", msg)

    prototype_support = _outside_test_support(
        image.shape[:2], top, left, args.test_size
    )
    prototypes, prototype_counts, missing_classes = _prototypes(
        image, labels, prototype_support, all_test_classes
    )
    classes = [c for c in all_test_classes if c in prototypes]
    if missing_classes:
        print(
            "WARNING UNSUPPORTED_TEST_CLASSES "
            f"{missing_classes} have no labeled prototype pixels outside the "
            "held-out test rectangle; excluded from C1-1..C1-5 fair semantic comparisons."
        )
    if len(classes) < 2:
        raise RuntimeError(
            "Fewer than 2 test classes have leakage-free prototype support outside "
            "the held-out rectangle; class-attribution diagnostic is not identifiable."
        )

    supported_np = np.isin(labels_np, np.asarray(classes, dtype=np.int64))
    supported_coverage = float(supported_np.mean())
    labeled_retention = float(
        supported_np.sum() / max(int((labels_np > 0).sum()), 1)
    )
    test_counts_all = {c: int((labels_np == c).sum()) for c in all_test_classes}
    test_counts_supported = {c: test_counts_all[c] for c in classes}
    perm = _perm_mapping(classes)

    print("=" * 150)
    print(
        f"DIAGNOSTIC stage=C1 dataset={args.dataset} test_coord=({top},{left}) "
        f"label_coverage={coverage:.6f} supported_coverage={supported_coverage:.6f} "
        f"labeled_retention={labeled_retention:.6f}"
    )
    print(f"TEST_CLASSES_ALL {all_test_classes}")
    print(f"TEST_CLASSES_SUPPORTED {classes}")
    print(f"TEST_CLASSES_EXCLUDED {missing_classes}")
    print(f"TEST_CLASS_COUNTS_ALL {test_counts_all}")
    print(f"TEST_CLASS_COUNTS_SUPPORTED {test_counts_supported}")
    print(f"OUTSIDE_TEST_PROTOTYPE_COUNTS {prototype_counts}")
    print(f"PERMUTED_CLASS_MAPPING {perm}")

    batch = next(iter(test_loader))
    gt, msi = batch["gt"].to(device), batch["hr_msi"].to(device)
    labels_t = torch.from_numpy(labels_np).to(device=device, dtype=torch.long).unsqueeze(0)
    supported_classes_t = torch.as_tensor(classes, device=device, dtype=torch.long)
    label_mask = (labels_t.unsqueeze(-1) == supported_classes_t.view(1, 1, 1, -1)).any(dim=-1)
    masks = _class_masks(labels_t, classes)
    risk = ranked_msi_heterogeneity(msi, eps=args.eps)

    with torch.no_grad():
        y = process.terminal_observation(gt)
        base = reconstruct_from_terminal_lr(baseline, process, y, target_size=tuple(gt.shape[-2:]), hr_msi=msi)
        e, raw = _physical_raw(process, projector, y, base)
        _, allowed = _ideal_allowed(projector, base, gt, args.eps)
        obs_weight = process.terminal_observation(label_mask.float().unsqueeze(1)).clamp_min(0)
        obs_weight = obs_weight / obs_weight.max().clamp_min(args.eps)

        alpha_raw, _ = _selfcal(process, e, raw, args.eps, None)
        upd_raw = alpha_raw * raw

        dir_local = _project_target(projector, base, _local_target(base, args.local_kernel), label_mask, args.eps)
        dir_class = _project_target(projector, base, _target_from_prototypes(base, labels_t, prototypes), label_mask, args.eps)
        alpha_class, _ = _selfcal(process, e, dir_class, args.eps, obs_weight)
        upd_class_global = alpha_class * dir_class
        dir_perm = _project_target(projector, base, _target_from_prototypes(base, labels_t, prototypes, perm), label_mask, args.eps)
        dir_oracle = _unit_direction(allowed, label_mask, args.eps)

    # Run the two decision-critical groups first: C1-5 closure ceiling and C1-3 semantic attribution.
    upd_oracle, st_oracle = _solve_magnitude(process, e.detach(), dir_oracle.detach(), obs_weight.detach(), args, "C1-5 oracle-dir", None)
    upd_class, st_class = _solve_magnitude(process, e.detach(), dir_class.detach(), obs_weight.detach(), args, "C1-3 class", masks.detach())
    upd_perm, st_perm = _solve_magnitude(process, e.detach(), dir_perm.detach(), obs_weight.detach(), args, "C1-4 permuted", masks.detach())
    upd_local, st_local = _solve_magnitude(process, e.detach(), dir_local.detach(), obs_weight.detach(), args, "C1-1 local", None)

    with torch.no_grad():
        rows = {
            "C1-0_raw_physical": _row(base + upd_raw, base, gt, upd_raw, allowed, label_mask, risk, args, {
                "ALPHA_GLOBAL": alpha_raw,
                "CLOSURE_REDUCTION_GLOBAL": _closure_reduction(e, process.terminal_observation(upd_raw), None, args.eps),
                "CLOSURE_REDUCTION_WEIGHTED": _closure_reduction(e, process.terminal_observation(upd_raw), obs_weight, args.eps),
            }),
            "C1-1_local_proto_inversion": _row(base + upd_local, base, gt, upd_local, allowed, label_mask, risk, args, st_local),
            "C1-2_oracle_class_global": _row(base + upd_class_global, base, gt, upd_class_global, allowed, label_mask, risk, args, {
                "ALPHA_GLOBAL": alpha_class,
                "CLOSURE_REDUCTION_GLOBAL": _closure_reduction(e, process.terminal_observation(upd_class_global), None, args.eps),
                "CLOSURE_REDUCTION_WEIGHTED": _closure_reduction(e, process.terminal_observation(upd_class_global), obs_weight, args.eps),
            }),
            "C1-3_oracle_class_attribution": _row(base + upd_class, base, gt, upd_class, allowed, label_mask, risk, args, st_class),
            "C1-4_permuted_class_attribution": _row(base + upd_perm, base, gt, upd_perm, allowed, label_mask, risk, args, st_perm),
            "C1-5_oracle_gt_direction_ceiling": _row(base + upd_oracle, base, gt, upd_oracle, allowed, label_mask, risk, args, st_oracle),
        }
        raw_all = _alignment(upd_raw, allowed, torch.ones_like(label_mask, dtype=torch.bool), args.eps)
        rows["C1-0_raw_physical"]["ALIGN_ALLOWED_ALL"] = raw_all[0]
        rows["C1-0_raw_physical"]["ALIGN_POS_FRAC_ALL"] = raw_all[1]
        base_metrics = calc_metrics(base, gt, args.scale_ratio)
        base_sem = _semantic_sam(base, gt, label_mask, risk, args.region_fraction, args.eps)

    print("=" * 150)
    print(f"BASELINE PSNR={base_metrics['PSNR']:.6f} SAM={base_metrics['SAM']:.6f} SAM_L={base_sem['SAM_LABELED']:.6f} SAM_H={base_sem['SAM_HIGH_LABELED']:.6f} SAM_LO={base_sem['SAM_LOW_LABELED']:.6f}")
    for name, row in rows.items():
        _print_row(name, row)

    r0, r3, r4, r5 = rows["C1-0_raw_physical"], rows["C1-3_oracle_class_attribution"], rows["C1-4_permuted_class_attribution"], rows["C1-5_oracle_gt_direction_ceiling"]
    print("=" * 150)
    print(f"DECISION_ALIGN raw_all={r0['ALIGN_ALLOWED_ALL']:+.6f} raw_labeled={r0['ALIGN_ALLOWED_LABELED']:+.6f} class_attr={r3['ALIGN_ALLOWED_LABELED']:+.6f} permuted={r4['ALIGN_ALLOWED_LABELED']:+.6f} oracle_dir={r5['ALIGN_ALLOWED_LABELED']:+.6f}")
    print(f"DECISION_SAM_L baseline={base_sem['SAM_LABELED']:.6f} raw={r0['SAM_LABELED']:.6f} class_attr={r3['SAM_LABELED']:.6f} permuted={r4['SAM_LABELED']:.6f} oracle_dir={r5['SAM_LABELED']:.6f}")
    print(f"ATTRIBUTION_GAPS class-vs-raw={r3['dSAM_LABELED']-r0['dSAM_LABELED']:+.6f} class-vs-permuted={r3['dSAM_LABELED']-r4['dSAM_LABELED']:+.6f} oracle-vs-class={r5['dSAM_LABELED']-r3['dSAM_LABELED']:+.6f}")

    payload = {
        "conditions": {
            "stage": "Innovation-3 Stage-C1 oracle semantic/material residual attribution",
            "dataset": args.dataset,
            "label_file": args.label_file,
            "label_key": args.label_key,
            "test_coord": [int(top), int(left)],
            "test_size": args.test_size,
            "test_label_coverage": coverage,
            "supported_label_coverage": supported_coverage,
            "supported_fraction_of_labeled_test_pixels": labeled_retention,
            "test_classes_all": all_test_classes,
            "test_classes_supported": classes,
            "test_classes_excluded_no_external_prototype": missing_classes,
            "test_class_counts_all": test_counts_all,
            "test_class_counts_supported": test_counts_supported,
            "outside_test_prototype_counts": prototype_counts,
            "prototype_support": "all labeled pixels outside held-out center test rectangle",
            "label_zero": "unlabeled, never a material class",
            "permuted_class_mapping": {str(k): int(v) for k, v in perm.items()},
            "solver_steps": args.solver_steps,
            "solver_lr": args.solver_lr,
            "solver_patience": args.solver_patience,
            "solver_tol": args.solver_tol,
            "lambda_amp": args.lambda_amp,
            "lambda_smooth": args.lambda_smooth,
            "amp_max": args.amp_max,
            "baseline_checkpoint": args.baseline_checkpoint,
            "seed": args.seed,
        },
        "definitions": {
            "C1-0": "raw terminal physical residual + global observable self-calibrated gain",
            "C1-1": "local prototype unit direction + LR nonnegative physical magnitude inversion",
            "C1-2": "oracle class identity + training class prototype unit direction + one weighted-closure scalar gain",
            "C1-3": "oracle class identity + class-specific LR magnitude fields from physical closure",
            "C1-4": "same class masks/DOF as C1-3 but cyclically wrong class-prototype identity",
            "C1-5": "unit oracle-allowed GT direction; GT magnitude removed; magnitude from physical closure",
            "semantic_closure_weight": "normalized D_T(1[label in leakage-free supported test classes])",
        },
        "baseline": {**base_metrics, **base_sem},
        "rows": rows,
        "decision": {
            "raw_align_all": r0["ALIGN_ALLOWED_ALL"],
            "raw_align_labeled": r0["ALIGN_ALLOWED_LABELED"],
            "class_attr_align": r3["ALIGN_ALLOWED_LABELED"],
            "permuted_attr_align": r4["ALIGN_ALLOWED_LABELED"],
            "oracle_direction_align": r5["ALIGN_ALLOWED_LABELED"],
            "class_vs_raw_extra_dSAM_labeled": r3["dSAM_LABELED"] - r0["dSAM_LABELED"],
            "class_vs_permuted_extra_dSAM_labeled": r3["dSAM_LABELED"] - r4["dSAM_LABELED"],
            "oracle_direction_vs_class_extra_dSAM_labeled": r5["dSAM_LABELED"] - r3["dSAM_LABELED"],
        },
    }
    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
