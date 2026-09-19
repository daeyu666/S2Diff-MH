"""Innovation-3 Stage-C2: fine-grained intra-class spectral-state capacity diagnostic.

Stage-C1 showed:
  * class-mean direction is essentially unaligned with the oracle SAM direction;
  * an oracle pixel-level direction plus physical closure gives a large SAM gain.

Stage-C2 asks how much spectral-state granularity is needed between those two
extremes.  The held-out center test rectangle never contributes to the class
dictionaries/manifolds.

Candidate direction families:
  mean                    : one leakage-free mean spectrum per class
  kmeans-base-K           : class spherical K-means; state selected by X_hat
  kmeans-oracle-K         : same codebook; state selected by GT (capacity only)
  pca-base-r              : class PCA manifold; coordinate inferred from X_hat
  pca-oracle-r            : same manifold; coordinate supplied by GT (capacity)
  oracle-nearest-training : GT chooses the closest same-class training spectrum
  oracle-gt-direction     : exact oracle-allowed direction ceiling

All candidate spectral targets are converted to the same C4:L tangent correction
space and normalized per HR pixel.  For fair comparison, every candidate uses
ONE shared nonnegative LR amplitude field optimized only by terminal physical
closure.  Hence differences primarily measure direction quality, not solver
degrees of freedom.

GT usage:
  * test class labels are oracle semantic identities for all class-conditioned
    candidates;
  * GT spectral values are used only for scoring, kmeans-oracle selection,
    pca-oracle coordinates, oracle-nearest-training selection, and the final
    oracle-direction ceiling;
  * base-selected candidates do not use GT spectra to select the fine state.

This is a capacity diagnostic, not a deployable Innovation-3 module.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from data_loader import build_loaders
from diagnose_innovation3_direction_calibration_capacity import (
    _build_baseline,
    _config,
    _ideal_allowed,
    _physical_raw,
)
from diagnose_innovation3_semantic_residual_attribution import (
    _alignment,
    _class_masks,
    _closure_reduction,
    _load_labels,
    _outside_test_support,
    _print_row,
    _project_target,
    _row,
    _selfcal,
    _solve_magnitude,
    _target_from_prototypes,
    _unit_direction,
)
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from metrics import calc_metrics
from models import HeterogeneityGuidedSpectralRefiner, ranked_msi_heterogeneity
from utils import ensure_dir, get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 Stage-C2 fine-grained intra-class spectral-state diagnostic"
    )
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
    p.add_argument(
        "--baseline_checkpoint_type",
        choices=["legacy", "standard"],
        default="legacy",
    )

    p.add_argument("--k_values", default="2,4,8,16")
    p.add_argument("--pca_ranks", default="2,4,8")
    p.add_argument("--kmeans_iters", type=int, default=40)
    p.add_argument("--nn_chunk", type=int, default=256)
    p.add_argument("--min_test_label_coverage", type=float, default=0.05)
    p.add_argument("--fail_on_low_coverage", action="store_true")
    p.add_argument("--region_fraction", type=float, default=0.25)

    p.add_argument("--solver_steps", type=int, default=300)
    p.add_argument("--solver_lr", type=float, default=5e-2)
    p.add_argument("--lambda_amp", type=float, default=1e-4)
    p.add_argument("--lambda_smooth", type=float, default=1e-4)
    p.add_argument("--amp_max", type=float, default=1.0)
    p.add_argument("--solver_log_interval", type=int, default=100)
    p.add_argument("--solver_patience", type=int, default=80)
    p.add_argument("--solver_tol", type=float, default=1e-7)
    p.add_argument("--eps", type=float, default=1e-8)

    p.add_argument(
        "--alignment_only",
        action="store_true",
        help="Compute the full granularity/alignment curve but skip physical magnitude inversion.",
    )
    p.add_argument(
        "--output_json",
        default="./results/innovation3_intra_class_spectral_state_PaviaU.json",
    )
    return p.parse_args()


def _parse_ints(text: str, *, name: str) -> Tuple[int, ...]:
    vals = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(f"{name} values must be >=1")
        if value not in vals:
            vals.append(value)
    if not vals:
        raise ValueError(f"{name} is empty")
    return tuple(vals)


def _normalize_rows_np(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, float(eps))


def _spherical_kmeans(
    spectra: np.ndarray,
    k: int,
    *,
    iters: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic spherical K-means, returning raw-mean prototypes."""
    raw = np.asarray(spectra, dtype=np.float32)
    n = int(raw.shape[0])
    if n < 1:
        raise ValueError("empty spectra")
    k_eff = min(int(k), n)
    unit = _normalize_rows_np(raw, eps)

    mean_dir = unit.mean(axis=0, keepdims=True)
    mean_dir = _normalize_rows_np(mean_dir, eps)[0]
    first = int(np.argmax(unit @ mean_dir))
    chosen = [first]
    min_dist = 1.0 - np.clip(unit @ unit[first], -1.0, 1.0)
    while len(chosen) < k_eff:
        idx = int(np.argmax(min_dist))
        if idx in chosen:
            remaining = [j for j in range(n) if j not in chosen]
            if not remaining:
                break
            idx = remaining[0]
        chosen.append(idx)
        dist = 1.0 - np.clip(unit @ unit[idx], -1.0, 1.0)
        min_dist = np.minimum(min_dist, dist)

    centers_unit = unit[np.asarray(chosen, dtype=np.int64)].copy()
    assign = np.full(n, -1, dtype=np.int64)
    for _ in range(max(int(iters), 1)):
        new_assign = np.argmax(unit @ centers_unit.T, axis=1).astype(np.int64)
        if np.array_equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        new_centers = []
        for j in range(centers_unit.shape[0]):
            m = assign == j
            if not np.any(m):
                new_centers.append(centers_unit[j])
            else:
                c = unit[m].mean(axis=0, keepdims=True)
                new_centers.append(_normalize_rows_np(c, eps)[0])
        centers_unit = np.stack(new_centers, axis=0)

    prototypes = []
    for j in range(centers_unit.shape[0]):
        m = assign == j
        if not np.any(m):
            prototypes.append(raw[chosen[j]])
        else:
            prototypes.append(raw[m].mean(axis=0))
    return np.stack(prototypes, axis=0).astype(np.float32), assign


def _fit_pca(spectra: np.ndarray, max_rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.asarray(spectra, dtype=np.float32)).double()
    mean = x.mean(dim=0)
    centered = x - mean
    if x.shape[0] <= 1:
        basis = torch.zeros((x.shape[1], 0), dtype=torch.float64)
        evals = torch.zeros((0,), dtype=torch.float64)
    else:
        cov = centered.T @ centered / float(max(int(x.shape[0]) - 1, 1))
        evals_all, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals_all, descending=True)
        r = min(int(max_rank), int(x.shape[1]), max(int(x.shape[0]) - 1, 0))
        order = order[:r]
        evals = evals_all[order].clamp_min(0.0)
        basis = evecs[:, order]
    return (
        mean.float().numpy(),
        basis.float().numpy(),
        evals.float().numpy(),
    )


def _build_training_spectra(
    image: np.ndarray,
    labels: np.ndarray,
    support: np.ndarray,
    classes: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, int], List[int]]:
    spectra, counts, missing = {}, {}, []
    for c in classes:
        x = image[support & (labels == int(c))]
        counts[int(c)] = int(x.shape[0])
        if x.shape[0] == 0:
            missing.append(int(c))
        else:
            spectra[int(c)] = np.asarray(x, dtype=np.float32)
    return spectra, counts, missing


def _mean_prototypes(training: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    return {c: x.mean(axis=0).astype(np.float32) for c, x in training.items()}


def _build_kmeans_codebooks(
    training: Dict[int, np.ndarray],
    k_values: Sequence[int],
    *,
    iters: int,
    eps: float,
):
    books = {}
    effective = {}
    for k in k_values:
        books[k] = {}
        effective[k] = {}
        for c, x in training.items():
            p, _ = _spherical_kmeans(x, k, iters=iters, eps=eps)
            books[k][c] = p
            effective[k][c] = int(p.shape[0])
    return books, effective


def _build_pca_models(training: Dict[int, np.ndarray], max_rank: int):
    models = {}
    for c, x in training.items():
        mean, basis, evals = _fit_pca(x, max_rank)
        models[c] = {"mean": mean, "basis": basis, "evals": evals}
    return models


def _assign_codebook_target(
    base: torch.Tensor,
    selector: torch.Tensor,
    labels: torch.Tensor,
    codebook: Dict[int, np.ndarray],
    *,
    eps: float,
) -> torch.Tensor:
    out = base.detach().clone().float()
    lab = labels[0]
    for c in sorted(codebook):
        m = lab == int(c)
        if not bool(m.any()):
            continue
        q = selector[0, :, m].T.float()
        qn = q / torch.linalg.vector_norm(q, dim=1, keepdim=True).clamp_min(eps)
        p = torch.as_tensor(codebook[c], device=base.device, dtype=torch.float32)
        pn = p / torch.linalg.vector_norm(p, dim=1, keepdim=True).clamp_min(eps)
        idx = torch.argmax(qn @ pn.T, dim=1)
        out[0, :, m] = p[idx].T
    return out


def _pca_target(
    base: torch.Tensor,
    selector: torch.Tensor,
    labels: torch.Tensor,
    models,
    rank: int,
) -> torch.Tensor:
    out = base.detach().clone().float()
    lab = labels[0]
    for c in sorted(models):
        m = lab == int(c)
        if not bool(m.any()):
            continue
        model = models[c]
        mean = torch.as_tensor(model["mean"], device=base.device, dtype=torch.float32)
        basis_np = model["basis"]
        r = min(int(rank), int(basis_np.shape[1]))
        if r <= 0:
            out[0, :, m] = mean.view(-1, 1)
            continue
        basis = torch.as_tensor(basis_np[:, :r], device=base.device, dtype=torch.float32)
        x = selector[0, :, m].T.float()
        coord = (x - mean) @ basis
        projection = mean + coord @ basis.T
        out[0, :, m] = projection.T
    return out


def _nearest_training_target(
    base: torch.Tensor,
    gt: torch.Tensor,
    labels: torch.Tensor,
    training: Dict[int, np.ndarray],
    *,
    chunk: int,
    eps: float,
) -> torch.Tensor:
    if chunk < 1:
        raise ValueError("--nn_chunk must be >=1")
    out = base.detach().clone().float()
    lab = labels[0]
    for c in sorted(training):
        m = lab == int(c)
        if not bool(m.any()):
            continue
        query = gt[0, :, m].T.float()
        query = query / torch.linalg.vector_norm(query, dim=1, keepdim=True).clamp_min(eps)
        bank_raw = torch.as_tensor(training[c], device=base.device, dtype=torch.float32)
        bank = bank_raw / torch.linalg.vector_norm(bank_raw, dim=1, keepdim=True).clamp_min(eps)
        selected = []
        for start in range(0, query.shape[0], int(chunk)):
            q = query[start:start + int(chunk)]
            idx = torch.argmax(q @ bank.T, dim=1)
            selected.append(bank_raw[idx])
        chosen = torch.cat(selected, dim=0)
        out[0, :, m] = chosen.T
    return out


def _candidate_direction(
    projector,
    base,
    target,
    support,
    eps,
):
    return _project_target(projector, base, target, support, eps)


def _candidate_alignment(name, direction, allowed, support, eps):
    align, pos = _alignment(direction, allowed, support, eps)
    print(f"DIRECTION {name:<28} ALIGN={align:+.6f} POS={pos:.6f}")
    return {"ALIGN_ALLOWED_LABELED": align, "ALIGN_POS_FRAC_LABELED": pos}


def _variance_capture(models, rank: int) -> Dict[int, float]:
    out = {}
    for c, model in models.items():
        eig = np.asarray(model["evals"], dtype=np.float64)
        if eig.size == 0 or float(eig.sum()) <= 0.0:
            out[c] = float("nan")
        else:
            out[c] = float(eig[: min(int(rank), eig.size)].sum() / eig.sum())
    return out


def main():
    args = parse_args()
    k_values = _parse_ints(args.k_values, name="k_values")
    pca_ranks = _parse_ints(args.pca_ranks, name="pca_ranks")
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")

    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    train_loader, test_loader, info = build_loaders(cfg)
    process = build_progressive_process(cfg)
    baseline = _build_baseline(args, info, device)

    projector = HeterogeneityGuidedSpectralRefiner(
        n_bands=info["n_bands"],
        total_steps=args.diffusion_steps,
        hidden_channels=1,
        variant="full",
        eps=args.eps,
    ).to(device).eval()
    for parameter in projector.parameters():
        parameter.requires_grad_(False)

    image = np.asarray(train_loader.dataset.image, dtype=np.float32)
    labels = _load_labels(args.label_file, args.label_key, image.shape[:2])
    if len(test_loader.dataset.coords) != 1:
        raise RuntimeError("Stage-C2 expects the repository single center test patch")
    top, left = test_loader.dataset.coords[0]
    labels_np = labels[top:top + args.test_size, left:left + args.test_size]
    if labels_np.shape != (args.test_size, args.test_size):
        raise RuntimeError(
            f"test label crop shape={labels_np.shape}, expected={(args.test_size, args.test_size)}"
        )

    all_test_classes = sorted(int(c) for c in np.unique(labels_np) if c > 0)
    label_coverage = float((labels_np > 0).mean())
    if not all_test_classes:
        raise RuntimeError("center test patch has no labeled pixels")
    if label_coverage < args.min_test_label_coverage:
        msg = (
            f"label coverage {label_coverage:.6f} < {args.min_test_label_coverage:.6f}; "
            "conclusions apply only to sparse labeled support"
        )
        if args.fail_on_low_coverage:
            raise RuntimeError(msg)
        print("WARNING", msg)

    outside = _outside_test_support(image.shape[:2], top, left, args.test_size)
    training, train_counts, missing = _build_training_spectra(
        image, labels, outside, all_test_classes
    )
    classes = [c for c in all_test_classes if c in training]
    if missing:
        print(
            "WARNING UNSUPPORTED_TEST_CLASSES "
            f"{missing} have no labeled spectra outside held-out test rectangle; excluded."
        )
    if not classes:
        raise RuntimeError("No test classes have leakage-free external spectra")

    supported_np = np.isin(labels_np, np.asarray(classes, dtype=np.int64))
    supported_coverage = float(supported_np.mean())
    labeled_retention = float(
        supported_np.sum() / max(int((labels_np > 0).sum()), 1)
    )
    test_counts = {c: int((labels_np == c).sum()) for c in all_test_classes}

    means = _mean_prototypes(training)
    codebooks, effective_k = _build_kmeans_codebooks(
        training, k_values, iters=args.kmeans_iters, eps=args.eps
    )
    pca_models = _build_pca_models(training, max(pca_ranks))

    print("=" * 160)
    print(
        f"DIAGNOSTIC stage=C2 dataset={args.dataset} test_coord=({top},{left}) "
        f"label_coverage={label_coverage:.6f} supported_coverage={supported_coverage:.6f} "
        f"labeled_retention={labeled_retention:.6f}"
    )
    print(f"TEST_CLASSES_ALL {all_test_classes}")
    print(f"TEST_CLASSES_SUPPORTED {classes}")
    print(f"TEST_CLASSES_EXCLUDED {missing}")
    print(f"TEST_CLASS_COUNTS {test_counts}")
    print(f"TRAIN_SPECTRA_COUNTS {train_counts}")
    print(f"K_VALUES {k_values} EFFECTIVE_K {effective_k}")
    for rank in pca_ranks:
        print(f"PCA_VARIANCE_CAPTURE rank={rank} {_variance_capture(pca_models, rank)}")

    batch = next(iter(test_loader))
    gt = batch["gt"].to(device)
    msi = batch["hr_msi"].to(device)
    labels_t = torch.from_numpy(labels_np).to(device=device, dtype=torch.long).unsqueeze(0)
    supported_t = torch.as_tensor(classes, device=device, dtype=torch.long)
    label_mask = (
        labels_t.unsqueeze(-1) == supported_t.view(1, 1, 1, -1)
    ).any(dim=-1)
    risk = ranked_msi_heterogeneity(msi, eps=args.eps)

    with torch.no_grad():
        terminal_lr = process.terminal_observation(gt)
        base = reconstruct_from_terminal_lr(
            baseline,
            process,
            terminal_lr,
            target_size=tuple(gt.shape[-2:]),
            hr_msi=msi,
        )
        residual_native, raw = _physical_raw(
            process, projector, terminal_lr, base
        )
        _, allowed = _ideal_allowed(projector, base, gt, args.eps)
        obs_weight = process.terminal_observation(
            label_mask.float().unsqueeze(1)
        ).clamp_min(0.0)
        obs_weight = obs_weight / obs_weight.max().clamp_min(args.eps)

        candidates: Dict[str, torch.Tensor] = {}

        mean_target = _target_from_prototypes(base, labels_t, means)
        candidates["mean"] = _candidate_direction(
            projector, base, mean_target, label_mask, args.eps
        )

        for k in k_values:
            target_base = _assign_codebook_target(
                base, base, labels_t, codebooks[k], eps=args.eps
            )
            target_oracle = _assign_codebook_target(
                base, gt, labels_t, codebooks[k], eps=args.eps
            )
            candidates[f"kmeans_base_k{k}"] = _candidate_direction(
                projector, base, target_base, label_mask, args.eps
            )
            candidates[f"kmeans_oracle_k{k}"] = _candidate_direction(
                projector, base, target_oracle, label_mask, args.eps
            )

        for rank in pca_ranks:
            target_base = _pca_target(
                base, base, labels_t, pca_models, rank
            )
            target_oracle = _pca_target(
                base, gt, labels_t, pca_models, rank
            )
            candidates[f"pca_base_r{rank}"] = _candidate_direction(
                projector, base, target_base, label_mask, args.eps
            )
            candidates[f"pca_oracle_r{rank}"] = _candidate_direction(
                projector, base, target_oracle, label_mask, args.eps
            )

        target_nn = _nearest_training_target(
            base,
            gt,
            labels_t,
            training,
            chunk=args.nn_chunk,
            eps=args.eps,
        )
        candidates["oracle_nearest_training"] = _candidate_direction(
            projector, base, target_nn, label_mask, args.eps
        )
        candidates["oracle_gt_direction"] = _unit_direction(
            allowed, label_mask, args.eps
        )

        alpha_raw, _ = _selfcal(
            process, residual_native, raw, args.eps, None
        )
        update_raw = alpha_raw * raw

    direction_rows = {
        name: _candidate_alignment(
            name, direction, allowed, label_mask, args.eps
        )
        for name, direction in candidates.items()
    }
    raw_align_l, raw_pos_l = _alignment(
        update_raw, allowed, label_mask, args.eps
    )
    raw_align_all, raw_pos_all = _alignment(
        update_raw,
        allowed,
        torch.ones_like(label_mask, dtype=torch.bool),
        args.eps,
    )
    print(
        f"DIRECTION {'raw_physical':<28} ALIGN_L={raw_align_l:+.6f} "
        f"POS_L={raw_pos_l:.6f} ALIGN_ALL={raw_align_all:+.6f} POS_ALL={raw_pos_all:.6f}"
    )

    base_metrics = calc_metrics(base, gt, args.scale_ratio)
    rows = {}

    if not args.alignment_only:
        raw_q = process.terminal_observation(update_raw)
        rows["raw_physical"] = _row(
            base + update_raw,
            base,
            gt,
            update_raw,
            allowed,
            label_mask,
            risk,
            args,
            {
                "ALPHA_GLOBAL": alpha_raw,
                "CLOSURE_REDUCTION_GLOBAL": _closure_reduction(
                    residual_native, raw_q, None, args.eps
                ),
                "CLOSURE_REDUCTION_WEIGHTED": _closure_reduction(
                    residual_native, raw_q, obs_weight, args.eps
                ),
            },
        )

        for index, (name, direction) in enumerate(candidates.items(), start=1):
            print("=" * 160)
            print(
                f"INVERT {index}/{len(candidates)} method={name} "
                "magnitude=one_shared_LR_field closure_only"
            )
            update, stats = _solve_magnitude(
                process,
                residual_native.detach(),
                direction.detach(),
                obs_weight.detach(),
                args,
                name,
                class_masks=None,
            )
            rows[name] = _row(
                base + update,
                base,
                gt,
                update,
                allowed,
                label_mask,
                risk,
                args,
                stats,
            )

        print("=" * 160)
        print(
            f"BASELINE PSNR={base_metrics['PSNR']:.6f} "
            f"SAM={base_metrics['SAM']:.6f}"
        )
        _print_row("raw_physical", rows["raw_physical"])
        for name in candidates:
            _print_row(name, rows[name])

    ordered_curve = ["mean"]
    for k in k_values:
        ordered_curve.extend([f"kmeans_base_k{k}", f"kmeans_oracle_k{k}"])
    for rank in pca_ranks:
        ordered_curve.extend([f"pca_base_r{rank}", f"pca_oracle_r{rank}"])
    ordered_curve.extend(["oracle_nearest_training", "oracle_gt_direction"])

    print("=" * 160)
    print("GRANULARITY_ALIGNMENT_CURVE")
    print(
        f"raw_physical labeled={raw_align_l:+.6f} all={raw_align_all:+.6f}"
    )
    for name in ordered_curve:
        d = direction_rows[name]
        if args.alignment_only:
            print(
                f"{name:<28} ALIGN={d['ALIGN_ALLOWED_LABELED']:+.6f} "
                f"POS={d['ALIGN_POS_FRAC_LABELED']:.6f}"
            )
        else:
            r = rows[name]
            print(
                f"{name:<28} ALIGN={d['ALIGN_ALLOWED_LABELED']:+.6f} "
                f"POS={d['ALIGN_POS_FRAC_LABELED']:.6f} "
                f"SAM={r['SAM']:.6f} SAM_L={r['SAM_LABELED']:.6f} "
                f"dSAM_L={r['dSAM_LABELED']:+.6f}"
            )

    base_selected = [
        "mean",
        *[f"kmeans_base_k{k}" for k in k_values],
        *[f"pca_base_r{r}" for r in pca_ranks],
    ]
    oracle_state = [
        *[f"kmeans_oracle_k{k}" for k in k_values],
        *[f"pca_oracle_r{r}" for r in pca_ranks],
        "oracle_nearest_training",
    ]
    best_base_align = max(
        base_selected,
        key=lambda n: direction_rows[n]["ALIGN_ALLOWED_LABELED"],
    )
    best_oracle_state_align = max(
        oracle_state,
        key=lambda n: direction_rows[n]["ALIGN_ALLOWED_LABELED"],
    )

    print("=" * 160)
    print(
        "DECISION "
        f"raw_align={raw_align_l:+.6f} "
        f"mean_align={direction_rows['mean']['ALIGN_ALLOWED_LABELED']:+.6f} "
        f"best_base_state={best_base_align}:"
        f"{direction_rows[best_base_align]['ALIGN_ALLOWED_LABELED']:+.6f} "
        f"best_oracle_state={best_oracle_state_align}:"
        f"{direction_rows[best_oracle_state_align]['ALIGN_ALLOWED_LABELED']:+.6f} "
        f"nearest_train={direction_rows['oracle_nearest_training']['ALIGN_ALLOWED_LABELED']:+.6f} "
        f"oracle_gt={direction_rows['oracle_gt_direction']['ALIGN_ALLOWED_LABELED']:+.6f}"
    )
    if not args.alignment_only:
        print(
            "DECISION_SAM_L "
            f"raw={rows['raw_physical']['SAM_LABELED']:.6f} "
            f"mean={rows['mean']['SAM_LABELED']:.6f} "
            f"best_base={best_base_align}:{rows[best_base_align]['SAM_LABELED']:.6f} "
            f"best_oracle_state={best_oracle_state_align}:"
            f"{rows[best_oracle_state_align]['SAM_LABELED']:.6f} "
            f"nearest_train={rows['oracle_nearest_training']['SAM_LABELED']:.6f} "
            f"oracle_gt={rows['oracle_gt_direction']['SAM_LABELED']:.6f}"
        )

    payload = {
        "conditions": {
            "stage": "Innovation-3 Stage-C2 fine-grained intra-class spectral-state capacity",
            "dataset": args.dataset,
            "label_file": args.label_file,
            "label_key": args.label_key,
            "test_coord": [int(top), int(left)],
            "test_size": args.test_size,
            "label_coverage": label_coverage,
            "supported_label_coverage": supported_coverage,
            "supported_fraction_of_labeled_test_pixels": labeled_retention,
            "test_classes_all": all_test_classes,
            "test_classes_supported": classes,
            "test_classes_excluded": missing,
            "test_class_counts": test_counts,
            "training_spectra_counts": train_counts,
            "training_support": "all labeled spectra outside held-out center test rectangle",
            "k_values": list(k_values),
            "effective_k": {
                str(k): {str(c): int(v) for c, v in effective_k[k].items()}
                for k in k_values
            },
            "pca_ranks": list(pca_ranks),
            "pca_variance_capture": {
                str(r): {str(c): v for c, v in _variance_capture(pca_models, r).items()}
                for r in pca_ranks
            },
            "magnitude_solver": "one shared nonnegative LR amplitude field for every candidate",
            "solver_steps": args.solver_steps,
            "solver_lr": args.solver_lr,
            "lambda_amp": args.lambda_amp,
            "lambda_smooth": args.lambda_smooth,
            "amp_max": args.amp_max,
            "alignment_only": args.alignment_only,
            "seed": args.seed,
            "baseline_checkpoint": args.baseline_checkpoint,
        },
        "definitions": {
            "mean": "one class mean spectrum",
            "kmeans_base": "class spherical-K-means prototype selected by current reconstruction X_hat",
            "kmeans_oracle": "same class codebook, prototype selected by GT spectrum; diagnostic only",
            "pca_base": "class PCA projection coordinate inferred from X_hat",
            "pca_oracle": "class PCA projection coordinate supplied by GT spectrum; diagnostic only",
            "oracle_nearest_training": "GT selects closest same-class leakage-free training spectrum; diagnostic only",
            "oracle_gt_direction": "unit oracle-allowed C4:L tangent direction; diagnostic ceiling",
            "direction_projection": "all targets converted through the same C4:L + tangent projection and per-pixel unit normalization",
        },
        "baseline": base_metrics,
        "raw_direction": {
            "ALIGN_ALLOWED_LABELED": raw_align_l,
            "ALIGN_POS_FRAC_LABELED": raw_pos_l,
            "ALIGN_ALLOWED_ALL": raw_align_all,
            "ALIGN_POS_FRAC_ALL": raw_pos_all,
        },
        "direction_rows": direction_rows,
        "metric_rows": rows,
        "decision": {
            "best_base_selected_alignment_method": best_base_align,
            "best_base_selected_alignment": direction_rows[best_base_align]["ALIGN_ALLOWED_LABELED"],
            "best_oracle_state_alignment_method": best_oracle_state_align,
            "best_oracle_state_alignment": direction_rows[best_oracle_state_align]["ALIGN_ALLOWED_LABELED"],
            "oracle_nearest_training_alignment": direction_rows["oracle_nearest_training"]["ALIGN_ALLOWED_LABELED"],
            "oracle_gt_alignment": direction_rows["oracle_gt_direction"]["ALIGN_ALLOWED_LABELED"],
        },
    }
    if not args.alignment_only:
        payload["decision"].update(
            {
                "raw_sam_labeled": rows["raw_physical"]["SAM_LABELED"],
                "mean_sam_labeled": rows["mean"]["SAM_LABELED"],
                "best_base_sam_labeled": rows[best_base_align]["SAM_LABELED"],
                "best_oracle_state_sam_labeled": rows[best_oracle_state_align]["SAM_LABELED"],
                "nearest_training_sam_labeled": rows["oracle_nearest_training"]["SAM_LABELED"],
                "oracle_gt_sam_labeled": rows["oracle_gt_direction"]["SAM_LABELED"],
            }
        )

    ensure_dir(os.path.dirname(args.output_json))
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"SAVED_JSON {args.output_json}")


if __name__ == "__main__":
    main()
