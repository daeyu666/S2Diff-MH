"""Diagnose whether recursive Wald-CDRDI steps improve on its -0.5px seed.

One forward pass for max(depths) per identical validation tile, extracting
intermediate predictions. No training, no EnMAP10, no flow GT, no input warping.
Uses the exact normalized Charbonnier observation-closure metric of Real-C.

Example:
python diagnose_augsburg2_wald_cdrdi_depths.py \
    --checkpoint ./checkpoints/augsburg_real/Augsburg2_Wald_C_seeded_minus05.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from augsburg_real import AugsburgRealDataset
from cdrdi_geometry import jacobian_determinant
from degradations.effective_gaussian import EffectiveGaussianDegradation
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver, image_gradients
from train_augsburg_real_cdrdi import (
    _assert_wald_checkpoint_settings,
    _charbonnier,
    _estimate_batch,
    _load_radiometry,
    _load_sigma,
    _local_standardize,
    _wald_checkpoint_guard,
)
from utils import get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Identical-tile Wald-CDRDI diagnostic at 0,1,3,6,9 recursive steps"
    )
    p.add_argument("--checkpoint", required=True, help="Early best Wald-CDRDI checkpoint")
    p.add_argument("--wald_root", default="./data/augsburg2_wald")
    p.add_argument("--psf_json", default="./data/augsburg2_wald/wald_psf.json")
    p.add_argument("--radiometry_json", default="./data/calibration/Augsburg2_Wald_radiometry.json")
    p.add_argument("--eval_patch_size", type=int, default=48)
    p.add_argument("--min_valid_fraction", type=float, default=0.8)
    p.add_argument("--local_window", type=int, default=5)
    p.add_argument("--depths", default="0,1,3,6,9",
                   help="Comma-separated depths. A single max-depth pass supplies all intermediate states")
    p.add_argument("--modes", default="both",
                   help="Comma-separated per-iteration modes: both,rigid_only,local_only,seed_only")
    p.add_argument("--policies", default="plain",
                   help="Comma-separated policies: plain,closure_backtrack; guarded policy never trains")
    p.add_argument("--acceptance_min_jac", type=float, default=0.5)
    p.add_argument("--acceptance_relative_gain", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--output_csv", default="./logs/augsburg_real/Augsburg2_Wald_C_depth_diagnostic.csv")
    p.add_argument("--per_tile_csv", default="",
                   help="Optional per-tile CSV for locating harmful residual updates")
    return p.parse_args()


def parse_depths(value: str):
    depths = sorted(set(int(d.strip()) for d in value.split(",") if d.strip()))
    if not depths or depths[0] < 0 or depths[-1] > 64:
        raise ValueError("--depths must specify integers from 0 through 64")
    return depths


def parse_modes(value: str):
    allowed = ("both", "rigid_only", "local_only", "seed_only")
    modes = list(dict.fromkeys(m.strip() for m in value.split(",") if m.strip()))
    if not modes or any(mode not in allowed for mode in modes):
        raise ValueError("modes must be chosen from: " + ",".join(allowed))
    return modes


def parse_policies(value: str):
    allowed = ("plain", "closure_backtrack")
    policies = list(dict.fromkeys(p.strip() for p in value.split(",") if p.strip()))
    if not policies or any(policy not in allowed for policy in policies):
        raise ValueError("policies must be chosen from: " + ",".join(allowed))
    return policies


def _closure_values(target, prediction, mask, *, window):
    target_normalized = _local_standardize(target, window)
    prediction_normalized = _local_standardize(prediction, window)
    norm = _charbonnier(target_normalized - prediction_normalized, mask)
    raw = _charbonnier(target - prediction, mask)
    tgx, tgy = image_gradients(target_normalized)
    pgx, pgy = image_gradients(prediction_normalized)
    grad = 0.5 * (_charbonnier(tgx - pgx, mask) + _charbonnier(tgy - pgy, mask))
    return {"norm": float(norm.item()), "raw": float(raw.item()), "grad": float(grad.item())}


def _geometry_values(rigid, local, *, seed_dx, seed_dy):
    local_magnitude = torch.linalg.vector_norm(local, dim=1)
    jac = jacobian_determinant(local)
    return {
        "dx": float(rigid[:, 0].mean().item()),
        "dy": float(rigid[:, 1].mean().item()),
        "theta": float(rigid[:, 2].mean().item()),
        "theta_abs": float(rigid[:, 2].abs().mean().item()),
        "delta_dx": float((rigid[:, 0] - seed_dx).mean().item()),
        "delta_dy": float((rigid[:, 1] - seed_dy).mean().item()),
        "local_mean": float(local_magnitude.mean().item()),
        "local_max": float(local_magnitude.amax().item()),
        "min_jac": float(jac.amin().item()),
    }


def _new_record():
    return {
        "weighted": {k: 0.0 for k in (
            "norm", "raw", "grad", "dx", "dy", "theta", "theta_abs",
            "delta_dx", "delta_dy", "local_mean", "local_max",
            "step_accept", "cumulative_accept",
        )},
        "pixels": 0.0,
        "min_jac": float("inf"),
        "wins": 0,
        "tiles": 0,
    }


def collect_depth_metrics(model, loader, *, p0, srf, radiometry, depths, device,
                          local_window=5, tile_positions=None, update_mode="both",
                          update_policy="plain", acceptance_min_jac=0.5,
                          acceptance_relative_gain=1e-4):
    """Evaluate identity, seed (depth 0), and recursively updated states.

    All rows use the exact same LR observation, MSI, ROI mask, and tile weighting.
    The model runs max(depths) once for each sample, so intermediate depths
    cannot differ due to separate augmentation or sampling.
    """
    depths = sorted(set(int(d) for d in depths))
    if not depths or depths[0] < 0:
        raise ValueError("depths must be nonnegative")
    if not hasattr(model, "initial_dx_px") or not hasattr(model, "initial_dy_px"):
        raise TypeError("model must provide initial physical geometry offsets")
    if max(depths) < 1:
        # The solver's forward requires at least one step; ignore that step.
        eval_steps = 1
    else:
        eval_steps = max(depths)
    seed_dx, seed_dy = model.initial_dx_px, model.initial_dy_px
    model.eval()
    records = {d: _new_record() for d in ("identity", *depths)}
    tiles_out = []

    with torch.no_grad():
        for tile_index, batch in enumerate(loader):
            target, mask, outputs = _estimate_batch(
                model, batch, p0=p0, srf=srf, radiometry=radiometry,
                steps=eval_steps, device=device, augment_shift_px=0.0,
                update_mode=update_mode,
                update_policy=update_policy,
                acceptance_window=local_window,
                acceptance_min_jac=acceptance_min_jac,
                acceptance_relative_gain=acceptance_relative_gain,
            )
            weight = float(mask[:, 0].sum().item())
            if weight <= 0:
                continue
            b, _, height, width = batch["hr_msi"].shape
            seed_rigid = torch.tensor(
                [[seed_dx, seed_dy, 0.0]], device=device, dtype=target.dtype
            ).expand(b, -1)
            identity_rigid = torch.zeros_like(seed_rigid)
            zero_field = torch.zeros(
                (b, 2, height, width), device=device, dtype=target.dtype
            )
            states = {
                "identity": (outputs["unaligned_prediction"], identity_rigid, zero_field)
            }
            for d in depths:
                if d == 0:
                    states[d] = (outputs["initial_prediction"], seed_rigid, zero_field)
                else:
                    states[d] = (
                        outputs["predictions"][d - 1],
                        outputs["rigid_states"][d - 1],
                        outputs["local_fields"][d - 1],
                    )

            # Even if only positive depths were requested, the same fixed
            # seed is always the residual-reduction reference.
            seed_norm = _closure_values(
                target, outputs["initial_prediction"], mask, window=local_window
            )["norm"]

            for d, (prediction, rigid, local) in states.items():
                close = _closure_values(target, prediction, mask, window=local_window)
                geo = _geometry_values(
                    rigid, local, seed_dx=seed_dx, seed_dy=seed_dy
                )
                if isinstance(d, int) and d > 0:
                    step_accept = float(outputs["accepted_steps"][d - 1].float().mean().item())
                    cumulative_accept = float(torch.stack(
                        outputs["accepted_steps"][:d], dim=0
                    ).float().mean().item())
                else:
                    step_accept, cumulative_accept = 0.0, 0.0
                rec = records[d]
                for k in rec["weighted"]:
                    value = close[k] if k in close else (
                        step_accept if k == "step_accept" else (
                            cumulative_accept if k == "cumulative_accept" else geo[k]
                        )
                    )
                    rec["weighted"][k] += value * weight
                rec["pixels"] += weight
                rec["min_jac"] = min(rec["min_jac"], geo["min_jac"])
                rec["wins"] += int(close["norm"] < seed_norm - 1e-12)
                rec["tiles"] += 1
                if tile_positions is not None:
                    top, left, ph, pw = tile_positions[tile_index]
                    tiles_out.append({
                        "mode": update_mode,
                        "policy": update_policy,
                        "tile": tile_index, "top": top, "left": left,
                        "height": ph, "width": pw, "depth": d,
                        "norm": close["norm"], "raw": close["raw"],
                        "dx": geo["dx"], "dy": geo["dy"], "theta": geo["theta"],
                        "local_mean": geo["local_mean"], "min_jac": geo["min_jac"],
                        "step_accept": step_accept, "cumulative_accept": cumulative_accept,
                        "improved_over_seed": int(close["norm"] < seed_norm - 1e-12),
                    })

    if any(r["pixels"] <= 0 for r in records.values()):
        raise ValueError("Validation returned no valid observation pixels")
    summary = {}
    for depth, rec in records.items():
        row = {k: v / rec["pixels"] for k, v in rec["weighted"].items()}
        row["depth"] = depth
        row["valid_lr_pixels"] = rec["pixels"]
        row["n_tiles"] = rec["tiles"]
        row["win_rate_vs_seed"] = 100.0 * rec["wins"] / max(rec["tiles"], 1)
        row["min_jac"] = rec["min_jac"]
        summary[depth] = row

    identity = summary["identity"]["norm"]
    seed = (summary[0]["norm"] if 0 in summary else None)
    if seed is None:
        # Allow requests without 0 but always evaluate it in practice.
        raise ValueError("Include depth 0 in --depths for an unbiased seed comparison")
    for depth, row in summary.items():
        row["reduction_vs_identity_pct"] = 100.0 * (1.0 - row["norm"] / max(identity, 1e-12))
        row["reduction_vs_seed_pct"] = 100.0 * (1.0 - row["norm"] / max(seed, 1e-12))
    return summary, tiles_out


def _write_csv(path, rows):
    if not rows:
        return
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    depths = parse_depths(args.depths)
    modes = parse_modes(args.modes)
    policies = parse_policies(args.policies)
    if not 0.0 <= args.acceptance_relative_gain < 1.0:
        raise ValueError("--acceptance_relative_gain must lie in [0,1)")
    if not 0.0 < args.acceptance_min_jac <= 1.5:
        raise ValueError("--acceptance_min_jac must lie in (0,1.5]")
    if 0 not in depths:
        raise ValueError("--depths must include 0 for the -0.5px seed comparison")
    if args.eval_patch_size % 3:
        raise ValueError("eval_patch_size must be divisible by 3")
    set_seed(args.seed)
    device = get_device(args.device)
    extra = _wald_checkpoint_guard(args.checkpoint, stage="diagnostic")
    geometry = extra["geometry_config"]
    sigma = _load_sigma(args.psf_json)
    config = SimpleNamespace(
        base_channels=int(geometry["base_channels"]),
        control_grid=int(geometry["control_grid"]),
        max_translation=float(geometry["max_translation"]),
        max_rotation_deg=float(geometry["max_rotation_deg"]),
        max_local_px=float(geometry["max_local_px"]),
        initial_dx_px=float(geometry["initial_dx_px"]),
        initial_dy_px=float(geometry["initial_dy_px"]),
        radiometry_json=args.radiometry_json,
    )
    _assert_wald_checkpoint_settings(extra, config, sigma=sigma)

    model = LearnedPhysicalResidualSolver(
        4,
        base_channels=config.base_channels,
        control_grid=config.control_grid,
        max_translation=config.max_translation,
        max_rotation_deg=config.max_rotation_deg,
        max_local_px=config.max_local_px,
        initial_dx_px=config.initial_dx_px,
        initial_dy_px=config.initial_dy_px,
    ).to(device)
    epoch, best_score = load_checkpoint(
        model, args.checkpoint, map_location=str(device), load_optimizer=False
    )
    model.eval()
    dataset = AugsburgRealDataset(
        args.wald_root,
        "validation",
        train_patch_size=72,
        train_stride=6,
        eval_patch_size=args.eval_patch_size,
        min_valid_fraction=args.min_valid_fraction,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    srf = torch.tensor(
        np.load(os.path.join(args.wald_root, "srf_weights.npy")),
        dtype=torch.float32, device=device
    )
    p0 = EffectiveGaussianDegradation(
        scale_ratio=3, terminal_sigma=sigma, truncate=3.0
    ).to(device)
    radiometry = _load_radiometry(args.radiometry_json, device)

    print(
        f"DEPTH_DIAGNOSTIC checkpoint={args.checkpoint} epoch={epoch} "
        f"best_score={best_score:.8f} train_steps={extra.get('train_steps')} "
        f"saved_eval_steps={extra.get('eval_steps')} "
        f"fixed_init=(dx={config.initial_dx_px},dy={config.initial_dy_px}) "
        f"wald_sigma={sigma} valid_tiles={len(dataset)} "
        "augmentation=False spatial_split=validation"
    )
    all_rows = {}
    combined = []
    all_tile_rows = []
    combinations = [(policy, mode) for policy in policies for mode in modes]
    for policy, mode in combinations:
        rows, tile_rows = collect_depth_metrics(
            model, loader, p0=p0, srf=srf, radiometry=radiometry,
            depths=depths, device=device, local_window=args.local_window,
            tile_positions=dataset.samples if args.per_tile_csv else None,
            update_mode=mode,
            update_policy=policy,
            acceptance_min_jac=args.acceptance_min_jac,
            acceptance_relative_gain=args.acceptance_relative_gain,
        )
        all_rows[(policy, mode)] = rows
        all_tile_rows.extend(tile_rows)
        for depth in ("identity", *depths):
            m = rows[depth]
            simple = combinations == [("plain", "both")]
            tag = "CDRDI_DEPTH" if simple else "CDRDI_GUARD"
            prefix = "" if simple else f"policy={policy} mode={mode} "
            print(
                f"{tag} {prefix}step={depth} NORM={m['norm']:.8f} RAW={m['raw']:.8f} "
                f"GRAD={m['grad']:.8f} "
                f"REDUCTION={m['reduction_vs_identity_pct']:+.3f}% "
                f"RESIDUAL_REDUCTION={m['reduction_vs_seed_pct']:+.3f}% "
                f"DX={m['dx']:+.4f} DY={m['dy']:+.4f} "
                f"DELTA_DX={m['delta_dx']:+.4f} DELTA_DY={m['delta_dy']:+.4f} "
                f"THETA={m['theta']:+.4f} LOCAL_MEAN={m['local_mean']:.4f} "
                f"LOCAL_MAX={m['local_max']:.4f} MIN_JAC={m['min_jac']:.5f} "
                f"STEP_ACCEPT={100*m['step_accept']:.2f}% "
                f"CUM_ACCEPT={100*m['cumulative_accept']:.2f}% "
                f"WIN_RATE={m['win_rate_vs_seed']:.2f}%"
            )
            combined.append({"policy": policy, "mode": mode, **m})

        best_depth = min(depths, key=lambda d: rows[d]["norm"])
        harmful = [d for d in depths if d > 0 and rows[d]["norm"] >= rows[0]["norm"]]
        print(
            f"CDRDI_DEPTH_SUMMARY policy={policy} mode={mode} best_depth={best_depth} "
            f"first_harmful_tested_depth={min(harmful) if harmful else None} "
            f"seed_reduction={rows[0]['reduction_vs_identity_pct']:+.3f}% "
            f"best_residual_reduction={rows[best_depth]['reduction_vs_seed_pct']:+.3f}%"
        )

    # Enforce comparison over identical validation tiles, masks and physical seed.
    first = all_rows[combinations[0]]
    for choice in combinations[1:]:
        other = all_rows[choice]
        for control in ("identity", 0):
            if abs(other[control]["norm"] - first[control]["norm"]) > 1e-7:
                raise RuntimeError(f"Control changed across {choice}: step={control}")
        for d in depths:
            if other[d]["valid_lr_pixels"] != first[d]["valid_lr_pixels"]:
                raise RuntimeError(f"Validation pixels changed across {choice}: step={d}")

    # The guarded policy must be non-worsening for the exact metric used
    # by its acceptance rule (local standardized observable closure).
    for policy, mode in combinations:
        if policy == "closure_backtrack":
            scores = [all_rows[(policy, mode)][d]["norm"] for d in depths]
            if any(b > a + 1e-6 for a, b in zip(scores, scores[1:])):
                raise RuntimeError(f"Guarded physical closure unexpectedly increased: {mode}")

    _write_csv(args.output_csv, combined)
    print(f"CDRDI_DEPTH_CSV={os.path.abspath(args.output_csv)}")
    if args.per_tile_csv:
        _write_csv(args.per_tile_csv, all_tile_rows)
        print(f"CDRDI_TILE_CSV={os.path.abspath(args.per_tile_csv)}")

    if len(combinations) > 1:
        for d in (depth for depth in depths if depth > 0):
            ranked = sorted(combinations, key=lambda key: all_rows[key][d]["norm"])
            best = ranked[0]
            print(
                f"CDRDI_GUARD_RANK step={d} "
                f"best_policy={best[0]} best_mode={best[1]} "
                f"best_norm={all_rows[best][d]['norm']:.8f} "
                f"best_residual_reduction={all_rows[best][d]['reduction_vs_seed_pct']:+.3f}% "
                f"ranking={','.join(p + ':' + m for p, m in ranked)}"
            )
    print(
        "CDRDI_GUARD_NOTE=only the specified masked normalized closure is "
        "protected; this is not evidence of improved registration or 30m HSI reconstruction"
    )

if __name__ == "__main__":
    main()
