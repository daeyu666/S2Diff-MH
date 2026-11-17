"""Evaluate sparse deterministic DDIM and teacher-forced/free-running mismatch.

All DDIM step counts share exactly the same initial Gaussian noise and the same
validation batch. The full 1000-step trajectory is also used to capture the
free-running state at a selected timestep (default t=250).
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
import torch.nn.functional as F

from data_loader import build_train_val_test_loaders
from diffusion import GaussianDDPMSchedule
from diffusion.gaussian_v_ddpm import (
    v_prediction_target,
    v_to_epsilon,
    v_to_x0,
)
from innovation1 import build_progressive_process
from metrics import calc_metrics
from train_innovation1_ddpm import (
    build_condition,
    build_data_config,
    resolve_device,
)
from train_innovation1_residual_v_unitvar_ddpm import (
    build_v_model,
    denormalize_residual,
    load_saved_normalizer,
    normalize_residual,
    validate_unitvar_checkpoint,
)
from utils import load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Sparse DDIM sweep and teacher-forced/free-running diagnostics"
    )
    p.add_argument("--dataset", default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--total_diffusion_steps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument(
        "--ddim_steps_list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 1000],
    )
    p.add_argument("--capture_timestep", type=int, default=250)
    p.add_argument("--residual_scale_factor", type=float, default=1.0)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output_csv",
        default="./outputs/innovation1_sparse_ddim/PaviaU_sparse_ddim.csv",
    )
    return p.parse_args()


def global_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    den = torch.sqrt(a.square().sum() * b.square().sum())
    if den.item() < 1e-12:
        return float("nan")
    return (a * b).sum().div(den).item()


def band_mean_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.ndim != 4 or b.shape != a.shape:
        raise ValueError("Expected matching [B,C,H,W] residual tensors")
    dims = (0, 2, 3)
    ac = a.float() - a.float().mean(dim=dims, keepdim=True)
    bc = b.float() - b.float().mean(dim=dims, keepdim=True)
    numerator = (ac * bc).sum(dim=dims)
    denominator = torch.sqrt(ac.square().sum(dim=dims) * bc.square().sum(dim=dims))
    valid = denominator > 1e-12
    if not valid.any():
        return float("nan")
    return (numerator[valid] / denominator[valid]).mean().item()


def residual_stats(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.detach().float()
    target = target.detach().float()
    return {
        "residual_rmse": torch.sqrt(F.mse_loss(pred, target)).item(),
        "residual_corr_global": global_corr(pred, target),
        "residual_corr_band_mean": band_mean_corr(pred, target),
        "residual_std_ratio": (
            pred.std(unbiased=False)
            / target.std(unbiased=False).clamp_min(1e-12)
        ).item(),
        "residual_mean_bias": (pred.mean() - target.mean()).item(),
    }


def reconstruction_metrics(
    baseline: torch.Tensor,
    residual_pred: torch.Tensor,
    gt: torch.Tensor,
    residual_gt: torch.Tensor,
    scale_ratio: int,
):
    prediction = (baseline + residual_pred).clamp(0.0, 1.0)
    metrics = calc_metrics(prediction, gt, scale_ratio)
    metrics.update(residual_stats(residual_pred, residual_gt))
    return metrics


def print_metrics(tag: str, metrics: dict):
    print(
        f"{tag}: PSNR={metrics['PSNR']:.6f} SAM={metrics['SAM']:.6f} "
        f"RMSE={metrics['RMSE']:.6f} ERGAS={metrics['ERGAS']:.6f} "
        f"SSIM={metrics['SSIM']:.6f} CC={metrics['CC']:.6f} "
        f"residual_RMSE={metrics['residual_rmse']:.6f} "
        f"corr_global={metrics['residual_corr_global']:.6f} "
        f"corr_band_mean={metrics['residual_corr_band_mean']:.6f} "
        f"std_ratio={metrics['residual_std_ratio']:.6f} "
        f"mean_bias={metrics['residual_mean_bias']:.6e}"
    )


def make_timestep_sequence(total_steps: int, inference_steps: int):
    if inference_steps < 1 or inference_steps > total_steps:
        raise ValueError(
            f"inference_steps must be in [1, {total_steps}], got {inference_steps}"
        )
    sequence = torch.linspace(
        total_steps, 0, inference_steps + 1, dtype=torch.float64
    ).round().to(torch.long).tolist()
    if any(a <= b for a, b in zip(sequence[:-1], sequence[1:])):
        raise RuntimeError(
            f"Non-descending DDIM sequence for inference_steps={inference_steps}: "
            f"{sequence}"
        )
    return sequence


@torch.no_grad()
def run_ddim_trajectory(
    model,
    condition: torch.Tensor,
    initial_noise: torch.Tensor,
    schedule: GaussianDDPMSchedule,
    inference_steps: int,
    capture_timestep: int | None = None,
):
    """Run eta=0 DDIM from a supplied common x_T and optionally capture x_t."""
    sequence = make_timestep_sequence(schedule.total_steps, inference_steps)
    x_t = initial_noise.clone()
    captured_state = None

    for timestep, previous_timestep in zip(sequence[:-1], sequence[1:]):
        if timestep == capture_timestep:
            captured_state = x_t.detach().clone()

        t_batch = torch.full(
            (x_t.shape[0],), timestep, device=x_t.device, dtype=torch.long
        )
        predicted_v = model(x_t, condition, t_batch)
        predicted_x0 = v_to_x0(x_t, predicted_v, t_batch, schedule)
        predicted_epsilon = v_to_epsilon(
            x_t, predicted_v, t_batch, schedule
        )

        alpha_bar_t = schedule.alpha_bars[timestep - 1].to(
            device=x_t.device, dtype=x_t.dtype
        )
        if previous_timestep == 0:
            alpha_bar_previous = torch.ones(
                (), device=x_t.device, dtype=x_t.dtype
            )
        else:
            alpha_bar_previous = schedule.alpha_bars[
                previous_timestep - 1
            ].to(device=x_t.device, dtype=x_t.dtype)

        # Deterministic DDIM (eta=0).
        direction_variance = (1.0 - alpha_bar_previous).clamp_min(0.0)
        x_t = (
            alpha_bar_previous.sqrt() * predicted_x0
            + direction_variance.sqrt() * predicted_epsilon
        )

    if capture_timestep == 0:
        captured_state = x_t.detach().clone()
    return x_t, captured_state, sequence


@torch.no_grad()
def diagnose_state(
    label: str,
    x_t: torch.Tensor,
    effective_noise: torch.Tensor,
    model,
    condition: torch.Tensor,
    residual_gt_norm: torch.Tensor,
    residual_gt: torch.Tensor,
    baseline: torch.Tensor,
    gt: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_scale: torch.Tensor,
    schedule: GaussianDDPMSchedule,
    timestep: int,
    scale_ratio: int,
    reference_noise: torch.Tensor,
):
    t_batch = torch.full(
        (gt.shape[0],), timestep, device=gt.device, dtype=torch.long
    )
    target_v = v_prediction_target(
        residual_gt_norm, effective_noise, t_batch, schedule
    )
    predicted_v = model(x_t, condition, t_batch)
    predicted_x0_norm = v_to_x0(x_t, predicted_v, t_batch, schedule)
    predicted_epsilon = v_to_epsilon(
        x_t, predicted_v, t_batch, schedule
    )
    predicted_residual = denormalize_residual(
        predicted_x0_norm, residual_mean, residual_scale
    )
    metrics = reconstruction_metrics(
        baseline, predicted_residual, gt, residual_gt, scale_ratio
    )
    v_mse = F.mse_loss(predicted_v, target_v).item()
    eps_mse = F.mse_loss(predicted_epsilon, effective_noise).item()
    v_corr = global_corr(predicted_v, target_v)

    alpha_bar = schedule.alpha_bars[timestep - 1].to(
        device=gt.device, dtype=gt.dtype
    )
    x_t_stats = {
        "xt_mean": x_t.mean().item(),
        "xt_std": x_t.std(unbiased=False).item(),
        "xt_absmax": x_t.abs().max().item(),
        "effective_noise_mean": effective_noise.mean().item(),
        "effective_noise_std": effective_noise.std(unbiased=False).item(),
        "effective_noise_corr_init": global_corr(
            effective_noise, reference_noise
        ),
        "x0_out_of_range_fraction": (
            (predicted_x0_norm < -1.0) | (predicted_x0_norm > 1.0)
        ).float().mean().item(),
        "v_mse": v_mse,
        "v_corr": v_corr,
        "epsilon_mse": eps_mse,
    }
    print(
        f"{label} t={timestep}: "
        f"xt_mean={x_t_stats['xt_mean']:.6f} "
        f"xt_std={x_t_stats['xt_std']:.6f} "
        f"xt_absmax={x_t_stats['xt_absmax']:.6f} "
        f"effective_eps_mean={x_t_stats['effective_noise_mean']:.6f} "
        f"effective_eps_std={x_t_stats['effective_noise_std']:.6f} "
        f"effective_eps_corr_init={x_t_stats['effective_noise_corr_init']:.6f} "
        f"v_mse={v_mse:.7f} v_corr={v_corr:.6f} "
        f"epsilon_mse={eps_mse:.7f} "
        f"x0_out_of_range_fraction={x_t_stats['x0_out_of_range_fraction']:.6f}"
    )
    print_metrics(label, metrics)
    row = {"mode": label, "timestep": timestep, **x_t_stats, **metrics}
    return row


def save_rows(path: str, rows: list[dict]):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={path}")


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.capture_timestep < 1 or args.capture_timestep > args.total_diffusion_steps:
        raise ValueError(
            "--capture_timestep must be in [1, total_diffusion_steps]"
        )
    if abs(float(args.residual_scale_factor) - 1.0) > 1e-10:
        raise ValueError("This checkpoint requires residual_scale_factor=1.0")

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_cfg = build_data_config(args)
    _, val_loader, _, info = build_train_val_test_loaders(data_cfg)

    process = build_progressive_process(data_cfg)
    schedule = GaussianDDPMSchedule(
        total_steps=args.total_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    validate_unitvar_checkpoint(args.checkpoint, args)
    model = build_v_model(args, info, device)
    epoch, best = load_checkpoint(
        model, args.checkpoint, map_location=str(device)
    )
    residual_mean, residual_scale = load_saved_normalizer(
        args.checkpoint, device
    )
    model.eval()

    print(
        f"checkpoint={args.checkpoint} epoch={epoch} best_metric={best} "
        f"T={schedule.total_steps} alpha_bar_T={schedule.alpha_bars[-1].item():.8e}"
    )

    batch = next(iter(val_loader))
    gt = batch["gt"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    condition, baseline = build_condition(gt, hr_msi, process)
    residual_gt = gt - baseline
    residual_gt_norm, val_outlier_fraction = normalize_residual(
        residual_gt, residual_mean, residual_scale
    )
    print(
        f"validation_batch_shape={tuple(gt.shape)} "
        f"val_residual_out_of_range_fraction={val_outlier_fraction.item():.6%}"
    )

    # One initial Gaussian tensor is reused for every inference-step count.
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(args.seed + 50505)
    initial_noise = torch.randn(
        residual_gt_norm.shape,
        device=device,
        dtype=residual_gt_norm.dtype,
        generator=noise_generator,
    )
    print(
        f"common_initial_noise mean={initial_noise.mean().item():.6f} "
        f"std={initial_noise.std(unbiased=False).item():.6f}"
    )

    step_counts = list(dict.fromkeys(args.ddim_steps_list))
    if schedule.total_steps not in step_counts:
        step_counts.append(schedule.total_steps)
    for steps in step_counts:
        if steps < 1 or steps > schedule.total_steps:
            raise ValueError(f"Invalid DDIM step count: {steps}")
    step_counts = sorted(step_counts)

    print("\n[1] SPARSE DETERMINISTIC DDIM SWEEP (COMMON INITIAL NOISE)")
    rows = []
    free_running_state = None
    full_sequence = None
    for steps in step_counts:
        final_norm, captured, sequence = run_ddim_trajectory(
            model,
            condition,
            initial_noise,
            schedule,
            inference_steps=steps,
            capture_timestep=(
                args.capture_timestep if steps == schedule.total_steps else None
            ),
        )
        final_residual = denormalize_residual(
            final_norm, residual_mean, residual_scale
        )
        metrics = reconstruction_metrics(
            baseline, final_residual, gt, residual_gt, args.scale_ratio
        )
        out_of_range = (
            (final_norm < -1.0) | (final_norm > 1.0)
        ).float().mean().item()
        tag = f"DDIM_steps_{steps}"
        print_metrics(tag, metrics)
        print(
            f"  normalized_residual_std={final_norm.std(unbiased=False).item():.6f} "
            f"x0_out_of_range_fraction={out_of_range:.6%} "
            f"timestep_sequence={sequence if steps <= 16 else f'{sequence[:4]} ... {sequence[-3:]}'}"
        )
        rows.append({
            "mode": "sparse_ddim",
            "ddim_steps": steps,
            "x0_out_of_range_fraction": out_of_range,
            **metrics,
        })
        if steps == schedule.total_steps:
            free_running_state = captured
            full_sequence = sequence

    if free_running_state is None:
        raise RuntimeError(
            "Full-step DDIM trajectory was not run; cannot diagnose free-running state"
        )
    if free_running_state.shape != residual_gt_norm.shape:
        raise RuntimeError(
            f"Captured free-running state has invalid shape: {free_running_state.shape}"
        )

    print(
        f"\n[2] TEACHER-FORCED VS FREE-RUNNING AT t={args.capture_timestep}"
    )
    t = args.capture_timestep
    t_batch = torch.full(
        (gt.shape[0],), t, device=device, dtype=torch.long
    )
    alpha_bar = schedule.alpha_bars[t - 1].to(
        device=device, dtype=gt.dtype
    )
    teacher_forced_state = (
        alpha_bar.sqrt() * residual_gt_norm
        + (1.0 - alpha_bar).clamp_min(0.0).sqrt() * initial_noise
    )
    teacher_row = diagnose_state(
        "TEACHER_FORCED",
        teacher_forced_state,
        initial_noise,
        model,
        condition,
        residual_gt_norm,
        residual_gt,
        baseline,
        gt,
        residual_mean,
        residual_scale,
        schedule,
        t,
        args.scale_ratio,
        initial_noise,
    )

    effective_free_noise = (
        free_running_state - alpha_bar.sqrt() * residual_gt_norm
    ) / (1.0 - alpha_bar).clamp_min(1e-12).sqrt()
    free_row = diagnose_state(
        "FREE_RUNNING",
        free_running_state,
        effective_free_noise,
        model,
        condition,
        residual_gt_norm,
        residual_gt,
        baseline,
        gt,
        residual_mean,
        residual_scale,
        schedule,
        t,
        args.scale_ratio,
        initial_noise,
    )
    rows.extend([teacher_row, free_row])

    # Explicitly quantify how far the generated state departs from the
    # teacher-forced state produced with the same Gaussian noise realization.
    state_difference = free_running_state - teacher_forced_state
    print(
        f"state_gap_at_t{t}: rmse={torch.sqrt(state_difference.square().mean()).item():.6f} "
        f"mean={state_difference.mean().item():.6f} "
        f"std={state_difference.std(unbiased=False).item():.6f} "
        f"corr={global_corr(free_running_state, teacher_forced_state):.6f}"
    )

    save_rows(args.output_csv, rows)
    print("\nInterpretation guide:")
    print(
        "- Teacher-forced good + free-running poor: reverse-trajectory distribution shift."
    )
    print(
        "- Both poor: the model itself has insufficient prediction accuracy at this timestep."
    )
    print(
        "- Sparse DDIM improves as steps decrease: fewer self-generated transitions reduce error accumulation."
    )


if __name__ == "__main__":
    main()
