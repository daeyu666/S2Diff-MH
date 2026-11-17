"""Diagnose residual-conditioned Gaussian DDPM v-prediction reconstruction.

Separates:
- deterministic LR-HSI baseline quality;
- fixed-timestep v-prediction and x0 reconstruction;
- residual generation through the complete ancestral reverse chain.

Uses the train-split residual statistics stored in the v-prediction checkpoint.
"""

from __future__ import annotations

import argparse

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
from train_innovation1_residual_v_ddpm import (
    build_v_model,
    validate_v_checkpoint,
)
from train_innovation1_residual_ddpm import (
    denormalize_residual,
    load_saved_normalizer,
    normalize_residual,
)
from utils import load_checkpoint, set_seed


def stats(name, x):
    x = x.detach().float()
    print(
        f"{name}: shape={tuple(x.shape)} "
        f"min={x.min().item():.6f} max={x.max().item():.6f} "
        f"mean={x.mean().item():.6f} std={x.std(unbiased=False).item():.6f} "
        f"absmax={x.abs().max().item():.6f}"
    )


def corrcoef(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    den = torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(1e-12)
    return (a * b).sum().div(den).item()


def report_reconstruction(tag, baseline, residual_pred, gt, residual_gt):
    pred = (baseline + residual_pred).clamp(0.0, 1.0)
    m = calc_metrics(pred, gt, 4)
    residual_rmse = torch.sqrt(F.mse_loss(residual_pred, residual_gt)).item()
    residual_corr = corrcoef(residual_pred, residual_gt)
    print(
        f"{tag}: PSNR={m['PSNR']:.6f} SAM={m['SAM']:.6f} "
        f"RMSE={m['RMSE']:.6f} ERGAS={m['ERGAS']:.6f} "
        f"SSIM={m['SSIM']:.6f} CC={m['CC']:.6f} "
        f"residual_RMSE={residual_rmse:.6f} "
        f"residual_corr={residual_corr:.6f}"
    )


def main():
    p = argparse.ArgumentParser()
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
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    cfg = build_data_config(args)
    _, val_loader, _, info = build_train_val_test_loaders(cfg)

    process = build_progressive_process(cfg)
    schedule = GaussianDDPMSchedule(
        total_steps=args.total_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    validate_v_checkpoint(args.checkpoint)
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
    print(
        f"normalizer: mean_abs_avg={residual_mean.abs().mean().item():.6e} "
        f"scale_avg={residual_scale.mean().item():.6e}"
    )

    batch = next(iter(val_loader))
    gt = batch["gt"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    condition, baseline = build_condition(gt, hr_msi, process)
    residual_gt = gt - baseline
    residual_norm, clip_fraction = normalize_residual(
        residual_gt, residual_mean, residual_scale
    )

    print("\n[1] BASELINE AND RESIDUAL STATISTICS")
    stats("GT_[0,1]", gt)
    stats("baseline_[0,1]", baseline)
    stats("condition_[-1,1]", condition)
    stats("true_residual", residual_gt)
    stats("true_normalized_residual", residual_norm)
    print(f"val_residual_clip_fraction={clip_fraction.item():.6%}")
    report_reconstruction(
        "BASELINE_ONLY", baseline, torch.zeros_like(residual_gt), gt, residual_gt
    )
    mean_only = residual_mean.expand_as(residual_gt)
    report_reconstruction(
        "BASELINE_PLUS_TRAIN_RESIDUAL_MEAN",
        baseline,
        mean_only,
        gt,
        residual_gt,
    )
    report_reconstruction(
        "ORACLE_TRUE_RESIDUAL",
        baseline,
        residual_gt,
        gt,
        residual_gt,
    )

    selected_ts = [1000, 750, 500, 250, 100, 50, 10, 1]
    selected_ts = [t for t in selected_ts if t <= schedule.total_steps]
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 40404)
    fixed_noise = torch.randn(
        residual_norm.shape,
        device=device,
        dtype=residual_norm.dtype,
        generator=generator,
    )

    print("\n[2] FIXED-T V-PREDICTION AND X0 DIAGNOSTICS")
    with torch.no_grad():
        for t in selected_ts:
            tb = torch.full(
                (gt.shape[0],), t, device=device, dtype=torch.long
            )
            xt = schedule.q_sample(residual_norm, tb, fixed_noise)
            target_v = v_prediction_target(
                residual_norm, fixed_noise, tb, schedule
            )
            predicted_v = model(xt, condition, tb)
            v_mse = F.mse_loss(predicted_v, target_v).item()
            v_corr = corrcoef(predicted_v, target_v)
            eps_pred = v_to_epsilon(xt, predicted_v, tb, schedule)
            eps_mse = F.mse_loss(eps_pred, fixed_noise).item()
            x0_raw = v_to_x0(xt, predicted_v, tb, schedule)
            x0_clipped = x0_raw.clamp(-1.0, 1.0)
            clip_x0_fraction = (
                (x0_raw < -1.0) | (x0_raw > 1.0)
            ).float().mean().item()
            residual_est = denormalize_residual(
                x0_clipped, residual_mean, residual_scale
            )
            rec = (baseline + residual_est).clamp(0.0, 1.0)
            m = calc_metrics(rec, gt, args.scale_ratio)
            r_rmse = torch.sqrt(
                F.mse_loss(residual_est, residual_gt)
            ).item()
            r_corr = corrcoef(residual_est, residual_gt)
            print(
                f"t={t:4d} v_mse={v_mse:.7f} v_corr={v_corr:.5f} "
                f"v_std={predicted_v.std(unbiased=False).item():.5f} "
                f"reconstructed_eps_mse={eps_mse:.7f} "
                f"eps_std={eps_pred.std(unbiased=False).item():.5f} "
                f"x0_norm_std={x0_clipped.std(unbiased=False).item():.5f} "
                f"x0_clip_fraction={clip_x0_fraction:.5f} "
                f"residual_RMSE={r_rmse:.6f} residual_corr={r_corr:.5f} "
                f"PSNR={m['PSNR']:.4f} SAM={m['SAM']:.4f}"
            )

    print("\n[3] FULL ANCESTRAL REVERSE-CHAIN TRACE")
    with torch.no_grad():
        x_t = torch.randn(
            residual_norm.shape,
            device=device,
            dtype=residual_norm.dtype,
            generator=generator,
        )
        trace = set(selected_ts)
        for t in range(schedule.total_steps, 0, -1):
            tb = torch.full(
                (gt.shape[0],), t, device=device, dtype=torch.long
            )
            predicted_v = model(x_t, condition, tb)
            x0_raw = v_to_x0(x_t, predicted_v, tb, schedule)
            eps_pred = v_to_epsilon(x_t, predicted_v, tb, schedule)
            x0 = x0_raw.clamp(-1.0, 1.0)

            if t in trace:
                residual_est = denormalize_residual(
                    x0, residual_mean, residual_scale
                )
                rec = (baseline + residual_est).clamp(0.0, 1.0)
                m = calc_metrics(rec, gt, args.scale_ratio)
                r_rmse = torch.sqrt(
                    F.mse_loss(residual_est, residual_gt)
                ).item()
                r_corr = corrcoef(residual_est, residual_gt)
                print(
                    f"reverse_t={t:4d} xt_std={x_t.std(unbiased=False).item():.5f} "
                    f"v_std={predicted_v.std(unbiased=False).item():.5f} "
                    f"eps_std={eps_pred.std(unbiased=False).item():.5f} "
                    f"x0_std={x0.std(unbiased=False).item():.5f} "
                    f"x0_clip_fraction={((x0_raw < -1) | (x0_raw > 1)).float().mean().item():.5f} "
                    f"residual_RMSE={r_rmse:.6f} residual_corr={r_corr:.5f} "
                    f"PSNR={m['PSNR']:.4f} SAM={m['SAM']:.4f}"
                )

            c0 = schedule.posterior_mean_coef_x0[t - 1].to(
                device=device, dtype=gt.dtype
            )
            ct = schedule.posterior_mean_coef_xt[t - 1].to(
                device=device, dtype=gt.dtype
            )
            posterior_mean = c0 * x0 + ct * x_t
            if t > 1:
                variance = schedule.posterior_variance[t - 1].to(
                    device=device, dtype=gt.dtype
                )
                z = torch.randn(
                    residual_norm.shape,
                    device=device,
                    dtype=residual_norm.dtype,
                    generator=generator,
                )
                x_t = posterior_mean + variance.clamp_min(0.0).sqrt() * z
            else:
                x_t = posterior_mean

        final_residual = denormalize_residual(
            x_t, residual_mean, residual_scale
        )
        stats("generated_normalized_residual", x_t)
        stats("generated_residual", final_residual)
        report_reconstruction(
            "FINAL_ANCESTRAL",
            baseline,
            final_residual,
            gt,
            residual_gt,
        )


if __name__ == "__main__":
    main()
