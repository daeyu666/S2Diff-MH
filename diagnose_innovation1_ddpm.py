"""Diagnose conditional Gaussian DDPM training and reverse sampling.

Reports:
1. Input/condition data ranges.
2. Epsilon prediction MSE at fixed diffusion timesteps.
3. Single-timestep x0 reconstruction quality.
4. Intermediate x0 estimates along the complete ancestral reverse chain.

This script does not modify checkpoints.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from data_loader import build_train_val_test_loaders
from diffusion import GaussianDDPMSchedule
from metrics import calc_metrics
from train_innovation1_ddpm import (
    build_condition,
    build_data_config,
    build_model,
    resolve_device,
)
from utils import load_checkpoint, set_seed


def stats(name: str, x: torch.Tensor) -> None:
    x = x.detach().float()
    print(
        f"{name}: shape={tuple(x.shape)} "
        f"min={x.min().item():.6f} max={x.max().item():.6f} "
        f"mean={x.mean().item():.6f} std={x.std(unbiased=False).item():.6f}"
    )


def normalized_to_reflectance(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def corrcoef_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt(torch.sum(a.square()) * torch.sum(b.square())).clamp_min(1e-12)
    return float((torch.sum(a * b) / denom).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--test_size", type=int, default=128)
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

    physical_process = __import__(
        "innovation1", fromlist=["build_progressive_process"]
    ).build_progressive_process(cfg)
    schedule = GaussianDDPMSchedule(
        total_steps=args.total_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    model = build_model(args, info, device)
    epoch, best = load_checkpoint(
        model, args.checkpoint, map_location=str(device)
    )
    model.eval()
    print(f"checkpoint={args.checkpoint} epoch={epoch} best_metric={best}")
    print(
        f"schedule=T{schedule.total_steps} beta=[{args.beta_start},{args.beta_end}] "
        f"alpha_bar_T={schedule.alpha_bars[-1].item():.8e}"
    )

    batch = next(iter(val_loader))
    gt = batch["gt"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    condition, lr_hsi_up = build_condition(gt, hr_msi, physical_process)
    x0 = gt.mul(2.0).sub(1.0)

    print("\n[1] DATA / CONDITION STATISTICS")
    stats("gt_[0,1]", gt)
    stats("x0_[-1,1]", x0)
    stats("hr_msi_[0,1]", hr_msi)
    stats("lr_hsi_up_[0,1]", lr_hsi_up)
    stats("condition_[-1,1]", condition)

    selected_ts = [1, 10, 50, 100, 250, 500, 750, 1000]
    selected_ts = [t for t in selected_ts if t <= schedule.total_steps]
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 30303)

    print("\n[2] FIXED-T EPSILON PREDICTION")
    with torch.no_grad():
        fixed_noise = torch.randn(
            x0.shape, device=device, dtype=x0.dtype, generator=generator
        )
        for t in selected_ts:
            t_batch = torch.full(
                (x0.shape[0],), t, dtype=torch.long, device=device
            )
            xt = schedule.q_sample(x0, t_batch, fixed_noise)
            eps_pred = model(xt, condition, t_batch)
            eps_mse = F.mse_loss(eps_pred, fixed_noise).item()
            eps_corr = corrcoef_flat(eps_pred, fixed_noise)

            alpha_bar = schedule.alpha_bars[t - 1].to(
                device=device, dtype=x0.dtype
            )
            x0_hat = (
                xt - (1.0 - alpha_bar).sqrt() * eps_pred
            ) / alpha_bar.sqrt().clamp_min(1e-8)
            x0_hat_clip = x0_hat.clamp(-1.0, 1.0)
            metrics = calc_metrics(
                normalized_to_reflectance(x0_hat_clip), gt, args.scale_ratio
            )
            clip_fraction = (
                (x0_hat < -1.0) | (x0_hat > 1.0)
            ).float().mean().item()
            print(
                f"t={t:4d} eps_mse={eps_mse:.7f} eps_corr={eps_corr:.5f} "
                f"x0hat_PSNR={metrics['PSNR']:.4f} "
                f"x0hat_SAM={metrics['SAM']:.4f} "
                f"x0hat_clip_fraction={clip_fraction:.4f} "
                f"eps_pred_std={eps_pred.std(unbiased=False).item():.5f}"
            )

    print("\n[3] FULL ANCESTRAL REVERSE-CHAIN TRACE")
    with torch.no_grad():
        x_t = torch.randn(
            x0.shape, device=device, dtype=x0.dtype, generator=generator
        )
        trace_ts = set(selected_ts)
        for t in range(schedule.total_steps, 0, -1):
            t_batch = torch.full(
                (x0.shape[0],), t, dtype=torch.long, device=device
            )
            eps_pred = model(x_t, condition, t_batch)
            alpha_bar = schedule.alpha_bars[t - 1].to(
                device=device, dtype=x0.dtype
            )
            x0_hat_raw = (
                x_t - (1.0 - alpha_bar).sqrt() * eps_pred
            ) / alpha_bar.sqrt().clamp_min(1e-8)
            x0_hat = x0_hat_raw.clamp(-1.0, 1.0)

            if t in trace_ts:
                metrics = calc_metrics(
                    normalized_to_reflectance(x0_hat), gt, args.scale_ratio
                )
                print(
                    f"reverse_t={t:4d} xt_std={x_t.std(unbiased=False).item():.5f} "
                    f"eps_std={eps_pred.std(unbiased=False).item():.5f} "
                    f"x0raw_absmax={x0_hat_raw.abs().max().item():.5f} "
                    f"clip_fraction={((x0_hat_raw < -1) | (x0_hat_raw > 1)).float().mean().item():.4f} "
                    f"x0hat_PSNR={metrics['PSNR']:.4f} "
                    f"x0hat_SAM={metrics['SAM']:.4f}"
                )

            coef_x0 = schedule.posterior_mean_coef_x0[t - 1].to(
                device=device, dtype=x0.dtype
            )
            coef_xt = schedule.posterior_mean_coef_xt[t - 1].to(
                device=device, dtype=x0.dtype
            )
            posterior_mean = coef_x0 * x0_hat + coef_xt * x_t
            if t > 1:
                variance = schedule.posterior_variance[t - 1].to(
                    device=device, dtype=x0.dtype
                )
                step_noise = torch.randn(
                    x0.shape, device=device, dtype=x0.dtype, generator=generator
                )
                x_t = posterior_mean + variance.clamp_min(0.0).sqrt() * step_noise
            else:
                x_t = posterior_mean

        final_pred = normalized_to_reflectance(x_t)
        final_metrics = calc_metrics(final_pred, gt, args.scale_ratio)
        print(
            f"FINAL PSNR={final_metrics['PSNR']:.6f} "
            f"SAM={final_metrics['SAM']:.6f} "
            f"RMSE={final_metrics['RMSE']:.6f}"
        )

    print("\n[4] REFERENCE: LR-HSI UPSAMPLE")
    init_metrics = calc_metrics(lr_hsi_up, gt, args.scale_ratio)
    print(
        f"INIT PSNR={init_metrics['PSNR']:.6f} "
        f"SAM={init_metrics['SAM']:.6f} "
        f"RMSE={init_metrics['RMSE']:.6f}"
    )


if __name__ == "__main__":
    main()
