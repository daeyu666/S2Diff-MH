"""Diagnose unit-variance residual-conditioned Gaussian DDPM v-prediction.

The normalized residual is not clipped, and x0 estimates are kept unbounded
throughout the reverse chain. Only final reflectance reconstruction is clamped.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from data_loader import build_train_val_test_loaders
from diffusion import GaussianDDPMSchedule, ddim_v_sample
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
    den = torch.sqrt(a.square().sum() * b.square().sum())
    if den.item() < 1e-12:
        return float("nan")
    return (a * b).sum().div(den).item()


def per_band_corrcoef(a, b):
    if a.ndim != 4 or b.shape != a.shape:
        raise ValueError("Per-band residual correlation expects matching [B,C,H,W] tensors")
    dims = (0, 2, 3)
    a_centered = a.float() - a.float().mean(dim=dims, keepdim=True)
    b_centered = b.float() - b.float().mean(dim=dims, keepdim=True)
    numerator = (a_centered * b_centered).sum(dim=dims)
    denominator = torch.sqrt(
        a_centered.square().sum(dim=dims)
        * b_centered.square().sum(dim=dims)
    )
    valid = denominator > 1e-12
    if not valid.any():
        return float("nan")
    return (numerator[valid] / denominator[valid]).mean().item()


def residual_diagnostics(residual_pred, residual_gt):
    pred = residual_pred.detach().float()
    target = residual_gt.detach().float()
    target_std = target.std(unbiased=False)
    pred_std = pred.std(unbiased=False)
    return {
        "rmse": torch.sqrt(F.mse_loss(pred, target)).item(),
        "corr_global": corrcoef(pred, target),
        "corr_band_mean": per_band_corrcoef(pred, target),
        "std_ratio": (pred_std / target_std.clamp_min(1e-12)).item(),
        "mean_bias": (pred.mean() - target.mean()).item(),
    }


def report_reconstruction(tag, baseline, residual_pred, gt, residual_gt):
    pred = (baseline + residual_pred).clamp(0.0, 1.0)
    m = calc_metrics(pred, gt, 4)
    r = residual_diagnostics(residual_pred, residual_gt)
    print(
        f"{tag}: PSNR={m['PSNR']:.6f} SAM={m['SAM']:.6f} "
        f"RMSE={m['RMSE']:.6f} ERGAS={m['ERGAS']:.6f} "
        f"SSIM={m['SSIM']:.6f} CC={m['CC']:.6f} "
        f"residual_RMSE={r['rmse']:.6f} "
        f"residual_corr_global={r['corr_global']:.6f} "
        f"residual_corr_band_mean={r['corr_band_mean']:.6f} "
        f"residual_std_ratio={r['std_ratio']:.6f} "
        f"residual_mean_bias={r['mean_bias']:.6f}"
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
    p.add_argument("--ddim_step_list", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--diagnostic_timestep", type=int, default=250)
    p.add_argument("--residual_scale_factor", type=float, default=1.0)
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
    print(f"val_residual_out_of_range_fraction={clip_fraction.item():.6%}")
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
    fixed_noise_generator = torch.Generator(device=device)
    fixed_noise_generator.manual_seed(args.seed + 40404)
    fixed_noise = torch.randn(
        residual_norm.shape,
        device=device,
        dtype=residual_norm.dtype,
        generator=fixed_noise_generator,
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
            x0_outlier_fraction = (
                (x0_raw < -1.0) | (x0_raw > 1.0)
            ).float().mean().item()
            residual_est = denormalize_residual(
                x0_raw, residual_mean, residual_scale
            )
            rec = (baseline + residual_est).clamp(0.0, 1.0)
            m = calc_metrics(rec, gt, args.scale_ratio)
            r = residual_diagnostics(residual_est, residual_gt)
            print(
                f"t={t:4d} v_mse={v_mse:.7f} v_corr={v_corr:.5f} "
                f"v_std={predicted_v.std(unbiased=False).item():.5f} "
                f"reconstructed_eps_mse={eps_mse:.7f} "
                f"eps_std={eps_pred.std(unbiased=False).item():.5f} "
                f"x0_norm_std={x0_raw.std(unbiased=False).item():.5f} "
                f"x0_out_of_range_fraction={x0_outlier_fraction:.5f} "
                f"residual_RMSE={r['rmse']:.6f} "
                f"corr_global={r['corr_global']:.5f} "
                f"corr_band_mean={r['corr_band_mean']:.5f} "
                f"std_ratio={r['std_ratio']:.5f} "
                f"mean_bias={r['mean_bias']:.5f} "
                f"PSNR={m['PSNR']:.4f} SAM={m['SAM']:.4f}"
            )

    print("\n[3] FULL ANCESTRAL REVERSE-CHAIN TRACE")
    free_running_state_at_diagnostic_t = None
    with torch.no_grad():
        chain_seed = args.seed + 50505
        ancestral_generator = torch.Generator(device=device)
        ancestral_generator.manual_seed(chain_seed)
        x_t = torch.randn(
            residual_norm.shape,
            device=device,
            dtype=residual_norm.dtype,
            generator=ancestral_generator,
        )
        trace = set(selected_ts)
        for t in range(schedule.total_steps, 0, -1):
            tb = torch.full(
                (gt.shape[0],), t, device=device, dtype=torch.long
            )
            predicted_v = model(x_t, condition, tb)
            x0_raw = v_to_x0(x_t, predicted_v, tb, schedule)
            eps_pred = v_to_epsilon(x_t, predicted_v, tb, schedule)
            x0 = x0_raw
            x0_outlier_fraction = (
                (x0_raw < -1.0) | (x0_raw > 1.0)
            ).float().mean().item()

            if t == args.diagnostic_timestep:
                free_running_state_at_diagnostic_t = x_t.detach().clone()

            if t in trace:
                residual_est = denormalize_residual(
                    x0, residual_mean, residual_scale
                )
                rec = (baseline + residual_est).clamp(0.0, 1.0)
                m = calc_metrics(rec, gt, args.scale_ratio)
                r = residual_diagnostics(residual_est, residual_gt)
                print(
                    f"reverse_t={t:4d} xt_std={x_t.std(unbiased=False).item():.5f} "
                    f"v_std={predicted_v.std(unbiased=False).item():.5f} "
                    f"eps_std={eps_pred.std(unbiased=False).item():.5f} "
                    f"x0_std={x0.std(unbiased=False).item():.5f} "
                    f"x0_out_of_range_fraction={x0_outlier_fraction:.5f} "
                    f"residual_RMSE={r['rmse']:.6f} "
                    f"corr_global={r['corr_global']:.5f} "
                    f"corr_band_mean={r['corr_band_mean']:.5f} "
                    f"std_ratio={r['std_ratio']:.5f} "
                    f"mean_bias={r['mean_bias']:.5f} "
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
                    generator=ancestral_generator,
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


    diagnostic_t = int(args.diagnostic_timestep)
    if not 1 <= diagnostic_t <= schedule.total_steps:
        raise ValueError("--diagnostic_timestep outside diffusion schedule")
    if free_running_state_at_diagnostic_t is None:
        raise RuntimeError(f"Missing free-running state at t={diagnostic_t}")
    print(f"\n[3B] TEACHER-FORCED VS FREE-RUNNING AT t={diagnostic_t}")
    with torch.no_grad():
        diagnostic_tb = torch.full((gt.shape[0],), diagnostic_t, device=device, dtype=torch.long)
        teacher_xt = schedule.q_sample(residual_norm, diagnostic_tb, fixed_noise)
        teacher_v = model(teacher_xt, condition, diagnostic_tb)
        teacher_x0 = v_to_x0(teacher_xt, teacher_v, diagnostic_tb, schedule)
        teacher_res = denormalize_residual(teacher_x0, residual_mean, residual_scale)
        report_reconstruction(f"TEACHER_FORCED_t{diagnostic_t}", baseline, teacher_res, gt, residual_gt)

        free_xt = free_running_state_at_diagnostic_t
        free_v = model(free_xt, condition, diagnostic_tb)
        free_x0 = v_to_x0(free_xt, free_v, diagnostic_tb, schedule)
        free_res = denormalize_residual(free_x0, residual_mean, residual_scale)
        report_reconstruction(f"FREE_RUNNING_t{diagnostic_t}", baseline, free_res, gt, residual_gt)
        stats("teacher_forced_xt", teacher_xt)
        stats("free_running_xt", free_xt)
        print(
            "state_distance: "
            f"rmse={torch.sqrt(F.mse_loss(free_xt, teacher_xt)).item():.6f} "
            f"corr={corrcoef(free_xt, teacher_xt):.6f} "
            f"std_ratio={(free_xt.std(unbiased=False) / teacher_xt.std(unbiased=False).clamp_min(1e-12)).item():.6f}"
        )

    print("\n[4] SPARSE DETERMINISTIC DDIM STEP-COUNT SWEEP")
    if not 0.0 <= args.ddim_eta <= 1.0:
        raise ValueError("--ddim_eta must lie in [0,1]")
    step_counts = sorted(set(int(v) for v in args.ddim_step_list))
    if not step_counts or any(v < 1 or v > schedule.total_steps for v in step_counts):
        raise ValueError("--ddim_step_list values must be within [1,T]")

    init_gen = torch.Generator(device=device)
    init_gen.manual_seed(chain_seed)
    shared_xT = torch.randn(residual_norm.shape, device=device, dtype=residual_norm.dtype, generator=init_gen)
    sweep_rows = []
    with torch.no_grad():
        for n_steps in step_counts:
            ts = torch.linspace(schedule.total_steps, 0, n_steps + 1, dtype=torch.float64).round().long().tolist()
            if any(a <= b for a, b in zip(ts[:-1], ts[1:])):
                raise RuntimeError(f"Non-descending DDIM timestep list for {n_steps} steps")
            x = shared_xT.clone()
            for t, prev_t in zip(ts[:-1], ts[1:]):
                tb = torch.full((gt.shape[0],), t, device=device, dtype=torch.long)
                v = model(x, condition, tb)
                x0 = v_to_x0(x, v, tb, schedule)
                eps = v_to_epsilon(x, v, tb, schedule)
                if t in set(selected_ts):
                    r_now = denormalize_residual(x0, residual_mean, residual_scale)
                    m_now = calc_metrics((baseline + r_now).clamp(0, 1), gt, args.scale_ratio)
                    d_now = residual_diagnostics(r_now, residual_gt)
                    print(
                        f"ddim_steps={n_steps:2d} t={t:4d} "
                        f"PSNR={m_now['PSNR']:.4f} SAM={m_now['SAM']:.4f} "
                        f"residual_RMSE={d_now['rmse']:.6f} "
                        f"corr_band_mean={d_now['corr_band_mean']:.5f} "
                        f"std_ratio={d_now['std_ratio']:.5f}"
                    )
                ab_t = schedule.alpha_bars[t - 1].to(device=device, dtype=gt.dtype)
                ab_prev = torch.ones((), device=device, dtype=gt.dtype) if prev_t == 0 else schedule.alpha_bars[prev_t - 1].to(device=device, dtype=gt.dtype)
                sigma2 = (
                    args.ddim_eta ** 2 * (1 - ab_prev)
                    / (1 - ab_t).clamp_min(1e-12)
                    * (1 - ab_t / ab_prev.clamp_min(1e-12))
                ).clamp_min(0)
                direction_var = (1 - ab_prev - sigma2).clamp_min(0)
                x_prev = ab_prev.sqrt() * x0 + direction_var.sqrt() * eps
                if args.ddim_eta > 0 and prev_t > 0:
                    z_gen = torch.Generator(device=device)
                    z_gen.manual_seed(chain_seed + n_steps * 100003 + t)
                    z = torch.randn(x.shape, device=device, dtype=x.dtype, generator=z_gen)
                    x_prev = x_prev + sigma2.sqrt() * z
                x = x_prev

            residual_final = denormalize_residual(x, residual_mean, residual_scale)
            metrics = calc_metrics((baseline + residual_final).clamp(0, 1), gt, args.scale_ratio)
            diag = residual_diagnostics(residual_final, residual_gt)
            sweep_rows.append((n_steps, metrics["PSNR"], metrics["SAM"], diag["rmse"], diag["corr_band_mean"], diag["std_ratio"], diag["mean_bias"]))
            report_reconstruction(f"FINAL_DDIM_steps{n_steps}_eta{args.ddim_eta:g}", baseline, residual_final, gt, residual_gt)

    print("\n[5] SPARSE DDIM SUMMARY (shared initial x_T)")
    print("steps,PSNR,SAM,residual_RMSE,corr_band_mean,std_ratio,mean_bias")
    for row in sweep_rows:
        print(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},{row[6]:.6f}")



if __name__ == "__main__":
    main()
