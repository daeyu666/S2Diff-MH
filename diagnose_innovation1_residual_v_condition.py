"""Inspect residual-DDPM state/condition scale and multiscale feature balance."""

from __future__ import annotations

import argparse

import torch

from data_loader import build_train_val_test_loaders
from diffusion import GaussianDDPMSchedule
from innovation1 import build_progressive_process
from train_innovation1_ddpm import build_condition, build_data_config, resolve_device
from train_innovation1_residual_ddpm import (
    load_saved_normalizer,
    normalize_residual,
)
from train_innovation1_residual_v_ddpm import (
    build_v_model,
    validate_v_checkpoint,
)
from utils import load_checkpoint, set_seed


def describe(name: str, value: torch.Tensor) -> float:
    x = value.detach().float()
    rms = x.square().mean().sqrt().item()
    print(
        f"{name:24s} shape={tuple(x.shape)} "
        f"mean={x.mean().item():+.5f} std={x.std(unbiased=False).item():.5f} "
        f"rms={rms:.5f} min={x.min().item():+.4f} max={x.max().item():+.4f}"
    )
    return rms


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
    validate_v_checkpoint(args.checkpoint, args)

    cfg = build_data_config(args)
    _, val_loader, _, info = build_train_val_test_loaders(cfg)
    process = build_progressive_process(cfg)
    schedule = GaussianDDPMSchedule(
        total_steps=args.total_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    model = build_v_model(args, info, device)
    epoch, best = load_checkpoint(
        model, args.checkpoint, map_location=str(device)
    )
    residual_mean, residual_scale = load_saved_normalizer(
        args.checkpoint, device
    )
    model.eval()

    batch = next(iter(val_loader))
    gt = batch["gt"].to(device, non_blocking=True)
    hr_msi = batch["hr_msi"].to(device, non_blocking=True)
    condition, baseline = build_condition(gt, hr_msi, process)
    residual_gt = gt - baseline
    residual_norm, clip_fraction = normalize_residual(
        residual_gt, residual_mean, residual_scale
    )

    n_hsi = int(info["n_bands"])
    print(
        f"checkpoint={args.checkpoint} epoch={epoch} best_PSNR={best:.6f} "
        f"T={schedule.total_steps} alpha_bar_T={schedule.alpha_bars[-1].item():.8e}"
    )
    print(f"train-normalizer clip fraction on this validation batch={clip_fraction.item():.6%}")
    print("\n[1] INPUT DISTRIBUTIONS")
    describe("GT [0,1]", gt)
    describe("baseline [0,1]", baseline)
    describe("residual raw", residual_gt)
    describe("residual normalized", residual_norm)
    describe("condition all [-1,1]", condition)
    describe("condition LR-HSI", condition[:, :n_hsi])
    describe("condition HR-MSI", condition[:, n_hsi:])

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 50505)
    fixed_noise = torch.randn(
        residual_norm.shape, device=device, dtype=residual_norm.dtype,
        generator=generator,
    )
    selected = [1, 50, 100, 250, 500, 750, 1000]
    selected = [t for t in selected if t <= schedule.total_steps]

    print("\n[2] STATE/CONDITION INPUT AND FEATURE BALANCE")
    with torch.no_grad():
        for timestep in selected:
            tb = torch.full(
                (gt.shape[0],), timestep, device=device, dtype=torch.long
            )
            xt = schedule.q_sample(residual_norm, tb, fixed_noise)
            time_emb = model._time_embedding(tb, gt.shape[0], device)

            state_spectral = model.spectral_stem(xt)
            state1 = model.in_proj(state_spectral)
            cond1 = model.msi_in(condition)
            fused1 = state1 + cond1
            skip1 = model._apply_blocks(fused1, model.enc1, time_emb)

            state2 = model.down1(skip1)
            cond2 = model.msi_down1(cond1)
            fused2 = state2 + cond2
            skip2 = model._apply_blocks(fused2, model.enc2, time_emb)

            state3 = model.down2(skip2)
            cond3 = model.msi_down2(cond2)

            print(f"\n--- timestep={timestep}, alpha_bar={schedule.alpha_bars[timestep - 1].item():.8e} ---")
            describe("x_t", xt)
            describe("state stem", state_spectral)
            state1_rms = describe("state feature L1", state1)
            cond1_rms = describe("condition feature L1", cond1)
            print(f"fusion L1 condition/state RMS ratio={cond1_rms / max(state1_rms, 1e-12):.5f}")

            state2_rms = describe("state feature L2", state2)
            cond2_rms = describe("condition feature L2", cond2)
            print(f"fusion L2 condition/state RMS ratio={cond2_rms / max(state2_rms, 1e-12):.5f}")

            state3_rms = describe("state feature L3", state3)
            cond3_rms = describe("condition feature L3", cond3)
            print(f"fusion L3 condition/state RMS ratio={cond3_rms / max(state3_rms, 1e-12):.5f}")


if __name__ == "__main__":
    main()
