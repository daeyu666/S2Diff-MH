"""Core diagnostics for Innovation 1 only."""

from __future__ import annotations

import torch

from innovation1 import build_progressive_process, evaluate
from metrics import calc_metrics
from models import CleanHSIPredictor, RawMSIDirectPredictor, SpectralSpatialCleanHSIPredictor
from utils import load_checkpoint


def _build_model(cfg, info, device):
    common = dict(
        n_bands=info["n_bands"],
        total_steps=cfg.diffusion_steps,
        base_channels=cfg.base_channels,
        time_dim=cfg.time_dim,
        dropout=cfg.dropout,
        residual_prediction=True,
    )
    if cfg.predictor == "v1":
        model = CleanHSIPredictor(**common)
    elif cfg.predictor == "v2":
        model = SpectralSpatialCleanHSIPredictor(
            **common,
            spectral_hidden=cfg.spectral_hidden,
        )
    elif cfg.predictor == "raw_direct":
        model = RawMSIDirectPredictor(
            **common,
            n_msi_bands=info["n_msi_bands"],
            spectral_hidden=cfg.spectral_hidden,
        )
    else:
        raise ValueError(cfg.predictor)
    return model.to(device)


def _format_metrics(metrics):
    keys = ["PSNR", "SAM", "RMSE", "ERGAS", "SSIM", "CC", "INIT_PSNR", "INIT_SAM"]
    return " ".join(f"{key}={metrics[key]:.6f}" for key in keys if key in metrics)


@torch.no_grad()
def run_diagnosis(cfg, test_loader, info, device):
    process = build_progressive_process(cfg)
    model = _build_model(cfg, info, device)
    if cfg.resume:
        load_checkpoint(model, cfg.resume, map_location=str(device), strict=True)
        model.eval()

    batch = next(iter(test_loader))
    gt = batch["gt"].to(device)
    process.assert_terminal_closure(gt)
    print("terminal_closure: PASS")
    print("t scale strength PSNR SAM mean")
    for t in range(process.total_steps + 1):
        state = process.state(t)
        lifted = process.state_at(gt, t)
        metrics = calc_metrics(lifted, gt, cfg.scale_ratio)
        print(
            f"{t:02d} {state.scale:d} {state.strength:.4f} "
            f"{metrics['PSNR']:.4f} {metrics['SAM']:.4f} {lifted.mean().item():.6f}"
        )

    if cfg.resume:
        metrics = evaluate(model, test_loader, process, device, scale_ratio=cfg.scale_ratio)
        print("full_reverse", _format_metrics(metrics))
