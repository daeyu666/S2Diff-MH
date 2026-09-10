"""Core diagnostics for Innovation 1 only."""

from __future__ import annotations

import torch

from innovation1 import build_progressive_process
from main import build_model, format_metrics
from metrics import calc_metrics
from utils import load_checkpoint


@torch.no_grad()
def run_diagnosis(cfg, test_loader, info, device):
    process = build_progressive_process(cfg)
    model = build_model(cfg, info, device)
    if cfg.resume:
        load_checkpoint(model, cfg.resume, map_location=str(device), strict=True)
        model.eval()

    batch = next(iter(test_loader))
    gt = batch["gt"].to(device)
    hr_msi = batch["hr_msi"].to(device)
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
        from innovation1 import evaluate
        metrics = evaluate(model, test_loader, process, device, scale_ratio=cfg.scale_ratio)
        print("full_reverse", format_metrics(metrics))
