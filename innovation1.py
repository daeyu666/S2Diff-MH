"""Training and inference engine for Innovation 1.

The engine contains no misregistration augmentation or alignment hooks.
Training states are generated exclusively by the selected progressive
observation process, so the data endpoint and diffusion endpoint stay closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from degradations import ProgressiveDegradation, build_degradation
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from utils import AverageMeter


@dataclass
class TrainStats:
    loss: float
    l1: float
    sam: float
    deg: float


def build_progressive_process(cfg) -> ProgressiveDegradation:
    mode = cfg.degradation_mode
    kwargs = {}
    if mode == "physical":
        kwargs.update(mtf_nyquist=cfg.mtf_nyquist, truncate=cfg.psf_truncate)
    elif mode == "gaussian_bicubic":
        kwargs.update(sigma=cfg.gaussian_sigma, kernel_size=cfg.gaussian_kernel_size)
    operator = build_degradation(mode, scale_ratio=cfg.scale_ratio, **kwargs)
    requested_lift = cfg.lift_mode
    default_lift = None if requested_lift == "auto" else requested_lift
    return ProgressiveDegradation(
        operator=operator,
        total_steps=cfg.diffusion_steps,
        default_lift_mode=default_lift,
    )


def batch_state_at(process: ProgressiveDegradation, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("x must be BxCxHxW")
    if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
        raise ValueError("timesteps must have shape [B]")
    outputs = torch.empty_like(x)
    for t_value in torch.unique(timesteps, sorted=True):
        mask = timesteps == t_value
        outputs[mask] = process.state_at(x[mask], int(t_value.item()))
    return outputs


def degradation_consistency_loss(
    process: ProgressiveDegradation,
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    total = pred_x0.new_zeros(())
    batch_size = pred_x0.shape[0]
    for t_value in torch.unique(timesteps, sorted=True):
        mask = timesteps == t_value
        count = int(mask.sum().item())
        pred_native = process.degrade_at(pred_x0[mask], int(t_value.item()))
        with torch.no_grad():
            target_native = process.degrade_at(target_x0[mask], int(t_value.item()))
        total = total + F.l1_loss(pred_native, target_native) * (count / batch_size)
    return total


def model_predict(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    hr_msi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if bool(getattr(model, "requires_msi", False)):
        if hr_msi is None:
            raise ValueError("Selected predictor requires HR-MSI")
        return model(x_t, hr_msi, timesteps)
    return model(x_t, timesteps)


def train_one_epoch(
    model,
    loader,
    optimizer,
    process: ProgressiveDegradation,
    device,
    *,
    lambda_l1: float = 1.0,
    lambda_sam: float = 0.1,
    lambda_deg: float = 0.0,
    boundary_probability: float = 0.2,
    boundary_radius: int = 1,
    grad_clip: float = 1.0,
) -> TrainStats:
    model.train()
    sam_loss_fn = SAMLoss()
    loss_meter, l1_meter, sam_meter, deg_meter = (AverageMeter() for _ in range(4))

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = None
        if bool(getattr(model, "requires_msi", False)):
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        batch_size = gt.shape[0]
        timesteps = process.sample_timesteps(
            batch_size,
            boundary_probability=boundary_probability,
            boundary_radius=boundary_radius,
            device=device,
        )
        with torch.no_grad():
            x_t = batch_state_at(process, gt, timesteps)
        pred_x0 = model_predict(model, x_t, timesteps, hr_msi)
        l1 = F.l1_loss(pred_x0, gt)
        sam = sam_loss_fn(pred_x0, gt)
        deg = (
            degradation_consistency_loss(process, pred_x0, gt, timesteps)
            if lambda_deg > 0.0
            else pred_x0.new_zeros(())
        )
        loss = lambda_l1 * l1 + lambda_sam * sam + lambda_deg * deg
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip), error_if_nonfinite=True)
        optimizer.step()
        loss_meter.update(loss.item(), batch_size)
        l1_meter.update(l1.item(), batch_size)
        sam_meter.update(sam.item(), batch_size)
        deg_meter.update(deg.item(), batch_size)

    return TrainStats(loss_meter.avg, l1_meter.avg, sam_meter.avg, deg_meter.avg)


@torch.no_grad()
def reconstruct_from_terminal_lr(
    model: torch.nn.Module,
    process: ProgressiveDegradation,
    lr_hsi: torch.Tensor,
    *,
    target_size: Tuple[int, int],
    hr_msi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    model.eval()
    x_t = process.terminal_state(lr_hsi, target_size=target_size)
    for t in range(process.total_steps, 0, -1):
        timestep = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        pred_x0 = model_predict(model, x_t, timestep, hr_msi)
        x_t = process.reverse_update(x_t, pred_x0, t)
    return x_t


@torch.no_grad()
def evaluate(model, loader, process: ProgressiveDegradation, device, *, scale_ratio: int) -> Dict[str, float]:
    model.eval()
    final_meter = MetricAverager()
    init_meter = MetricAverager()
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True) if bool(getattr(model, "requires_msi", False)) else None
        terminal_lr = process.terminal_observation(gt)
        init_state = process.terminal_state(terminal_lr, target_size=tuple(gt.shape[-2:]))
        pred = reconstruct_from_terminal_lr(
            model,
            process,
            terminal_lr,
            target_size=tuple(gt.shape[-2:]),
            hr_msi=hr_msi,
        )
        final_meter.update(calc_metrics(pred, gt, scale_ratio))
        init_meter.update(calc_metrics(init_state, gt, scale_ratio))
    metrics = final_meter.average()
    initial = init_meter.average()
    metrics["INIT_PSNR"] = initial.get("PSNR", float("nan"))
    metrics["INIT_SAM"] = initial.get("SAM", float("nan"))
    return metrics
