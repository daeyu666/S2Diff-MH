"""Matched-observation controls for registered Raw-MSI reconstruction.

The observation process is always physical. A generic trajectory must never
substitute its own synthetic LR-HSI for the common measured observation.
No CDRDI or HPGC-Refiner is used by these Innovation-1 controls.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from degradations.gaussian_bicubic import GaussianBicubicDegradation
from degradations.progressive import ProgressiveDegradation
from innovation1 import batch_state_at, build_progressive_process, model_predict, reconstruct_from_terminal_lr
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models import RawMSIDirectPredictor

EXPERIMENTS = ("single_step", "generic_cold", "physical_cold")
PROTOCOL = "registered-matched-observation-reconstruction-controls-v1"


class MatchedObservationProcess:
    """G_t with a common terminal initializer U_phys(Y_H).

    The clean and acquisition boundaries are shared exactly: G_0 = I and
    G_T = L_T^physical. Only t=1,...,T-1 changes to generic blur/resize. The
    common physical boundary is an endpoint adapter, not a generic blur step.
    Training, for t >= 1:
        q = U_phys(Y_H), delta = q - G_T(X), x_t = G_t(X) + delta.
    Hence x_T == q for both Cold variants, with the SAME unanchored endpoint
    and anchor residual, too. The shared
    terminal residual correction removes delta for an oracle clean predictor.
    """

    def __init__(self, observation_process, trajectory):
        if observation_process.total_steps != trajectory.total_steps:
            raise ValueError("Observation and trajectory timestep counts must match")
        self.observation_process = observation_process
        self.trajectory = trajectory
        self.operator = trajectory.operator
        self.total_steps = trajectory.total_steps

    def terminal_observation(self, gt):
        return self.observation_process.terminal_observation(gt)

    def terminal_state(self, lr_hsi, *, target_size):
        return self.observation_process.terminal_state(lr_hsi, target_size=target_size)

    def state_at(self, x, t):
        if int(t) == self.total_steps:
            return self.observation_process.state_at(x, t)
        return self.trajectory.state_at(x, t)

    def reverse_update(self, x_t, pred_x0, t):
        t = self.trajectory._validate_t(t)
        if x_t.shape != pred_x0.shape:
            raise ValueError("State and clean prediction shapes must match")
        if t == 0:
            return x_t
        return x_t + self.state_at(pred_x0, t - 1) - self.state_at(pred_x0, t)

    @torch.no_grad()
    def training_state(self, gt, lr_hsi, timesteps):
        if torch.any((timesteps < 1) | (timesteps > self.total_steps)):
            raise ValueError("Anchored training states require 1 <= t <= T")
        terminal = self.terminal_state(lr_hsi, target_size=tuple(gt.shape[-2:]))
        delta = terminal - self.state_at(gt, self.total_steps)
        state = batch_state_at(self, gt, timesteps) + delta
        # Avoid cancellation roundoff at the actual observation boundary.
        return torch.where((timesteps == self.total_steps)[:, None, None, None], terminal, state)

    def sample_timesteps(self, batch_size, *, rng, device, boundary_probability, boundary_radius):
        # A private stream cannot perturb the dataset's Python-random flips.
        boundary = self.trajectory.transition_timesteps(radius=boundary_radius)
        values = [rng.choice(boundary) if boundary and rng.random() < boundary_probability
                  else rng.randint(1, self.total_steps) for _ in range(batch_size)]
        return torch.tensor(values, dtype=torch.long, device=device)


def build_process(cfg):
    physical = build_progressive_process(cfg)
    if cfg.experiment == "generic_cold":
        generic = ProgressiveDegradation(
            GaussianBicubicDegradation(cfg.scale_ratio, sigma=cfg.gaussian_sigma,
                                      kernel_size=cfg.gaussian_kernel_size),
            total_steps=cfg.diffusion_steps, stages=physical.stages, default_lift_mode="bilinear")
    else:
        generic = physical
    return MatchedObservationProcess(physical, generic)


def build_model(cfg, info, device):
    # Even the direct control retains the exact time-conditioned backbone and
    # trainable parameter count; its timestep is a constant T, never sampled.
    return RawMSIDirectPredictor(
        n_bands=info["n_bands"], n_msi_bands=info["n_msi_bands"],
        total_steps=cfg.diffusion_steps, base_channels=cfg.base_channels,
        time_dim=cfg.time_dim, dropout=cfg.dropout, spectral_hidden=cfg.spectral_hidden,
        residual_prediction=True).to(device)


def training_prediction(model, process, gt, hr_msi, cfg, rng):
    with torch.no_grad():
        lr_hsi = process.terminal_observation(gt)
        if cfg.experiment == "single_step":
            x_t = process.terminal_state(lr_hsi, target_size=tuple(hr_msi.shape[-2:]))
            timesteps = torch.full((gt.shape[0],), process.total_steps, device=gt.device, dtype=torch.long)
        else:
            timesteps = process.sample_timesteps(
                gt.shape[0], rng=rng, device=gt.device,
                boundary_probability=cfg.boundary_probability, boundary_radius=cfg.boundary_radius)
            x_t = process.training_state(gt, lr_hsi, timesteps)
    return model_predict(model, x_t, timesteps, hr_msi), lr_hsi


@torch.no_grad()
def predict(model, process, lr_hsi, hr_msi, cfg):
    """Inference accepts observations only; never a ground-truth image."""
    model.eval()
    if cfg.experiment == "single_step":
        terminal = process.terminal_state(lr_hsi, target_size=tuple(hr_msi.shape[-2:]))
        t = torch.full((lr_hsi.shape[0],), process.total_steps, device=lr_hsi.device, dtype=torch.long)
        return model_predict(model, terminal, t, hr_msi)
    return reconstruct_from_terminal_lr(
        model, process, lr_hsi, target_size=tuple(hr_msi.shape[-2:]), hr_msi=hr_msi,
        output_mode=cfg.diffusion_output)


def train_one_epoch(model, loader, optimizer, process, cfg, device, *, epoch):
    model.train()
    sam_fn = SAMLoss()
    # Epoch-local stream makes resumed epoch boundaries reproducible, too.
    rng = random.Random(cfg.seed + epoch * 1000003 + 7919)
    sums = dict(loss=0., l1=0., sam=0., deg=0.)
    samples = updates = 0
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        msi = batch["hr_msi"].to(device, non_blocking=True)
        pred, lr_hsi = training_prediction(model, process, gt, msi, cfg, rng)
        l1, sam = F.l1_loss(pred, gt), sam_fn(pred, gt)
        # Optional regularization is the SAME terminal observation loss for
        # every experiment, never the variant-specific intermediate operator.
        deg = (F.l1_loss(process.terminal_observation(pred), lr_hsi)
               if cfg.lambda_deg else pred.new_zeros(()))
        loss = cfg.lambda_l1 * l1 + cfg.lambda_sam * sam + cfg.lambda_deg * deg
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, update={updates}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Check gradients even when clipping is disabled.
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip or float("inf"),
                                       error_if_nonfinite=True)
        optimizer.step()
        size = gt.shape[0]
        for key, value in zip(sums, (loss, l1, sam, deg)):
            sums[key] += float(value.detach()) * size
        samples += size
        updates += 1
    if not samples:
        raise ValueError("Empty training loader")
    return {**{key: value / samples for key, value in sums.items()},
            "samples": samples, "updates": updates}


@torch.no_grad()
def evaluate(model, loader, process, cfg, device):
    from hashlib import sha256

    meter = MetricAverager()
    initial = MetricAverager()
    digest = sha256()
    count = 0
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        msi = batch["hr_msi"].to(device, non_blocking=True)
        lr_hsi = process.terminal_observation(gt)
        pred = predict(model, process, lr_hsi, msi, cfg)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Nonfinite reconstruction")
        init = process.terminal_state(lr_hsi, target_size=tuple(msi.shape[-2:]))
        # Preserve per-patch averaging even if a caller batches validation.
        for i in range(gt.shape[0]):
            meter.update(calc_metrics(pred[i:i+1], gt[i:i+1], cfg.scale_ratio))
            initial.update(calc_metrics(init[i:i+1], gt[i:i+1], cfg.scale_ratio))
            for tensor in (gt[i], lr_hsi[i], msi[i]):
                array = tensor.detach().cpu().contiguous().numpy()
                digest.update(str((array.shape, str(array.dtype))).encode())
                digest.update(array.tobytes())
            count += 1
    if not count:
        raise ValueError("Empty evaluation loader")
    return {**meter.average(), "INIT_PSNR": initial.average()["PSNR"],
            "INIT_SAM": initial.average()["SAM"], "test_patches": count,
            "input_sha256": digest.hexdigest()}
