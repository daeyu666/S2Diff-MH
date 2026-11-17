"""Standard Gaussian DDPM forward process and deterministic DDIM sampling.

Timesteps are represented as integers in [1, T]. Array index t - 1 stores
the coefficient for diffusion step t. This module is independent of the
project's physical degradation operators.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn


class GaussianDDPMSchedule:
    """Linear-beta DDPM schedule with closed-form q(x_t | x_0) sampling."""

    def __init__(
        self,
        total_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[torch.device] = None,
    ):
        if total_steps < 2:
            raise ValueError("total_steps must be >= 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1")

        self.total_steps = int(total_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.total_steps,
            dtype=torch.float32,
            device=device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _extract(self, values: Tensor, timesteps: Tensor, ndim: int) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [B]")
        if timesteps.numel() == 0:
            raise ValueError("timesteps cannot be empty")
        if torch.any(timesteps < 1) or torch.any(timesteps > self.total_steps):
            raise ValueError(f"timesteps must lie in [1, {self.total_steps}]")
        selected = values.gather(0, timesteps.long() - 1)
        return selected.reshape(timesteps.shape[0], *([1] * (ndim - 1)))

    def q_sample(
        self,
        x0: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample q(x_t | x_0) using the closed-form Gaussian marginal."""
        if x0.ndim != 4:
            raise ValueError("x0 must have shape [B,C,H,W]")
        if timesteps.ndim != 1 or timesteps.shape[0] != x0.shape[0]:
            raise ValueError("timesteps must have shape [B] matching x0")
        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape:
            raise ValueError("noise and x0 must have identical shapes")

        sqrt_alpha_bar = self._extract(
            self.alpha_bars.sqrt(), timesteps, x0.ndim
        ).to(device=x0.device, dtype=x0.dtype)
        sqrt_one_minus_alpha_bar = self._extract(
            (1.0 - self.alpha_bars).clamp_min(0.0).sqrt(),
            timesteps,
            x0.ndim,
        ).to(device=x0.device, dtype=x0.dtype)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise


def ddim_timesteps(total_steps: int, sampling_steps: int) -> List[int]:
    """Return a strictly descending, endpoint-inclusive DDIM time grid."""
    if total_steps < 2:
        raise ValueError("total_steps must be >= 2")
    if sampling_steps < 1 or sampling_steps > total_steps:
        raise ValueError("sampling_steps must lie in [1, total_steps]")

    ascending = torch.linspace(
        1,
        total_steps,
        steps=sampling_steps,
        dtype=torch.float64,
    ).round().long()
    values = ascending.tolist()
    if len(set(values)) != sampling_steps:
        raise ValueError("DDIM time grid contains duplicate timesteps")
    return list(reversed(values))


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    condition: Tensor,
    sample_shape: Sequence[int],
    schedule: GaussianDDPMSchedule,
    sampling_steps: int = 12,
    generator: Optional[torch.Generator] = None,
    clip_x0: bool = True,
) -> Tensor:
    """Generate an HSI sample with eta=0 DDIM and a conditional noise model.

    The model contract is model(x_t, condition, timesteps) -> predicted_noise.
    For reflectance data normalized to [0,1], predicted x_0 is clipped to that
    interval and epsilon is recomputed before each deterministic DDIM update.
    """
    if len(sample_shape) != 4:
        raise ValueError("sample_shape must be [B,C,H,W]")
    shape = tuple(int(v) for v in sample_shape)
    if shape[0] != condition.shape[0]:
        raise ValueError("condition and sample batch sizes must match")
    if shape[-2:] != tuple(condition.shape[-2:]):
        raise ValueError("condition and sample spatial sizes must match")
    if sampling_steps < 1 or sampling_steps > schedule.total_steps:
        raise ValueError("Invalid sampling_steps")

    device = condition.device
    dtype = condition.dtype
    x_t = torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    time_grid = ddim_timesteps(schedule.total_steps, sampling_steps)

    for index, timestep in enumerate(time_grid):
        t_batch = torch.full(
            (shape[0],),
            int(timestep),
            dtype=torch.long,
            device=device,
        )
        predicted_noise = model(x_t, condition, t_batch)
        if predicted_noise.shape != x_t.shape:
            raise ValueError(
                "Noise predictor output shape must match the noisy HSI state"
            )

        alpha_bar_t = schedule.alpha_bars[timestep - 1].to(
            device=device, dtype=dtype
        )
        previous_timestep = (
            time_grid[index + 1] if index + 1 < len(time_grid) else 0
        )
        if previous_timestep == 0:
            alpha_bar_previous = torch.ones((), device=device, dtype=dtype)
        else:
            alpha_bar_previous = schedule.alpha_bars[
                previous_timestep - 1
            ].to(device=device, dtype=dtype)

        sqrt_alpha_bar_t = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha_bar_t = (
            1.0 - alpha_bar_t
        ).clamp_min(1e-12).sqrt()

        predicted_x0 = (
            x_t - sqrt_one_minus_alpha_bar_t * predicted_noise
        ) / sqrt_alpha_bar_t.clamp_min(1e-12)

        if clip_x0:
            predicted_x0 = predicted_x0.clamp(0.0, 1.0)
            predicted_noise = (
                x_t - sqrt_alpha_bar_t * predicted_x0
            ) / sqrt_one_minus_alpha_bar_t

        x_t = (
            alpha_bar_previous.sqrt() * predicted_x0
            + (1.0 - alpha_bar_previous).clamp_min(0.0).sqrt()
            * predicted_noise
        )

    return x_t.clamp(0.0, 1.0) if clip_x0 else x_t
