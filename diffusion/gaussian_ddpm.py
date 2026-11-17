"""Standard short-chain Gaussian DDPM and ancestral reverse sampling.

This ablation uses a 12-step cosine Gaussian schedule so the forward and
reverse process have the same number of steps as the physical-diffusion
baseline. Clean reflectance cubes are normalized to [-1, 1] by the trainer.
Timesteps are integer values in [1, T].
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn


class GaussianDDPMSchedule:
    """Cosine cumulative-alpha schedule and standard DDPM posterior."""

    def __init__(
        self,
        total_steps: int = 12,
        cosine_offset: float = 0.008,
        device: Optional[torch.device] = None,
    ):
        if total_steps < 2:
            raise ValueError("total_steps must be >= 2")
        if cosine_offset < 0:
            raise ValueError("cosine_offset must be non-negative")

        self.total_steps = int(total_steps)
        self.cosine_offset = float(cosine_offset)

        grid = torch.linspace(
            0, self.total_steps, self.total_steps + 1,
            dtype=torch.float64, device=device,
        ) / float(self.total_steps)
        angle = (
            (grid + self.cosine_offset)
            / (1.0 + self.cosine_offset)
            * (math.pi / 2.0)
        )
        alpha_bar_curve = torch.cos(angle).square()
        alpha_bar_curve = alpha_bar_curve / alpha_bar_curve[0]
        betas = 1.0 - alpha_bar_curve[1:] / alpha_bar_curve[:-1]
        self.betas = betas.clamp(min=1e-8, max=0.999).float()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        alpha_bar_previous = torch.cat([
            torch.ones(1, dtype=self.alpha_bars.dtype, device=device),
            self.alpha_bars[:-1],
        ])
        denominator = (1.0 - self.alpha_bars).clamp_min(1e-12)
        self.posterior_variance = (
            self.betas * (1.0 - alpha_bar_previous) / denominator
        ).clamp_min(0.0)
        self.posterior_mean_coef_x0 = (
            self.betas * alpha_bar_previous.sqrt() / denominator
        )
        self.posterior_mean_coef_xt = (
            (1.0 - alpha_bar_previous)
            * self.alphas.sqrt()
            / denominator
        )

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
        """Sample q(x_t | x_0) = N(sqrt(a_bar)x_0, (1-a_bar)I)."""
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
            timesteps, x0.ndim,
        ).to(device=x0.device, dtype=x0.dtype)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    condition: Tensor,
    sample_shape: Sequence[int],
    schedule: GaussianDDPMSchedule,
    generator: Optional[torch.Generator] = None,
    clip_x0: bool = True,
) -> Tensor:
    """Run the full ancestral DDPM reverse chain from x_T ~ N(0,I).

    The model contract is model(x_t, condition, timesteps) -> predicted epsilon.
    The returned x_0 is in the same normalized [-1,1] domain used for training.
    """
    if len(sample_shape) != 4:
        raise ValueError("sample_shape must be [B,C,H,W]")
    shape = tuple(int(v) for v in sample_shape)
    if shape[0] != condition.shape[0]:
        raise ValueError("condition and sample batch sizes must match")
    if shape[-2:] != tuple(condition.shape[-2:]):
        raise ValueError("condition and sample spatial sizes must match")

    device, dtype = condition.device, condition.dtype
    x_t = torch.randn(shape, device=device, dtype=dtype, generator=generator)

    for timestep in range(schedule.total_steps, 0, -1):
        t_batch = torch.full(
            (shape[0],), timestep, dtype=torch.long, device=device
        )
        predicted_noise = model(x_t, condition, t_batch)
        if predicted_noise.shape != x_t.shape:
            raise ValueError(
                "Noise predictor output shape must match the noisy HSI state"
            )

        alpha_bar_t = schedule.alpha_bars[timestep - 1].to(
            device=device, dtype=dtype
        )
        predicted_x0 = (
            x_t - (1.0 - alpha_bar_t).clamp_min(0.0).sqrt() * predicted_noise
        ) / alpha_bar_t.sqrt().clamp_min(1e-8)

        if clip_x0:
            predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

        coef_x0 = schedule.posterior_mean_coef_x0[timestep - 1].to(
            device=device, dtype=dtype
        )
        coef_xt = schedule.posterior_mean_coef_xt[timestep - 1].to(
            device=device, dtype=dtype
        )
        posterior_mean = coef_x0 * predicted_x0 + coef_xt * x_t

        if timestep > 1:
            variance = schedule.posterior_variance[timestep - 1].to(
                device=device, dtype=dtype
            )
            noise = torch.randn(
                shape, device=device, dtype=dtype, generator=generator
            )
            x_t = posterior_mean + variance.clamp_min(0.0).sqrt() * noise
        else:
            x_t = posterior_mean

    return x_t.clamp(-1.0, 1.0) if clip_x0 else x_t
