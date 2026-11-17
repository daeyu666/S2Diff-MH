"""Velocity-parameterized standard Gaussian DDPM utilities.

The forward process is the conventional linear-beta Gaussian Markov chain.
Only the network prediction parameterization changes from epsilon to v.
Timesteps are one-based: 1..T.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .gaussian_ddpm import GaussianDDPMSchedule


def _extract_alpha_bar(
    schedule: GaussianDDPMSchedule,
    timesteps: Tensor,
    reference: Tensor,
) -> Tensor:
    if timesteps.ndim != 1 or timesteps.shape[0] != reference.shape[0]:
        raise ValueError("timesteps must have shape [B] matching the reference")
    if torch.any(timesteps < 1) or torch.any(timesteps > schedule.total_steps):
        raise ValueError(
            f"timesteps must lie in [1, {schedule.total_steps}]"
        )
    alpha_bar = schedule.alpha_bars.gather(
        0, timesteps.to(device=schedule.alpha_bars.device, dtype=torch.long) - 1
    )
    return alpha_bar.reshape(
        timesteps.shape[0], *([1] * (reference.ndim - 1))
    ).to(device=reference.device, dtype=reference.dtype)


def v_prediction_target(
    x0: Tensor,
    noise: Tensor,
    timesteps: Tensor,
    schedule: GaussianDDPMSchedule,
) -> Tensor:
    """Return v = sqrt(alpha_bar)*noise - sqrt(1-alpha_bar)*x0."""
    if x0.shape != noise.shape:
        raise ValueError("x0 and noise must have identical shapes")
    alpha_bar = _extract_alpha_bar(schedule, timesteps, x0)
    return alpha_bar.sqrt() * noise - (1.0 - alpha_bar).clamp_min(0.0).sqrt() * x0


def v_to_x0(
    x_t: Tensor,
    predicted_v: Tensor,
    timesteps: Tensor,
    schedule: GaussianDDPMSchedule,
) -> Tensor:
    """Recover x0 = sqrt(alpha_bar)*x_t - sqrt(1-alpha_bar)*v."""
    if x_t.shape != predicted_v.shape:
        raise ValueError("x_t and predicted_v must have identical shapes")
    alpha_bar = _extract_alpha_bar(schedule, timesteps, x_t)
    return (
        alpha_bar.sqrt() * x_t
        - (1.0 - alpha_bar).clamp_min(0.0).sqrt() * predicted_v
    )


def v_to_epsilon(
    x_t: Tensor,
    predicted_v: Tensor,
    timesteps: Tensor,
    schedule: GaussianDDPMSchedule,
) -> Tensor:
    """Recover epsilon = sqrt(1-alpha_bar)*x_t + sqrt(alpha_bar)*v."""
    if x_t.shape != predicted_v.shape:
        raise ValueError("x_t and predicted_v must have identical shapes")
    alpha_bar = _extract_alpha_bar(schedule, timesteps, x_t)
    return (
        (1.0 - alpha_bar).clamp_min(0.0).sqrt() * x_t
        + alpha_bar.sqrt() * predicted_v
    )


@torch.no_grad()
def ddpm_v_sample(
    model: nn.Module,
    condition: Tensor,
    sample_shape: Sequence[int],
    schedule: GaussianDDPMSchedule,
    generator: Optional[torch.Generator] = None,
    clip_x0: bool = True,
) -> Tensor:
    """Run full ancestral DDPM sampling with a v-prediction network."""
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
        predicted_v = model(x_t, condition, t_batch)
        if predicted_v.shape != x_t.shape:
            raise ValueError(
                "Velocity predictor output shape must match the noisy state"
            )

        predicted_x0 = v_to_x0(x_t, predicted_v, t_batch, schedule)
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

@torch.no_grad()
def ddim_v_sample(
    model: nn.Module,
    condition: Tensor,
    sample_shape: Sequence[int],
    schedule: GaussianDDPMSchedule,
    generator: Optional[torch.Generator] = None,
    ddim_steps: Optional[int] = None,
    eta: float = 0.0,
    clip_x0: bool = False,
) -> Tensor:
    """Run DDIM sampling with a v-prediction network.

    eta=0 gives deterministic sampling. The sampler supports a reduced number
    of inference steps while retaining the training-time timestep indices.
    """
    if len(sample_shape) != 4:
        raise ValueError("sample_shape must be [B,C,H,W]")
    shape = tuple(int(v) for v in sample_shape)
    if shape[0] != condition.shape[0]:
        raise ValueError("condition and sample batch sizes must match")
    if shape[-2:] != tuple(condition.shape[-2:]):
        raise ValueError("condition and sample spatial sizes must match")
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("eta must lie in [0,1]")

    steps = schedule.total_steps if ddim_steps is None else int(ddim_steps)
    if steps < 1 or steps > schedule.total_steps:
        raise ValueError(
            f"ddim_steps must lie in [1, {schedule.total_steps}]"
        )

    # Include the clean endpoint t=0. Rounding creates a strictly descending
    # sequence because steps cannot exceed the training diffusion length.
    timestep_sequence = torch.linspace(
        schedule.total_steps, 0, steps + 1, dtype=torch.float64
    ).round().to(torch.long).tolist()
    if any(a <= b for a, b in zip(
        timestep_sequence[:-1], timestep_sequence[1:]
    )):
        raise RuntimeError("DDIM timestep sequence must be strictly descending")

    device, dtype = condition.device, condition.dtype
    x_t = torch.randn(shape, device=device, dtype=dtype, generator=generator)

    for timestep, previous_timestep in zip(
        timestep_sequence[:-1], timestep_sequence[1:]
    ):
        t_batch = torch.full(
            (shape[0],), timestep, dtype=torch.long, device=device
        )
        predicted_v = model(x_t, condition, t_batch)
        if predicted_v.shape != x_t.shape:
            raise ValueError(
                "Velocity predictor output shape must match the noisy state"
            )

        predicted_x0 = v_to_x0(x_t, predicted_v, t_batch, schedule)
        if clip_x0:
            predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
        predicted_epsilon = v_to_epsilon(
            x_t, predicted_v, t_batch, schedule
        )

        alpha_bar_t = schedule.alpha_bars[timestep - 1].to(
            device=device, dtype=dtype
        )
        if previous_timestep == 0:
            alpha_bar_previous = torch.ones((), device=device, dtype=dtype)
        else:
            alpha_bar_previous = schedule.alpha_bars[
                previous_timestep - 1
            ].to(device=device, dtype=dtype)

        sigma_squared = (
            float(eta) ** 2
            * (1.0 - alpha_bar_previous)
            / (1.0 - alpha_bar_t).clamp_min(1e-12)
            * (1.0 - alpha_bar_t / alpha_bar_previous.clamp_min(1e-12))
        ).clamp_min(0.0)
        direction_variance = (
            1.0 - alpha_bar_previous - sigma_squared
        ).clamp_min(0.0)

        x_previous = (
            alpha_bar_previous.sqrt() * predicted_x0
            + direction_variance.sqrt() * predicted_epsilon
        )
        if eta > 0.0 and previous_timestep > 0:
            noise = torch.randn(
                shape, device=device, dtype=dtype, generator=generator
            )
            x_previous = x_previous + sigma_squared.sqrt() * noise
        x_t = x_previous

    return x_t.clamp(-1.0, 1.0) if clip_x0 else x_t
