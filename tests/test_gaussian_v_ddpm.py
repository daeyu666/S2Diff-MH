import torch
import torch.nn as nn

from diffusion import (
    GaussianDDPMSchedule,
    ddpm_v_sample,
    v_prediction_target,
    v_to_epsilon,
    v_to_x0,
)


class ZeroVelocityPredictor(nn.Module):
    def forward(self, x_t, condition, timesteps):
        return torch.zeros_like(x_t)


def test_v_parameterization_exactly_recovers_x0_and_noise():
    schedule = GaussianDDPMSchedule(
        total_steps=1000, beta_start=1e-4, beta_end=2e-2
    )
    x0 = torch.randn(2, 5, 8, 8).clamp(-1.0, 1.0)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([1, 1000], dtype=torch.long)
    x_t = schedule.q_sample(x0, timesteps, noise)

    target_v = v_prediction_target(x0, noise, timesteps, schedule)
    recovered_x0 = v_to_x0(x_t, target_v, timesteps, schedule)
    recovered_noise = v_to_epsilon(x_t, target_v, timesteps, schedule)

    assert torch.allclose(recovered_x0, x0, atol=2e-6, rtol=2e-6)
    assert torch.allclose(recovered_noise, noise, atol=2e-6, rtol=2e-6)


def test_v_prediction_target_has_same_shape_and_finite_values():
    schedule = GaussianDDPMSchedule(total_steps=1000)
    x0 = torch.randn(1, 3, 8, 8)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([500], dtype=torch.long)

    target_v = v_prediction_target(x0, noise, timesteps, schedule)

    assert target_v.shape == x0.shape
    assert torch.isfinite(target_v).all()


def test_full_ancestral_v_ddpm_sampler_returns_finite_normalized_cube():
    schedule = GaussianDDPMSchedule(total_steps=12)
    model = ZeroVelocityPredictor()
    condition = torch.rand(2, 5, 16, 16) * 2.0 - 1.0
    generator = torch.Generator(device="cpu").manual_seed(123)

    result = ddpm_v_sample(
        model,
        condition,
        sample_shape=(2, 3, 16, 16),
        schedule=schedule,
        generator=generator,
    )

    assert result.shape == (2, 3, 16, 16)
    assert torch.isfinite(result).all()
    assert result.min() >= -1.0
    assert result.max() <= 1.0

class ConstantVelocityPredictor(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, x_t, condition, timesteps):
        return torch.full_like(x_t, self.value)


def test_v_sampler_can_preserve_unbounded_x0_when_clipping_is_disabled():
    schedule = GaussianDDPMSchedule(
        total_steps=2, beta_start=0.1, beta_end=0.2
    )
    model = ConstantVelocityPredictor(5.0)
    condition = torch.zeros(1, 2, 4, 4)
    generator = torch.Generator(device="cpu").manual_seed(7)

    result = ddpm_v_sample(
        model,
        condition,
        sample_shape=(1, 1, 4, 4),
        schedule=schedule,
        generator=generator,
        clip_x0=False,
    )

    assert torch.isfinite(result).all()
    assert (result.abs() > 1.0).any()
