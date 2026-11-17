import torch
import torch.nn as nn

from diffusion import GaussianDDPMSchedule, ddim_sample, ddim_timesteps
from models import RawMSIDirectPredictor


class ZeroNoisePredictor(nn.Module):
    def forward(self, x_t, condition, timesteps):
        return torch.zeros_like(x_t)


def test_linear_gaussian_schedule_and_q_sample():
    schedule = GaussianDDPMSchedule(total_steps=20, beta_start=1e-4, beta_end=2e-2)
    assert schedule.betas.shape == (20,)
    assert torch.all(schedule.betas[1:] > schedule.betas[:-1])
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])

    x0 = torch.rand(2, 3, 8, 8)
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([1, 20], dtype=torch.long)
    xt = schedule.q_sample(x0, timesteps, noise)
    expected_scale = schedule.alpha_bars[timesteps - 1].sqrt().view(2, 1, 1, 1)
    assert torch.allclose(xt, expected_scale * x0)
    assert torch.isfinite(xt).all()


def test_ddim_grid_is_descending_and_endpoint_inclusive():
    values = ddim_timesteps(total_steps=1000, sampling_steps=12)
    assert len(values) == 12
    assert values[0] == 1000
    assert values[-1] == 1
    assert all(a > b for a, b in zip(values[:-1], values[1:]))


def test_ddim_sampler_returns_finite_reflectance_cube():
    schedule = GaussianDDPMSchedule(total_steps=20)
    model = ZeroNoisePredictor()
    condition = torch.rand(2, 5, 16, 16)
    generator = torch.Generator(device="cpu").manual_seed(123)
    result = ddim_sample(
        model,
        condition,
        sample_shape=(2, 3, 16, 16),
        schedule=schedule,
        sampling_steps=4,
        generator=generator,
    )
    assert result.shape == (2, 3, 16, 16)
    assert torch.isfinite(result).all()
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_raw_direct_backbone_accepts_concatenated_lr_hsi_and_msi_condition():
    # 8 HSI bands + 3 MSI bands are concatenated into an 11-channel condition.
    model = RawMSIDirectPredictor(
        n_bands=8,
        n_msi_bands=11,
        total_steps=1000,
        base_channels=8,
        time_dim=16,
        residual_prediction=False,
        spectral_hidden=2,
    )
    noisy_hsi = torch.randn(1, 8, 16, 16)
    condition = torch.rand(1, 11, 16, 16)
    timestep = torch.tensor([1000], dtype=torch.long)
    predicted_noise = model(noisy_hsi, condition, timestep)
    assert predicted_noise.shape == noisy_hsi.shape
    assert torch.isfinite(predicted_noise).all()
