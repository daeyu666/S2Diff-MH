import torch
import torch.nn as nn

from diffusion import GaussianDDPMSchedule, ddpm_sample
from models import RawMSIDirectPredictor


class ZeroNoisePredictor(nn.Module):
    def forward(self, x_t, condition, timesteps):
        return torch.zeros_like(x_t)


def test_12_step_cosine_schedule_reaches_near_gaussian_terminal_state():
    schedule = GaussianDDPMSchedule(total_steps=12, cosine_offset=0.008)
    assert schedule.betas.shape == (12,)
    assert torch.all(schedule.betas > 0.0)
    assert torch.all(schedule.betas <= 0.999)
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])
    assert schedule.alpha_bars[-1] < 1e-3


def test_q_sample_matches_closed_form_and_preserves_shape():
    schedule = GaussianDDPMSchedule(total_steps=12)
    x0 = torch.rand(2, 3, 8, 8) * 2.0 - 1.0
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([1, 12], dtype=torch.long)
    xt = schedule.q_sample(x0, timesteps, noise)
    expected_scale = schedule.alpha_bars[timesteps - 1].sqrt().view(2, 1, 1, 1)
    assert torch.allclose(xt, expected_scale * x0)
    assert torch.isfinite(xt).all()


def test_ancestral_ddpm_sampler_returns_finite_normalized_cube():
    schedule = GaussianDDPMSchedule(total_steps=12)
    model = ZeroNoisePredictor()
    condition = torch.rand(2, 5, 16, 16) * 2.0 - 1.0
    generator = torch.Generator(device="cpu").manual_seed(123)
    result = ddpm_sample(
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


def test_raw_direct_backbone_accepts_concatenated_condition():
    # 8 HSI bands + 3 MSI bands are concatenated into an 11-channel condition.
    model = RawMSIDirectPredictor(
        n_bands=8,
        n_msi_bands=11,
        total_steps=12,
        base_channels=8,
        time_dim=16,
        residual_prediction=False,
        spectral_hidden=2,
    )
    noisy_hsi = torch.randn(1, 8, 16, 16)
    condition = torch.rand(1, 11, 16, 16) * 2.0 - 1.0
    timestep = torch.tensor([12], dtype=torch.long)
    predicted_noise = model(noisy_hsi, condition, timestep)
    assert predicted_noise.shape == noisy_hsi.shape
    assert torch.isfinite(predicted_noise).all()
