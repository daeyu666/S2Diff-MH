import torch
import torch.nn as nn

from diffusion import GaussianDDPMSchedule
from evaluate_innovation1_sparse_ddim import (
    make_timestep_sequence,
    run_ddim_trajectory,
)


class ZeroVelocityPredictor(nn.Module):
    def forward(self, x_t, condition, timesteps):
        return torch.zeros_like(x_t)


def test_sparse_ddim_timestep_sequences_are_strictly_descending():
    assert make_timestep_sequence(12, 1) == [12, 0]
    assert make_timestep_sequence(12, 2) == [12, 6, 0]
    assert make_timestep_sequence(12, 4) == [12, 9, 6, 3, 0]
    assert make_timestep_sequence(12, 12) == list(range(12, -1, -1))


def test_sparse_ddim_reuses_initial_noise_and_captures_requested_state():
    schedule = GaussianDDPMSchedule(
        total_steps=12, beta_start=1e-4, beta_end=2e-2
    )
    model = ZeroVelocityPredictor()
    condition = torch.zeros(1, 2, 8, 8)
    generator = torch.Generator(device="cpu").manual_seed(2026)
    initial_noise = torch.randn(
        1, 3, 8, 8, generator=generator
    )

    first, captured, sequence = run_ddim_trajectory(
        model,
        condition,
        initial_noise,
        schedule,
        inference_steps=12,
        capture_timestep=6,
    )
    second, captured_again, sequence_again = run_ddim_trajectory(
        model,
        condition,
        initial_noise,
        schedule,
        inference_steps=12,
        capture_timestep=6,
    )

    assert sequence == sequence_again
    assert first.shape == initial_noise.shape
    assert captured is not None and captured.shape == initial_noise.shape
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.equal(captured, captured_again)


def test_one_step_ddim_uses_only_the_tmax_prediction():
    schedule = GaussianDDPMSchedule(
        total_steps=12, beta_start=1e-4, beta_end=2e-2
    )
    model = ZeroVelocityPredictor()
    condition = torch.zeros(1, 2, 8, 8)
    initial_noise = torch.randn(1, 3, 8, 8)

    result, captured, sequence = run_ddim_trajectory(
        model,
        condition,
        initial_noise,
        schedule,
        inference_steps=1,
    )

    assert sequence == [12, 0]
    assert captured is None
    expected = (
        schedule.alpha_bars[-1].sqrt() * initial_noise
    )
    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)
