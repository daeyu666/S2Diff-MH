import torch

from train_innovation1_residual_ddpm import (
    denormalize_residual,
    normalize_residual,
)


def test_residual_normalization_round_trip_without_clipping():
    residual = torch.tensor(
        [[[[0.10, -0.10]], [[0.20, -0.20]], [[0.05, -0.05]]]],
        dtype=torch.float32,
    )
    mean = torch.tensor([[[[0.0]], [[0.0]], [[0.0]]]], dtype=torch.float32)
    scale = torch.tensor([[[[0.5]], [[1.0]], [[0.25]]]], dtype=torch.float32)

    normalized, clip_fraction = normalize_residual(residual, mean, scale)
    recovered = denormalize_residual(normalized, mean, scale)

    assert torch.allclose(recovered, residual, atol=1e-7)
    assert clip_fraction.item() == 0.0


def test_residual_normalization_clips_outliers_and_reports_fraction():
    residual = torch.tensor([[[[0.0, 2.0]], [[-3.0, 0.5]]]], dtype=torch.float32)
    mean = torch.zeros(1, 2, 1, 1)
    scale = torch.ones(1, 2, 1, 1)

    normalized, clip_fraction = normalize_residual(residual, mean, scale)

    assert normalized.min().item() >= -1.0
    assert normalized.max().item() <= 1.0
    assert abs(clip_fraction.item() - 0.5) < 1e-7
    assert torch.isfinite(normalized).all()
