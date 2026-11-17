import torch

from train_innovation1_residual_v_unitvar_ddpm import normalize_residual


def test_unitvar_residual_normalization_does_not_clip_values():
    residual = torch.tensor([[[[-2.0, -0.5, 0.0, 0.5, 2.0]]]])
    mean = torch.zeros(1, 1, 1, 1)
    scale = torch.ones(1, 1, 1, 1)

    normalized, outlier_fraction = normalize_residual(
        residual, mean, scale
    )

    assert torch.equal(normalized, residual)
    assert torch.isclose(
        outlier_fraction,
        torch.tensor(2.0 / 5.0),
    )
    assert normalized.min() < -1.0
    assert normalized.max() > 1.0
