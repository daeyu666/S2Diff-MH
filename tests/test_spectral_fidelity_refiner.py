import torch

from models.spectral_fidelity_refiner import (
    HeterogeneityGuidedSpectralRefiner,
    dct_ii_matrix,
    ranked_msi_heterogeneity,
)


def test_dct_is_orthonormal():
    c = dct_ii_matrix(31)
    eye = c @ c.T
    assert torch.allclose(eye, torch.eye(31), atol=1e-5, rtol=1e-5)


def test_ranked_msi_heterogeneity_is_finite_and_bounded():
    torch.manual_seed(0)
    msi = torch.rand(2, 4, 9, 11)
    rank = ranked_msi_heterogeneity(msi)
    assert rank.shape == (2, 9, 11)
    assert torch.isfinite(rank).all()
    assert float(rank.min()) >= 0.0
    assert float(rank.max()) <= 1.0


def test_broadshape_projection_keeps_only_c4l():
    torch.manual_seed(1)
    module = HeterogeneityGuidedSpectralRefiner(
        n_bands=31,
        total_steps=12,
        hidden_channels=8,
        variant="broad",
    )
    residual = torch.randn(2, 31, 5, 6)
    projected = module.project_broadshape(residual)
    coeff = module.dct(projected)
    outside = torch.ones(31, dtype=torch.bool)
    outside[4:module.low_end] = False
    assert float(coeff[:, outside].abs().max()) < 2e-5


def test_tangent_projection_is_orthogonal_to_base():
    torch.manual_seed(2)
    module = HeterogeneityGuidedSpectralRefiner(
        n_bands=31,
        total_steps=12,
        hidden_channels=8,
        variant="full",
    )
    base = torch.rand(2, 31, 5, 6) + 0.1
    residual = torch.randn_like(base)
    tangent = module.project_tangent(residual, base)
    unit = base / torch.linalg.vector_norm(base, dim=1, keepdim=True)
    dot = (tangent * unit).sum(dim=1)
    assert float(dot.abs().max()) < 2e-5


def test_gate_is_monotone_in_msi_heterogeneity_rank():
    module = HeterogeneityGuidedSpectralRefiner(
        n_bands=31,
        total_steps=12,
        hidden_channels=8,
        variant="full",
    )
    rank = torch.linspace(0.0, 1.0, 101)
    gate = module.gate_from_rank(rank)
    assert torch.all(gate[1:] >= gate[:-1])
    assert float(gate[-1]) > float(gate[0])


def test_zero_initialized_refiner_starts_from_exact_baseline():
    torch.manual_seed(3)
    module = HeterogeneityGuidedSpectralRefiner(
        n_bands=31,
        total_steps=12,
        hidden_channels=8,
        variant="full",
    )
    x_t = torch.rand(2, 31, 8, 8)
    base = torch.rand(2, 31, 8, 8)
    msi = torch.rand(2, 4, 8, 8)
    t = torch.tensor([3, 9])
    refined, details = module(x_t, base, msi, t, return_details=True)
    assert torch.equal(refined, base)
    assert torch.equal(details["pre_update"], torch.zeros_like(base))
