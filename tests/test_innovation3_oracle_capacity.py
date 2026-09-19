import torch

from diagnose_innovation3_oracle_capacity import oracle_projection_components
from models.spectral_fidelity_refiner import HeterogeneityGuidedSpectralRefiner


def test_oracle_full_projection_stays_in_broadshape_and_tangent_space():
    torch.manual_seed(4)
    refiner = HeterogeneityGuidedSpectralRefiner(
        n_bands=31,
        total_steps=12,
        hidden_channels=8,
        variant="full",
    )
    base = torch.rand(2, 31, 5, 6) + 0.2
    target = torch.rand(2, 31, 5, 6) + 0.2
    out = oracle_projection_components(refiner, base, target)

    allowed = out["allowed"]
    unit = base / torch.linalg.vector_norm(base, dim=1, keepdim=True)
    tangent_dot = (allowed * unit).sum(dim=1)
    assert float(tangent_dot.abs().max()) < 3e-5

    coeff = refiner.dct(allowed)
    outside = torch.ones(31, dtype=torch.bool)
    outside[4:refiner.low_end] = False
    assert float(coeff[:, outside].abs().max()) < 3e-5


def test_oracle_capture_is_bounded_and_oracle_sam_does_not_worsen():
    torch.manual_seed(5)
    refiner = HeterogeneityGuidedSpectralRefiner(
        n_bands=31,
        total_steps=12,
        hidden_channels=8,
        variant="full",
    )
    base = torch.rand(2, 31, 7, 7) + 0.3
    target = torch.rand(2, 31, 7, 7) + 0.3
    out = oracle_projection_components(refiner, base, target)

    valid = out["ideal_energy"] > 1e-8
    assert float(out["capture_c4l"][valid].min()) >= -1e-5
    assert float(out["capture_c4l"][valid].max()) <= 1.0001
    assert float(out["capture_full"][valid].min()) >= -1e-5
    assert float(out["capture_full"][valid].max()) <= 1.0001
    assert torch.all(out["capture_full"][valid] <= out["capture_c4l"][valid] + 1e-4)
    assert torch.all(out["oracle_sam"][valid] <= out["base_sam"][valid] + 1e-4)
