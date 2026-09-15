import torch

from cdrdi_geometry import forward_warp
from degradations.deformation_aware import (
    DeformationAwareProgressiveDegradation,
    bilinear_border_warp_adjoint,
)
from degradations.physical import PhysicalDegradation
from degradations.progressive import ProgressiveDegradation


def test_bilinear_border_warp_adjoint_matches_inner_product():
    torch.manual_seed(2)
    x = torch.rand(2, 3, 12, 10)
    y = torch.rand_like(x)
    dx = torch.tensor([1.2, -0.7])
    dy = torch.tensor([-0.8, 0.4])
    theta = torch.tensor([1.0, -0.6])
    local = 0.08 * torch.randn(2, 2, 12, 10)

    wx = forward_warp(x, dx, dy, theta, local)
    wty = bilinear_border_warp_adjoint(y, dx, dy, theta, local)
    lhs = (wx * y).sum()
    rhs = (x * wty).sum()
    assert torch.allclose(lhs, rhs, atol=2e-5, rtol=2e-5)


def test_identity_geometry_reduces_to_innovation1_physical_process():
    torch.manual_seed(3)
    x = torch.rand(1, 5, 16, 16)
    operator = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2, truncate=3.0)
    base = ProgressiveDegradation(operator=operator, total_steps=12)
    rigid = torch.zeros(1, 3)
    local = torch.zeros(1, 2, 16, 16)
    geo = DeformationAwareProgressiveDegradation(base, rigid=rigid, local_field=local)

    assert torch.equal(geo.state_at(x, 0), x)
    for t in (1, 4, 5, 8, 9, 12):
        expected = base.state_at(x, t)
        actual = geo.state_at(x, t)
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_terminal_observation_is_forward_geometry_then_sensor_degradation():
    torch.manual_seed(4)
    x = torch.rand(1, 4, 16, 16)
    operator = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2, truncate=3.0)
    base = ProgressiveDegradation(operator=operator, total_steps=12)
    rigid = torch.tensor([[1.1, -0.6, 0.8]])
    local = torch.zeros(1, 2, 16, 16)
    geo = DeformationAwareProgressiveDegradation(base, rigid=rigid, local_field=local)

    expected = operator.degrade(
        forward_warp(x, rigid[:, 0], rigid[:, 1], rigid[:, 2], local)
    )
    actual = geo.terminal_observation(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    geo.assert_terminal_closure(x)


def test_oracle_reverse_update_telescopes_to_clean_reference_state():
    torch.manual_seed(5)
    x = torch.rand(1, 6, 16, 16)
    operator = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2, truncate=3.0)
    base = ProgressiveDegradation(operator=operator, total_steps=12)
    rigid = torch.tensor([[1.3, -0.9, 1.1]])
    local = 0.05 * torch.randn(1, 2, 16, 16)
    geo = DeformationAwareProgressiveDegradation(base, rigid=rigid, local_field=local)

    terminal = geo.terminal_observation(x)
    state = geo.terminal_state(terminal, target_size=(16, 16))
    assert torch.allclose(state, geo.state_at(x, 12), atol=2e-5, rtol=2e-5)

    for t in range(12, 0, -1):
        state = geo.reverse_update(state, x, t)
        expected = geo.state_at(x, t - 1)
        assert torch.allclose(state, expected, atol=3e-5, rtol=3e-5)

    assert torch.allclose(state, x, atol=3e-5, rtol=3e-5)
