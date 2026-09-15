import torch

from cdrdi_geometry import cubic_bspline_field, forward_warp, spectral_project
from degradations.physical import PhysicalDegradation


def test_zero_control_field_and_identity_forward_warp():
    image = torch.rand(1, 4, 32, 32)
    control = torch.zeros(1, 2, 5, 5)
    field = cubic_bspline_field(control, (32, 32))
    zero = torch.zeros(1)
    warped = forward_warp(image, zero, zero, zero, field)
    assert torch.allclose(field, torch.zeros_like(field))
    assert torch.allclose(warped, image, atol=1e-6, rtol=1e-6)


def test_fixed_shared_p0_r0_physical_closure():
    torch.manual_seed(3)
    hsi = torch.rand(1, 12, 32, 32)
    weights = torch.rand(4, 12)
    weights = weights / weights.sum(dim=1, keepdim=True)
    control = torch.randn(1, 2, 5, 5) * 0.1
    control = control - control.mean(dim=(-2, -1), keepdim=True)
    field = cubic_bspline_field(control, (32, 32))
    dx = torch.tensor([0.5])
    dy = torch.tensor([-0.25])
    theta = torch.tensor([0.4])
    p0 = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2)

    y_h = p0.degrade(forward_warp(hsi, dx, dy, theta, field))
    left = spectral_project(y_h, weights)
    hr_msi = spectral_project(hsi, weights)
    right = p0.degrade(forward_warp(hr_msi, dx, dy, theta, field))

    assert torch.allclose(left, right, atol=2e-6, rtol=2e-5)
