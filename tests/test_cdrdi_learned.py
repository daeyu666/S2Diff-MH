import torch

from cdrdi_geometry import forward_warp
from degradations.physical import PhysicalDegradation
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver


def test_learned_cdrdi_one_shot_and_recursive_share_parameters_and_shapes():
    torch.manual_seed(0)
    hr_msi = torch.rand(2, 4, 32, 32)
    p0 = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2, truncate=3.0)
    dx = torch.tensor([1.0, -1.25])
    dy = torch.tensor([-0.5, 0.75])
    theta = torch.tensor([0.5, -0.75])
    local = torch.zeros(2, 2, 32, 32)
    target = p0.degrade(forward_warp(hr_msi, dx, dy, theta, local)).detach()

    model = LearnedPhysicalResidualSolver(
        4,
        base_channels=16,
        control_grid=5,
        max_translation=4.0,
        max_rotation_deg=2.0,
        max_local_px=4.0,
    )
    n_params_before = sum(p.numel() for p in model.parameters())
    one = model(target, hr_msi, p0, steps=1)
    rec = model(target, hr_msi, p0, steps=3)
    n_params_after = sum(p.numel() for p in model.parameters())

    assert n_params_before == n_params_after
    assert len(one["predictions"]) == 1
    assert len(rec["predictions"]) == 3
    assert one["final_prediction"].shape == target.shape
    assert rec["final_prediction"].shape == target.shape
    assert rec["final_local_field"].shape == (2, 2, 32, 32)
    assert rec["final_rigid"].shape == (2, 3)
    assert rec["final_control"].shape == (2, 2, 5, 5)


def test_learned_cdrdi_closure_is_differentiable_without_flow_supervision():
    torch.manual_seed(1)
    hr_msi = torch.rand(1, 4, 32, 32)
    p0 = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2, truncate=3.0)
    dx = torch.tensor([1.5])
    dy = torch.tensor([-1.0])
    theta = torch.tensor([0.8])
    local = torch.zeros(1, 2, 32, 32)
    target = p0.degrade(forward_warp(hr_msi, dx, dy, theta, local)).detach()

    model = LearnedPhysicalResidualSolver(
        4,
        base_channels=16,
        control_grid=5,
        max_translation=4.0,
        max_rotation_deg=2.0,
        max_local_px=4.0,
    )
    out = model(target, hr_msi, p0, steps=2)
    loss = (out["final_prediction"] - target).abs().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    dense_amp = torch.linalg.vector_norm(out["final_local_field"].detach(), dim=1).amax()
    assert float(dense_amp) <= 4.0001
