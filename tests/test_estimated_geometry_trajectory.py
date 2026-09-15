import torch

from cdrdi_geometry import forward_warp
from degradations.deformation_aware import DeformationAwareProgressiveDegradation
from degradations.physical import PhysicalDegradation
from degradations.progressive import ProgressiveDegradation
from train_cdrdi_diffusion_estimated import observation_anchored_batch_state


def _process(base, rigid, local):
    return DeformationAwareProgressiveDegradation(
        base,
        rigid=rigid,
        local_field=local,
    )


def test_observation_anchored_state_matches_model_state_when_geometry_is_exact():
    torch.manual_seed(5)
    x = torch.rand(2, 4, 32, 32)
    op = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2, truncate=3.0)
    base = ProgressiveDegradation(op, total_steps=12)
    rigid = torch.tensor([[1.2, -0.7, 0.8], [-0.8, 0.4, -0.5]])
    local = torch.zeros(2, 2, 32, 32)
    process = _process(base, rigid, local)
    y_h = process.terminal_observation(x)
    timesteps = torch.tensor([3, 10], dtype=torch.long)

    anchored = observation_anchored_batch_state(process, x, y_h, timesteps)
    expected = torch.empty_like(x)
    for t in torch.unique(timesteps):
        mask = timesteps == t
        expected[mask] = process.state_at(x, int(t.item()))[mask]

    assert torch.allclose(anchored, expected, atol=2e-5, rtol=2e-5)


def test_imperfect_geometry_keeps_terminal_model_mismatch_as_constant_offset():
    torch.manual_seed(6)
    x = torch.rand(1, 3, 32, 32)
    op = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2, truncate=3.0)
    base = ProgressiveDegradation(op, total_steps=12)

    true_rigid = torch.tensor([[2.0, -1.0, 1.1]])
    estimated_rigid = torch.tensor([[1.4, -0.6, 0.7]])
    local = torch.zeros(1, 2, 32, 32)
    true_process = _process(base, true_rigid, local)
    estimated_process = _process(base, estimated_rigid, local)
    y_h = true_process.terminal_observation(x)

    terminal_obs = estimated_process.terminal_state(y_h, target_size=(32, 32))
    terminal_model = estimated_process.state_at(x, estimated_process.total_steps)
    residual = terminal_obs - terminal_model

    for t in (1, 5, 9, 12):
        timesteps = torch.tensor([t], dtype=torch.long)
        anchored = observation_anchored_batch_state(
            estimated_process, x, y_h, timesteps
        )
        expected = estimated_process.state_at(x, t) + residual
        assert torch.allclose(anchored, expected, atol=2e-5, rtol=2e-5)

    # Ensure this is a genuinely mismatched acquisition, not a trivial zero offset.
    assert float(residual.abs().mean()) > 1e-5
