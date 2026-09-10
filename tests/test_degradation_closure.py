import torch

from degradations import ProgressiveDegradation, build_degradation


def test_physical_terminal_closure_and_shapes():
    x = torch.rand(2, 7, 64, 64)
    operator = build_degradation(
        "physical", scale_ratio=4, mtf_nyquist=0.2, truncate=3.0
    )
    process = ProgressiveDegradation(operator, total_steps=12)
    process.assert_terminal_closure(x)
    assert process.state(4).scale == 1
    assert process.state(5).scale == 2
    assert process.state(9).scale == 4
    assert process.state_at(x, 12).shape == x.shape
    assert process.terminal_observation(x).shape[-2:] == (16, 16)


def test_reverse_update_keeps_hr_grid():
    x0 = torch.rand(1, 5, 32, 32)
    process = ProgressiveDegradation(
        build_degradation("physical", scale_ratio=4), total_steps=12
    )
    x_t = process.state_at(x0, 12)
    out = process.reverse_update(x_t, x0, 12)
    assert out.shape == x0.shape
    assert torch.isfinite(out).all()
