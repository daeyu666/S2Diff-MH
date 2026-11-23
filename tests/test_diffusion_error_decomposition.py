"""Independent per-step sum, fixed-linear identity and genuine net compensation."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diagnose_diffusion_endpoint as entry
import train_innovation3_gigi_cdrdi as pipeline
from cdrdi_geometry import spectral_project
from diffusion_error_decomposition import PredictionErrorCollector, measure_terms, summarize_decomposition
from innovation1 import reconstruct_from_terminal_lr


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FixedPrediction(torch.nn.Module):
    def __init__(self, reference, error):
        super().__init__()
        self.register_buffer("reference", reference)
        self.register_buffer("error", error)
        self.calls = 0

    def forward(self, state, timestep):
        self.calls += 1
        return self.reference + self.error * timestep[:, None, None, None]


def setup(tmp_path, *extra):
    args = entry.parse_args(["--device", "cpu", "--diffusion_steps", "4", "--cases", "2",
                             "--output_dir", str(tmp_path / "out"), *extra])
    base = pipeline.build_progressive_process(pipeline._config(args))
    return args, base


@pytest.mark.parametrize("mismatched", [False, True])
def test_oracle_zero_C_and_zero_or_nonzero_delta(tmp_path, mismatched):
    _, base = setup(tmp_path)
    gt = torch.rand(1, 3, 64, 64)
    local = torch.zeros(1, 2, 64, 64)
    truth = pipeline._process_from_estimate(base, torch.tensor([[2., -1., .8]]), local)
    acquisition = pipeline._process_from_estimate(base, torch.tensor([[1.4, -.5, .2]]), local) if mismatched else truth
    observed = truth.terminal_observation(gt)
    collector = PredictionErrorCollector(acquisition, gt, torch.ones(1, 1, 64, 64, dtype=torch.bool))
    pair = entry.reconstruct_pair(FixedPrediction(gt, torch.zeros_like(gt)), acquisition,
                                  observed, target_size=(64, 64), hr_msi=None, step_observer=collector.observe)
    rows, tensors = collector.finish(pair)
    assert torch.count_nonzero(tensors["C_pred_diagnostic_only"]) == 0
    assert all(r["cos_delta_C"] is None and not r["net_compensation"] for r in rows)
    assert rows[0]["decomposition_max_abs_error"] < 2e-5
    if mismatched:
        assert rows[0]["delta_RMS"] > 1e-5
    else:
        assert rows[0]["delta_RMS"] == 0


def test_independent_step_sum_matches_actual_error_and_production_state(tmp_path):
    args, base = setup(tmp_path)
    gt, error = torch.rand(1, 3, 64, 64), torch.randn(1, 3, 64, 64) * .02
    process = pipeline._process_from_estimate(base, torch.tensor([[1.2, -.4, .7]]), torch.zeros(1, 2, 64, 64))
    observed = base.terminal_observation(gt)
    model = FixedPrediction(gt, error)
    original = reconstruct_from_terminal_lr(model, process, observed, target_size=(64, 64))
    model.calls = 0
    collector = PredictionErrorCollector(process, gt, torch.ones(1, 1, 64, 64, dtype=torch.bool), save_step_errors=True)
    rng_before = torch.get_rng_state().clone()
    pair = entry.reconstruct_pair(model, process, observed, target_size=(64, 64), hr_msi=None, step_observer=collector.observe)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)
    assert model.calls == args.diffusion_steps
    torch.testing.assert_close(pair["state_x0"], original, atol=0, rtol=0)
    rows, tensors = collector.finish(pair)
    # Recover independently from saved e_t fields, not from x0-X-delta.
    expected = torch.zeros_like(gt, dtype=torch.float64)
    for t in range(args.diffusion_steps, 0, -1):
        e = tensors[f"prediction_error_t{t:02d}"]
        expected += process.state_at(e, t-1).double() - process.state_at(e, t).double()
    torch.testing.assert_close(tensors["C_pred_diagnostic_only"], expected, atol=0, rtol=0)
    assert all(r["decomposition_max_abs_error"] < 2e-5 for r in rows)
    assert [r["t"] for r in collector.steps if r["scope"] == "full"] == [4, 3, 2, 1]


def test_negative_cosine_does_not_imply_net_compensation():
    delta = torch.ones(1, 2, 3, 3)
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    for scale, expected in ((-.5, True), (-3., False), (.5, False)):
        correction = delta * scale
        row = measure_terms(delta, correction, delta+correction, mask)
        assert row["net_compensation"] == expected
        assert row["opposing_component"] == (scale < 0)
        assert row["energy_identity_abs_error"] < 1e-12
    row = measure_terms(torch.zeros_like(delta), delta, delta, mask)
    assert row["cos_delta_C"] is None and not row["net_compensation"]


def test_invalid_nonlinear_trajectory_fails_instead_of_claiming_compensation():
    class Nonlinear:
        total_steps = 2

        def state_at(self, x, t):
            return x if t == 0 else x.square() * (t+1)

        def terminal_state(self, observed, *, target_size):
            return observed

        def reverse_update(self, state, prediction, t):
            return state + self.state_at(prediction, t-1) - self.state_at(prediction, t)

    process = Nonlinear()
    gt = torch.ones(1, 2, 4, 4) * .5
    collector = PredictionErrorCollector(process, gt, torch.ones(1, 1, 4, 4, dtype=torch.bool))
    pair = entry.reconstruct_pair(FixedPrediction(gt, torch.ones_like(gt)*.1), process,
                                  process.state_at(gt, 2), target_size=(4, 4), hr_msi=None, step_observer=collector.observe)
    with pytest.raises(AssertionError, match="fixed linear"):
        collector.finish(pair)


def test_runner_exports_paired_full_roi_and_preserves_old_outputs(tmp_path):
    args, base = setup(tmp_path)
    gt = torch.rand(1, 3, 128, 128)
    error = torch.rand_like(gt) * .01
    srf = torch.ones(2, 3) / 3
    dataset = [{"gt": gt[0], "hr_msi": spectral_project(gt, srf)[0]}]

    def estimate(observed, msi):
        return torch.zeros(1, 3), torch.zeros(1, 2, 128, 128)

    with torch.no_grad():
        before = entry.run_diagnostic(args, dataset, base, FixedPrediction(gt, error), estimate,
                                      srf, torch.device("cpu"))
        args.output_dir = str(tmp_path / "decomposed")
        args.error_decomposition, args.save_step_errors = True, True
        after = entry.run_diagnostic(args, dataset, base, FixedPrediction(gt, error), estimate,
                                     srf, torch.device("cpu"))
    assert before == after
    output = Path(args.output_dir)
    rows = json.loads((output / "error_decomposition.json").read_text())
    assert len(rows) == (1 + args.cases) * 2
    assert all(r["decomposition_max_abs_error"] < 2e-5 for r in rows)
    summary = summarize_decomposition(rows)
    assert len(summary) == 4
    registered = next(r for r in summary if r["condition"] == "registered" and r["scope"] == "full")
    assert registered["defined_cosine_cases"] == 0 and registered["mean_cos_delta_C"] is None
    for trace in json.loads((output / "traces.json").read_text()):
        with np.load(output / trace["archive"]) as data:
            assert data["C_pred_diagnostic_only"].dtype == np.float64
            assert "prediction_error_t04" in data
            original = np.load(tmp_path / "out" / trace["archive"])
            np.testing.assert_array_equal(data["state_x0"], original["state_x0"])
            np.testing.assert_array_equal(data["last_prediction"], original["last_prediction"])
            original.close()
    assert (output / "error_decomposition_steps.csv").is_file()
    assert (output / "error_decomposition_summary.csv").is_file()
