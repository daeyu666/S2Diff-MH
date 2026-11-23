"""Oracle endpoint closure and observation-paired archive replay."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diagnose_diffusion_endpoint as entry
import diagnose_terminal_residual_correction as correction
from cdrdi_geometry import spectral_project


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FixedModel(torch.nn.Module):
    def __init__(self, reference, error):
        super().__init__()
        self.register_buffer("reference", reference)
        self.register_buffer("error", error)

    def forward(self, state, timestep):
        return self.reference + self.error * timestep[:, None, None, None]


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def source_run(tmp_path):
    args = entry.parse_args(["--device", "cpu", "--cases", "2", "--diffusion_steps", "4",
                             "--error_decomposition", "--output_dir", str(tmp_path / "source")])
    process = correction.make_process(vars(args))
    generator = torch.Generator().manual_seed(51)
    gt = torch.rand(1, 4, 128, 128, generator=generator)
    error = torch.randn(1, 4, 128, 128, generator=generator) * .005
    srf = torch.ones(2, 4) / 4
    dataset = [{"gt": gt[0], "hr_msi": spectral_project(gt, srf)[0]}]

    def estimate(observed, msi):
        return torch.tensor([[.1, -.2, .3]]), torch.zeros(1, 2, 128, 128)

    with torch.no_grad():
        entry.run_diagnostic(args, dataset, process, FixedModel(gt, error), estimate, srf, torch.device("cpu"))
    source = Path(args.output_dir)
    entry.write_json(source / "manifest.json", {"complete": True, "protocol": "paired-diffusion-endpoint-v1",
        "args": vars(args), "full_test_samples": 1, "srf_sha256": entry.tensor_hash(srf),
        "checkpoints": {"diffusion": {"path": "/not/mounted/no/checkpoint.pth", "sha256": "provenance-only"}},
        "error_decomposition": {"enabled": True}})
    return source


def test_corrected_oracle_clean_endpoint_with_zero_and_nonzero_residuals():
    rows = correction.oracle_controls()
    assert all(r["passed"] and r["corrected_error_max_abs"] < 2e-5 for r in rows)
    assert rows[0]["delta_RMS"] == 0
    assert rows[1]["delta_RMS"] > 1e-5
    assert rows[1]["original_error_RMS"] > rows[1]["corrected_error_RMS"]


def test_single_step_limit_is_last_prediction_without_gt():
    class SingleStep:
        total_steps = 1

        def state_at(self, x, t):
            return x if t == 0 else x * .5

    last = torch.rand(1, 3, 8, 8)
    terminal = torch.rand_like(last)
    original = terminal + last - SingleStep().state_at(last, 1)
    corrected, estimated_delta = correction.correct_endpoint(original, last, terminal, SingleStep())
    torch.testing.assert_close(corrected, last, atol=2e-7, rtol=0)
    torch.testing.assert_close(estimated_delta, terminal-last*.5)


def test_replay_full_entrypoint_checks_decomposition_and_leaves_source_unchanged(source_run, tmp_path):
    output = tmp_path / "corrected"
    before = hashes(source_run)
    correction.main(["--source_dir", str(source_run), "--output_dir", str(output),
                     "--device", "cpu", "--require_decomposition"])
    assert hashes(source_run) == before
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and manifest["decomposition_complete"]
    assert manifest["formal_output_changed"] is False
    assert len(manifest["summary"]) == 6
    assert manifest["source_checkpoints"]["diffusion"]["path"] == "/not/mounted/no/checkpoint.pth"
    decomposition = json.loads((output / "corrected_decomposition.json").read_text())
    assert len(decomposition) == 6
    for row in decomposition:
        assert row["original_decomposition_max_abs_error"] < 2e-5
        assert row["delta_estimate_identity_max_abs_error"] < 2e-5
        assert row["corrected_decomposition_max_abs_error"] < 2e-5
        assert row["actual_MSE_change_vs_original"] == pytest.approx(row["theoretical_MSE_change_vs_original"], abs=1e-8)
    for trace in json.loads((output / "provenance.json").read_text()):
        with np.load(output / trace["archive"]) as new, np.load(source_run / trace["archive"]) as old:
            for name in ("state_x0", "last_prediction", "terminal_state", "gt", "hr_msi", "observed_lr_hsi", "hr_valid_mask", "lr_valid_mask"):
                np.testing.assert_array_equal(new[name], old[name])
            np.testing.assert_array_equal(new["corrected_x0"], new["state_x0"]-new["estimated_terminal_residual"])
            base = correction.make_process(manifest["source_args"])
            from degradations.deformation_aware import DeformationAwareProgressiveDegradation
            process = base if trace["condition"] == "registered" else DeformationAwareProgressiveDegradation(
                base, rigid=torch.tensor(old["estimated_rigid"]), local_field=torch.tensor(old["estimated_local"]))
            expected = torch.tensor(old["terminal_state"]) - process.state_at(torch.tensor(old["last_prediction"]), 4)
            np.testing.assert_array_equal(new["estimated_terminal_residual"], expected.numpy())
    assert (output / "paired_changes.csv").is_file()
    with pytest.raises(FileExistsError):
        correction.main(["--source_dir", str(source_run), "--output_dir", str(output), "--device", "cpu"])


def test_missing_C_is_explicit_and_never_reconstructed_from_final_error(source_run, tmp_path):
    for archive in source_run.rglob("*.npz"):
        with np.load(archive) as old:
            contents = {k: old[k].copy() for k in old.files if k != "C_pred_diagnostic_only"}
        np.savez_compressed(archive, **contents)
    with pytest.raises(ValueError, match="No independently accumulated"):
        correction.main(["--source_dir", str(source_run), "--output_dir", str(tmp_path / "strict"),
                         "--device", "cpu", "--require_decomposition"])
    output = tmp_path / "metrics_only"
    correction.main(["--source_dir", str(source_run), "--output_dir", str(output), "--device", "cpu"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and not manifest["decomposition_complete"]
    assert not (output / "corrected_decomposition.csv").exists()
    assert len(manifest["summary"]) == 6


def test_corrupt_saved_input_is_rejected(source_run, tmp_path):
    archive = source_run / "warp/sample0001_case001.npz"
    with np.load(archive) as old:
        contents = {k: old[k].copy() for k in old.files}
    contents["observed_lr_hsi"] += .01
    np.savez_compressed(archive, **contents)
    with pytest.raises(ValueError, match="hash mismatch"):
        correction.main(["--source_dir", str(source_run), "--output_dir", str(tmp_path / "corrupt"), "--device", "cpu"])


def test_dry_run_is_lightweight_and_source_overwrite_is_rejected(source_run):
    code = "import sys; import diagnose_terminal_residual_correction as d; d.main(['--dry_run']); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match="Output must be separate"):
        correction.main(["--source_dir", str(source_run), "--output_dir", str(source_run), "--overwrite_output"])

