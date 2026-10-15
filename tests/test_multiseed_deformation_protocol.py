"""Identical protocol tests in both repositories; no trained weights required."""
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import multiseed_deformation_protocol as protocol


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dataset():
    grid = torch.linspace(0.1, 0.8, 32).repeat(32, 1)
    return [{"gt": torch.stack([grid, grid.T]) + i * 0.01,
             "hr_msi": torch.full((1, 32, 32), (i + 1) / 4)} for i in range(2)]


def test_geometry_is_private_signed_bounded_and_independent_per_patch():
    state = torch.random.get_rng_state()
    records = []
    for seed, sample, case in ((10001, 0, 0), (10001, 1, 0), (10001, 0, 1), (10002, 0, 0)):
        phi, record = protocol.sample_geometry(128, 128, geometry_seed=seed, sample_index=sample, case_index=case)
        again, repeated = protocol.sample_geometry(128, 128, geometry_seed=seed, sample_index=sample, case_index=case)
        assert record == repeated
        assert -4 <= phi.dx.item() <= 4 and -4 <= phi.dy.item() <= 4
        assert -2 <= phi.theta_deg.item() <= 2
        assert 0 <= record["local_max_hr_px"] <= 4 + 1e-5
        assert record["min_local_jacobian"] >= 0.5
        torch.testing.assert_close(phi.local_field, again.local_field, rtol=0, atol=0)
        records.append(record)
    assert len({r["local_field_sha256"] for r in records}) == 4
    assert torch.equal(state, torch.random.get_rng_state())
    # Confirm the new protocol does sample weak local fields, unlike legacy S2.
    amplitudes = [protocol.sample_geometry(32, 32, geometry_seed=123, sample_index=0, case_index=i)[1]["local_max_hr_px"] for i in range(20)]
    assert min(amplitudes) < 2.6


def test_full_split_seed_means_std_and_paired_reconstruction_rng(tmp_path):
    data = dataset()
    warps, registered, noise = [], [], []
    def predict(lr, msi):
        warps.append(protocol.tensor_sha256(lr))
        draws = (random.random(), float(np.random.rand()), torch.randn(()).item())
        noise.append(draws)
        return F.interpolate(lr, scale_factor=4, mode="nearest") + draws[2] * 1e-4
    def reg(lr, msi):
        registered.append(protocol.tensor_sha256(lr))
        return F.interpolate(lr, scale_factor=4, mode="nearest")
    state, py, npstate = torch.random.get_rng_state(), random.getstate(), np.random.get_state()
    result = protocol.run_multiseed_suite(data, predict, lambda x: F.avg_pool2d(x, 4),
        predict_registered=reg, device="cpu", output_dir=tmp_path, metadata={"method": "test"},
        geometry_seeds=[101, 202, 303], cases=2)
    assert result["complete"] and len(warps) == 12 and len(registered) == 2
    assert noise[:4] == noise[4:8] == noise[8:]
    assert torch.equal(state, torch.random.get_rng_state())
    assert py == random.getstate()
    assert npstate[0] == np.random.get_state()[0] and np.array_equal(npstate[1], np.random.get_state()[1])
    expected_inputs = []
    for seed in (101, 202, 303):
        rows = [json.loads(line) for line in (tmp_path / f"seed_{seed}.jsonl").read_text().splitlines()]
        assert len(rows) == 4
        for i, row in enumerate(rows):
            sample, case = divmod(i, 2)
            phi, _ = protocol.sample_geometry(32, 32, geometry_seed=seed, sample_index=sample, case_index=case)
            expected = F.avg_pool2d(protocol.warp_hsi(data[sample]["gt"].unsqueeze(0), phi), 4)
            expected_inputs.append(protocol.tensor_sha256(expected))
            assert row["observed_sha256"] == expected_inputs[-1]
        seed_record = json.loads((tmp_path / f"seed_{seed}.json").read_text())
        for key in protocol.METRIC_KEYS:
            assert seed_record["average_metrics"][key] == pytest.approx(mean(r["metrics"][key] for r in rows))
    assert warps == expected_inputs
    for key in protocol.METRIC_KEYS:
        means = [r["average_metrics"][key] for r in result["per_seed"]]
        assert result["Warp"]["mean"][key] == mean(means)
        assert result["Warp"]["std"][key] == stdev(means)
    assert "std" not in result["Registered"]
    assert result["test_conditions"]["warp_observations"] == 12


def test_resume_reuses_completed_seeds_and_rejects_changed_model_or_data(tmp_path):
    data = dataset()
    calls = []
    def predict(lr, msi):
        calls.append(1)
        return F.interpolate(lr, scale_factor=4)
    kwargs = dict(device="cpu", output_dir=tmp_path, metadata={"checkpoint_sha256": "a"},
                  geometry_seeds=[123, 456], cases=1)
    original = protocol.run_multiseed_suite(data, predict, lambda x: F.avg_pool2d(x, 4), **kwargs)
    calls.clear()
    resumed = protocol.run_multiseed_suite(data, predict, lambda x: F.avg_pool2d(x, 4), resume_output=True, **kwargs)
    assert not calls and original == resumed
    with pytest.raises(ValueError, match="mismatch"):
        protocol.run_multiseed_suite(data, predict, lambda x: x, resume_output=True,
                                    **{**kwargs, "metadata": {"checkpoint_sha256": "b"}})
    data[0]["gt"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="mismatch"):
        protocol.run_multiseed_suite(data, predict, lambda x: x, resume_output=True, **kwargs)


def test_overwrite_discards_old_seed_markers_before_interrupted_new_run(tmp_path):
    data = dataset()
    predict = lambda lr, msi: F.interpolate(lr, scale_factor=4)
    kwargs = dict(device="cpu", output_dir=tmp_path, metadata={}, geometry_seeds=[123, 456], cases=1)
    protocol.run_multiseed_suite(data, predict, lambda x: F.avg_pool2d(x, 4), **kwargs)
    def fail(lr, msi):
        raise RuntimeError("interruption")
    with pytest.raises(RuntimeError, match="interruption"):
        protocol.run_multiseed_suite(data, fail, lambda x: F.avg_pool2d(x, 4), overwrite_output=True, **kwargs)
    assert not (tmp_path / "seed_123.json").exists()
    assert not (tmp_path / "seed_456.json").exists()


def test_std_is_across_seed_means_and_single_seed_has_no_estimated_std():
    rows = [{key: value for key in protocol.METRIC_KEYS} for value in (1.0, 3.0)]
    result = protocol.seed_statistics(rows)
    assert result["mean"]["PSNR"] == 2
    assert result["std"]["PSNR"] == pytest.approx(2 ** 0.5)
    assert protocol.seed_statistics(rows[:1])["std"]["PSNR"] is None


@pytest.mark.parametrize("seeds,cases", [([], 10), ([1, 1], 10), ([-1], 10), ([1], 0)])
def test_invalid_protocol_fails_before_dataset_or_prediction(tmp_path, seeds, cases):
    with pytest.raises(ValueError):
        protocol.run_multiseed_suite(None, None, None, device="cpu", output_dir=tmp_path,
                                    metadata={}, geometry_seeds=seeds, cases=cases)


def test_sibling_protocol_and_case_generation_are_identical():
    sibling = ROOT.parent / ("comparison_experiments" if ROOT.name == "S2Diff-MH" else "S2Diff-MH")
    path = sibling / "multiseed_deformation_protocol.py"
    if not path.is_file():
        pytest.skip("Sibling checkout not present")
    assert path.read_bytes() == (ROOT / path.name).read_bytes()
    assert (sibling / "fixed_deformation_protocol.py").read_bytes() == (ROOT / "fixed_deformation_protocol.py").read_bytes()
    spec = importlib.util.spec_from_file_location("sibling_multiseed", path)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    phi, record = protocol.sample_geometry(128, 128, geometry_seed=10001, sample_index=7, case_index=3)
    second, repeated = other.sample_geometry(128, 128, geometry_seed=10001, sample_index=7, case_index=3)
    assert record == repeated
    torch.testing.assert_close(phi.local_field, second.local_field, atol=0, rtol=0)
    # Also compare each repository's actual P0, in isolated module namespaces.
    code = """
import json, torch
from degradations.physical import PhysicalDegradation
from multiseed_deformation_protocol import sample_geometry, warp_hsi, tensor_sha256
torch.set_num_threads(1)
gt = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(771))
phi, record = sample_geometry(32, 32, geometry_seed=10001, sample_index=7, case_index=3)
observed = PhysicalDegradation(scale_ratio=4, mtf_nyquist=0.2, truncate=3).degrade(warp_hsi(gt, phi))
print(json.dumps({'observation': tensor_sha256(observed), 'geometry': record}))
"""
    outputs = [subprocess.check_output([sys.executable, "-c", code], cwd=path, text=True) for path in (ROOT, sibling)]
    assert json.loads(outputs[0]) == json.loads(outputs[1])
