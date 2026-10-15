"""Protocol tests shared verbatim with S2Diff-MH; no real data/weights needed."""
import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fixed_deformation_protocol as protocol


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_exact_six_requested_strengths_and_local_pairing():
    assert [(s.translation_hr_px, s.rotation_deg, s.local_max_hr_px) for s in protocol.STRENGTHS] == [
        (0.5, 0.25, 0.5), (1, 0.5, 1), (2, 1, 2), (3, 1.5, 3), (4, 2, 4), (6, 3, 4)]
    previous = torch.random.get_rng_state()
    records = []
    for strength in protocol.STRENGTHS:
        geometry, record = protocol.sample_fixed_geometry(
            128, 128, strength, sample_index=2, case_index=3)
        assert geometry.dx.item() == geometry.dy.item() == strength.translation_hr_px
        assert geometry.theta_deg.item() == strength.rotation_deg
        assert 0 <= record["local_max_hr_px"] <= strength.local_max_hr_px + 1e-5
        assert record["min_local_jacobian"] >= 0.5
        repeated, again = protocol.sample_fixed_geometry(
            128, 128, strength, sample_index=2, case_index=3)
        assert again == record
        torch.testing.assert_close(repeated.local_field, geometry.local_field, rtol=0, atol=0)
        records.append(record)
    assert torch.equal(torch.random.get_rng_state(), previous)
    assert records[4]["local_field_sha256"] == records[5]["local_field_sha256"]
    _, different = protocol.sample_fixed_geometry(128, 128, protocol.STRENGTHS[4], sample_index=3, case_index=3)
    assert different["local_field_sha256"] != records[4]["local_field_sha256"]


def test_warp_matches_existing_local_then_rigid_sampler():
    # Import the local repository's existing geometry implementation without
    # making either repository depend on the sibling checkout.
    source = ROOT / "cdrdi_geometry.py"
    if not source.is_file():
        source = ROOT / "comparison" / "UAFL" / "hsi_deformation.py"
    spec = importlib.util.spec_from_file_location("fixed_test_existing_geometry", source)
    existing = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = existing
    spec.loader.exec_module(existing)
    image = torch.rand(1, 3, 128, 128)
    geometry, _ = protocol.sample_fixed_geometry(128, 128, protocol.STRENGTHS[5], sample_index=0, case_index=0)
    if source.name == "cdrdi_geometry.py":
        expected = existing.forward_warp(image, geometry.dx, geometry.dy, geometry.theta_deg, geometry.local_field)
    else:
        expected = existing.forward_warp(image, geometry)
    torch.testing.assert_close(protocol.warp_hsi(image, geometry), expected, rtol=0, atol=0)


def test_private_model_rng_and_all_patch_case_means(tmp_path):
    class TestSet:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            grid = torch.linspace(0.1, 0.8, 32).repeat(32, 1)
            return {"gt": torch.stack([grid, grid.T]),
                    "hr_msi": torch.full((1, 32, 32), (index+1)/4)}

    inputs, noise = [], []

    def predict(lr_hsi, hr_msi):
        inputs.append((lr_hsi.clone(), hr_msi.clone()))
        draw = torch.randn(1).item()
        noise.append(draw)
        return F.interpolate(lr_hsi, scale_factor=4, mode="nearest") + draw*1e-4

    random_state = torch.random.get_rng_state()
    summary = protocol.run_fixed_suite(
        TestSet(), predict, lambda x: F.avg_pool2d(x, 4), device="cpu",
        output_dir=tmp_path, metadata={"method": "test"}, cases=2,
    )
    assert torch.equal(torch.random.get_rng_state(), random_state)
    assert len(inputs) == 6*3*2
    assert noise[:6] == noise[6:12] == noise[-6:]
    assert len(summary["results"]) == 6
    for result in summary["results"]:
        level = result["level"]
        assert result["test_patch_count"] == 3
        assert result["evaluated_cases"] == 6
        detail = json.loads((tmp_path / f"level_{level}.json").read_text())
        assert len(detail["test_inputs"]) == 3
        assert len(detail["per_patch_metrics"]) == 3
        assert len(detail["per_sample_case"]) == 6
        for key in protocol.METRIC_KEYS:
            expected = sum(r["metrics"][key] for r in detail["per_sample_case"])/6
            assert result["average_metrics"][key] == pytest.approx(expected)
        for row in detail["per_sample_case"]:
            assert row["dx_hr_px"] == row["dy_hr_px"] == result["translation_hr_px"]
            assert row["rotation_deg"] == result["rotation_deg"]
    for n, (_, hr_msi) in enumerate(inputs):
        assert hr_msi.unique().item() == ((n//2) % 3+1)/4
    with (tmp_path / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert float(rows[-1]["translation_hr_px"]) == 6
    assert float(rows[-1]["rotation_deg"]) == 3
    fifth = json.loads((tmp_path / "level_5.json").read_text())
    sixth = json.loads((tmp_path / "level_6.json").read_text())
    assert [r["local_field_sha256"] for r in fifth["per_sample_case"]] == [
        r["local_field_sha256"] for r in sixth["per_sample_case"]]


@pytest.mark.parametrize("levels,cases", [([1, 1], 1), ([7], 1), ([], 1), ([1], 0)])
def test_invalid_suite_configuration_fails_before_inference(tmp_path, levels, cases):
    with pytest.raises(ValueError):
        protocol.run_fixed_suite([None], None, None, device="cpu", output_dir=tmp_path,
                                 metadata={}, levels=levels, cases=cases)


def test_metrics_keep_existing_s2_final_formulas():
    source = ROOT / "metrics.py"
    if (ROOT / "comparison").is_dir():
        source = ROOT / "comparison" / "EMR-Diff" / "test_hsi_deformed.py"
    spec = importlib.util.spec_from_file_location("fixed_test_original_metrics", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    target = torch.rand(1, 3, 16, 16)
    prediction = target + torch.randn_like(target)*0.1
    expected = original.calc_metrics(prediction, target, 4)
    actual = protocol.calc_metrics(prediction, target, 4)
    for key in protocol.METRIC_KEYS:
        assert actual[key] == pytest.approx(expected[key], rel=2e-6, abs=2e-6)
