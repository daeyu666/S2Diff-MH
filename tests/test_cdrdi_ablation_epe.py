"""Known-field errors and offline table recovery for the CDRDI ablation."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paviau_cdrdi_iteration_ablation as ablation


def geometry(h=5, w=6, *, dx=0., dy=0., angle=0., local=(0., 0.)):
    field = torch.empty((1, 2, h, w), dtype=torch.float64)
    field[:, 0], field[:, 1] = local
    scalar = lambda v: torch.tensor([v], dtype=torch.float64)
    return SimpleNamespace(dx=scalar(dx), dy=scalar(dy), theta_deg=scalar(angle), local_field=field)


def test_translation_epe_is_in_hr_pixels_and_does_not_clip_coordinates():
    gt = geometry(dx=3., dy=4.)
    metrics = ablation._geometry_metrics(gt, torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros_like(gt.local_field), height=5, width=6)
    assert metrics["EPE_mean_px"] == pytest.approx(5.)
    assert metrics["EPE_p95_px"] == pytest.approx(5.)


def test_full_composition_includes_rotation_and_local_displacement():
    gt = geometry(dx=3., dy=4., angle=90., local=(1., 2.))
    yy, xx = torch.meshgrid(torch.arange(5, dtype=torch.float64), torch.arange(6, dtype=torch.float64), indexing="ij")
    expected_x = -(yy + 2. - 2.) + 2.5 + 3.
    expected_y = (xx + 1. - 2.5) + 2. + 4.
    expected = ((expected_x - xx).square() + (expected_y - yy).square()).sqrt()
    metrics = ablation._geometry_metrics(gt, torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros_like(gt.local_field), height=5, width=6)
    assert metrics["EPE_mean_px"] == pytest.approx(expected.mean().item())
    assert metrics["EPE_p95_px"] == pytest.approx(torch.quantile(expected.flatten(), .95).item())
    perfect = ablation._geometry_metrics(gt, torch.stack((gt.dx, gt.dy, gt.theta_deg), dim=1),
        gt.local_field, height=5, width=6)
    assert perfect["EPE_mean_px"] == perfect["EPE_p95_px"] == 0.


def test_offline_summary_recovers_stored_macro_metrics_without_loading_models(tmp_path, monkeypatch):
    rows = [dict(Variant=name, PSNR=40.+i, SAM=2.-i*.1, EPE_mean_px=.5+i*.1, EPE_p95_px=1.+i*.2)
            for i, name in enumerate(("w/o CDRDI", "One-shot", "Recursive K=9"))]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(protocol_id="PaviaU-CDRDI-iteration-ablation-v1", depth_diagnostics=[9], summary=rows)))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline recovery loaded model/data or ran inference")
    monkeypatch.setattr(ablation, "_validate_checkpoints", forbidden)
    monkeypatch.setattr(ablation, "run", forbidden)
    monkeypatch.setattr(ablation.pipeline, "build_loaders", forbidden)
    (tmp_path / "summary.csv").write_text("Variant,PSNR,SAM\nold,1,2\n")
    ablation.main(["--summarize_only", "--output_dir", str(tmp_path), "--geometry_checkpoint", "missing.pth"])
    with (tmp_path / "summary.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["Variant", "PSNR", "SAM", "EPE", "EPE95"]
        result = list(reader)
    for saved, row in zip(result, rows):
        assert float(saved["PSNR"]) == row["PSNR"]
        assert float(saved["SAM"]) == row["SAM"]
        assert float(saved["EPE"]) == row["EPE_mean_px"]
        assert float(saved["EPE95"]) == row["EPE_p95_px"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    definition = json.loads((tmp_path / "epe_metric_definition.json").read_text())
    assert definition["units"] == "HR pixels" and "not a pooled percentile" in definition["aggregation"]


def test_incomplete_or_invalid_stored_metrics_never_overwrite_old_table(tmp_path):
    path = tmp_path / "manifest.json"
    old = tmp_path / "summary.csv"
    old.write_text("original table")
    row = dict(Variant="One-shot", PSNR=40., SAM=2., EPE_mean_px=.5)
    path.write_text(json.dumps(dict(protocol_id="PaviaU-CDRDI-iteration-ablation-v1", depth_diagnostics=[], summary=[row])))
    with pytest.raises(ValueError, match="lacks valid"):
        ablation.summarize_existing(tmp_path)
    row["EPE_p95_px"] = 1.
    path.write_text(json.dumps(dict(protocol_id="PaviaU-CDRDI-iteration-ablation-v1", depth_diagnostics=[3, 9], summary=[row])))
    with pytest.raises(ValueError, match="all declared"):
        ablation.summarize_existing(tmp_path)
    assert old.read_text() == "original table"
