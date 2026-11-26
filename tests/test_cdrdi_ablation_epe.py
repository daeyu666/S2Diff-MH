"""Known-field errors and offline table recovery for the CDRDI ablation."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np

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


@pytest.mark.parametrize("mode", ["state", "terminal_correction"])
def test_full_rerun_routes_all_geometry_conditions_through_the_same_output_mode(tmp_path, monkeypatch, mode):
    from models import RawMSIDirectPredictor
    from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
    args = ablation.parse_args(["--device", "cpu", "--cases", "1", "--depths", "9",
        "--geometry_base_channels", "8", "--diffusion_base_channels", "8", "--time_dim", "32",
        "--spectral_hidden", "2", "--diffusion_steps", "4", "--diffusion_output", mode,
        "--output_dir", str(tmp_path / mode)])
    geometry = LearnedPhysicalResidualSolver(3, base_channels=8, control_grid=5,
        max_translation=4., max_rotation_deg=2., max_local_px=4.)
    diffusion = RawMSIDirectPredictor(31,3,total_steps=4,base_channels=8,time_dim=32,spectral_hidden=2)
    for role, model, extra in [
        ("geometry", geometry, {"dataset":"PaviaU", "variant":"recursive", "steps":6}),
        ("one_shot_geometry", geometry, {"dataset":"PaviaU", "variant":"one_shot", "steps":1}),
        ("diffusion", diffusion, {"dataset":"PaviaU", "geometry_steps":9})]:
        path = tmp_path / (role + ".pth")
        torch.save({"model":model.state_dict(), "extra":extra}, path)
        setattr(args, role + "_checkpoint", str(path))
    gt = torch.rand(1,31,64,64)
    sample = {"gt":gt, "hr_msi":gt[:,:3]}
    info = dict(n_bands=31,n_msi_bands=3,srf_weights=np.full((3,31),1/31))
    monkeypatch.setattr(ablation.pipeline, "build_loaders", lambda cfg:(None,[sample],info))
    calls=[]
    original=ablation.pipeline.reconstruct_from_terminal_lr
    def reconstruct(*a, **kw):
        calls.append((kw["output_mode"], ablation._sha256_tensor(a[2])))
        return original(*a, **kw)
    monkeypatch.setattr(ablation.pipeline, "reconstruct_from_terminal_lr", reconstruct)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ablation.run(args)
    finally:
        torch.set_num_threads(old)
    assert len(calls)==3 and all(value[0]==mode for value in calls)
    assert len({value[1] for value in calls})==1
    manifest=json.loads((tmp_path / mode / "manifest.json").read_text())
    assert manifest["diffusion_output"]==mode
    with (tmp_path / mode / "summary.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==3 and all("EPE" in row and "EPE95" in row for row in rows)
