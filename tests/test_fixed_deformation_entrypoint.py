"""Load real small CPU models through the new complete-method test adapter."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("s2_fixed_entry", ROOT / "test_fixed_deformation.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_actual_small_models_load_and_predict_without_gt_geometry(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args = entry.parse_args([
            "--dataset", "CAVE", "--device", "cpu", "--geometry_base_channels", "8",
            "--diffusion_base_channels", "8", "--time_dim", "32", "--spectral_hidden", "2",
            "--refine_hidden", "8", "--heads", "1", "--geometry_steps", "1", "--diffusion_steps", "4",
        ])
        args.geometry_checkpoint = str(tmp_path / "geometry.pth")
        args.diffusion_checkpoint = str(tmp_path / "diffusion.pth")
        args.refiner_checkpoint = str(tmp_path / "refiner.pth")
        device = torch.device("cpu")
        info = {"n_bands": 31, "n_msi_bands": 3, "srf_weights": [[1/31]*31]*3}
        geometry_model = entry.pipeline.LearnedPhysicalResidualSolver(3, base_channels=8)
        cfg = entry.pipeline._config(args)
        diffusion_model = entry.pipeline.build_model(cfg, info, device)
        refiner = entry.pipeline._build_refiner(args, info, device)
        torch.save({"model": geometry_model.state_dict()}, args.geometry_checkpoint)
        torch.save({"model": diffusion_model.state_dict()}, args.diffusion_checkpoint)
        torch.save({"model": refiner.state_dict(), "variant": "full", "epoch": 7,
                    "best_metric": 1.2, "extra": {"dataset": "CAVE"}}, args.refiner_checkpoint)
        process = entry.pipeline.build_progressive_process(cfg)
        predict, details = entry.build_predictor(args, info, process, device)
        with torch.no_grad():
            prediction = predict(torch.rand(1, 31, 8, 8), torch.rand(1, 3, 32, 32))
        assert prediction.shape == (1, 31, 32, 32)
        assert torch.isfinite(prediction).all()
        assert details["model_geometry_bounds"] == [4.0, 2.0, 4.0]
        assert details["checkpoint_epoch"] == 7
    finally:
        torch.set_num_threads(previous)


def test_old_cli_and_cases_alias_are_preserved(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_innovation3_gigi_cdrdi.py", "--eval_cases", "3"])
    old = entry.pipeline.parse_args()
    assert old.stage == "train"
    assert old.eval_cases == 3
    assert entry.parse_args(["--eval_cases", "3"]).cases == 3
    assert entry.parse_args(["--cases", "2"]).cases == 2
    with pytest.raises(SystemExit):
        entry.parse_args(["--max_translation", "6"])


def test_dry_run_entrypoint_needs_no_dataset_or_checkpoint(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "test_fixed_deformation.py"), "--dataset", "CAVE", "--dry_run"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"level": 6' in result.stdout
    assert '"translation_hr_px": 6.0' in result.stdout
