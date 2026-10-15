"""Real small CDRDI/diffusion/GIGI test through the new main-table entry."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_multiseed_deformation as entry


def test_complete_small_cpu_pipeline_and_resume(tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        argv = ["--dataset", "CAVE", "--device", "cpu", "--geometry_base_channels", "8",
                "--diffusion_base_channels", "8", "--time_dim", "32", "--spectral_hidden", "2",
                "--refine_hidden", "8", "--heads", "1", "--geometry_steps", "1", "--diffusion_steps", "4",
                "--geometry_seeds", "101", "202", "--cases", "1", "--output_dir", str(tmp_path / "out")]
        paths = {"geometry_checkpoint": tmp_path / "geometry.pth",
                 "diffusion_checkpoint": tmp_path / "diffusion.pth",
                 "refiner_checkpoint": tmp_path / "refiner.pth"}
        for option, path in paths.items():
            argv += ["--" + option, str(path)]
        args = entry.parse_args(argv)
        info = {"n_bands": 31, "n_msi_bands": 3, "srf_weights": [[1/31]*31]*3}
        cfg = entry.pipeline._config(args)
        geometry = entry.pipeline.LearnedPhysicalResidualSolver(3, base_channels=8)
        diffusion = entry.pipeline.build_model(cfg, info, torch.device("cpu"))
        refiner = entry.pipeline._build_refiner(args, info, torch.device("cpu"))
        torch.save({"model": geometry.state_dict()}, paths["geometry_checkpoint"])
        torch.save({"model": diffusion.state_dict()}, paths["diffusion_checkpoint"])
        torch.save({"model": refiner.state_dict(), "variant": "full", "epoch": 1, "best_metric": 1.0,
                    "extra": {"dataset": "CAVE"}}, paths["refiner_checkpoint"])
        data = [{"gt": torch.rand(31, 32, 32), "hr_msi": torch.rand(3, 32, 32)} for _ in range(2)]
        monkeypatch.setattr(entry.pipeline, "build_loaders", lambda cfg: (None, SimpleNamespace(dataset=data), info))
        result = entry.main(argv)
        assert result["complete"] and result["Warp"]["n_geometry_seeds"] == 2
        process = entry.pipeline.build_progressive_process(cfg)
        with torch.no_grad():
            registered = [entry.pipeline._registered_pair(diffusion, refiner,
                entry.pipeline._build_projector(args, info, "cpu"), process,
                item["gt"][None], item["hr_msi"][None])[1] for item in data]
        expected = sum(entry.pipeline.calc_metrics(p, item["gt"][None], 4)["PSNR"] for p, item in zip(registered, data)) / 2
        assert abs(result["Registered"]["PSNR"] - expected) < 1e-5
        resumed = entry.main(argv + ["--resume_output"])
        assert resumed == result
        disk = json.loads((tmp_path / "out" / "summary.json").read_text())
        assert disk["test_conditions"]["warp_observations"] == 4
    finally:
        torch.set_num_threads(old_threads)


def test_default_is_ten_geometry_seeds_and_ten_cases():
    args = entry.parse_args(["--dataset", "CAVE", "--dry_run"])
    assert args.geometry_seeds == list(range(10001, 10011))
    assert args.cases == 10
