"""Check geometry routing, exact observation pairing and signed diagnostic gaps."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diagnose_cdrdi_robustness as entry
import train_innovation3_gigi_cdrdi as pipeline
from cdrdi_geometry import SyntheticGeometry, sample_synthetic_geometry
from fixed_deformation_protocol import STRENGTHS, sample_fixed_geometry, tensor_sha256, warp_hsi


@pytest.fixture
def small_setup(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args = entry.parse_args([
            "--dataset", "CAVE", "--device", "cpu", "--geometry_base_channels", "8",
            "--diffusion_base_channels", "8", "--time_dim", "32", "--spectral_hidden", "2",
            "--refine_hidden", "8", "--heads", "1", "--geometry_steps", "1",
            "--diffusion_steps", "4", "--cases", "2", "--protocol", "both", "--levels", "1",
            "--refiner_mode", "shared", "--output_dir", str(tmp_path / "out"),
        ])
        info = {"n_bands": 31, "n_msi_bands": 3, "srf_weights": [[1 / 31] * 31] * 3}
        device = torch.device("cpu")
        cfg = pipeline._config(args)
        process = pipeline.build_progressive_process(cfg)
        geometry = pipeline.LearnedPhysicalResidualSolver(3, base_channels=8)
        diffusion = pipeline.build_model(cfg, info, device)
        refiner = pipeline._build_refiner(args, info, device)
        args.geometry_checkpoint = str(tmp_path / "geometry.pth")
        args.diffusion_checkpoint = str(tmp_path / "adapted.pth")
        args.registered_checkpoint = str(tmp_path / "registered.pth")
        args.refiner_checkpoint = str(tmp_path / "refiner.pth")
        torch.save({"model": geometry.state_dict(), "extra": {"dataset": "CAVE"}}, args.geometry_checkpoint)
        torch.save({"model": diffusion.state_dict(), "epoch": 2}, args.diffusion_checkpoint)
        with torch.no_grad():
            diffusion.out_conv.bias.add_(0.01)
        torch.save({"state_dict": {"module." + k: v for k, v in diffusion.state_dict().items()},
                    "epoch": 1}, args.registered_checkpoint)
        torch.save({"model": refiner.state_dict(), "variant": "full", "epoch": 3,
                    "extra": {"dataset": "CAVE"}}, args.refiner_checkpoint)
        yield args, info, process, device
    finally:
        torch.set_num_threads(previous_threads)


def test_exact_estimate_equals_oracle_and_identity_equals_naive(small_setup, monkeypatch):
    args, info, process, device = small_setup
    predict, oracle, _, records = entry.build_models(args, info, process, device)
    assert records["registered_backbone"]["epoch"] == 1
    gt, msi = torch.rand(1, 31, 32, 32), torch.rand(1, 3, 32, 32)
    zero = torch.zeros(1)
    identity = SyntheticGeometry(zero, zero, zero, torch.zeros(1, 2, 5, 5), torch.zeros(1, 2, 32, 32))
    fixed, _ = sample_fixed_geometry(32, 32, STRENGTHS[1], sample_index=0, case_index=0)
    for truth in (identity, fixed):
        rigid = torch.stack([truth.dx, truth.dy, truth.theta_deg], dim=1)
        monkeypatch.setattr(pipeline, "_estimate_geometry", lambda *a, **kw: (rigid, truth.local_field))
        with torch.no_grad():
            observed = pipeline._process_from_geometry(process, truth).terminal_observation(gt)
            predictions, _, _ = predict(observed, msi)
            oracle_outputs = oracle(observed, msi, truth)
        for stage in ("base", "refined"):
            torch.testing.assert_close(predictions["estimated"][stage], oracle_outputs[stage], atol=1e-6, rtol=1e-6)
            if truth is identity:
                torch.testing.assert_close(predictions["adapted_naive"][stage], oracle_outputs[stage], atol=2e-5, rtol=2e-5)


def test_full_runner_preserves_legacy_and_fixed_inputs(small_setup):
    args, info, process, device = small_setup
    predict, oracle, srf, _ = entry.build_models(args, info, process, device)
    test_set = [{"gt": torch.rand(31, 32, 32), "hr_msi": torch.rand(3, 32, 32)} for _ in range(2)]
    with torch.no_grad():
        result = entry.run_diagnostic(args, test_set, process, predict, oracle, srf, device, {})
    assert result["complete"]
    assert len(result["summary"]) == 3 * 4 * 2  # conditions x paths x base/refined
    root = Path(args.output_dir)
    assert len(json.loads((root / "test_inputs.json").read_text())) == 2
    random_rows = [json.loads(line) for line in (root / "random.jsonl").read_text().splitlines()]
    fixed_rows = [json.loads(line) for line in (root / "fixed_1.jsonl").read_text().splitlines()]
    for sample in range(2):
        generator = pipeline._make_generator(device, args.seed + 70000)
        gt = test_set[sample]["gt"].unsqueeze(0)
        for case in range(2):
            phi = sample_synthetic_geometry(32, 32, device=device, dtype=gt.dtype, generator=generator,
                max_translation=4, max_rotation_deg=2, max_local_px=4, control_grid=5, min_jacobian=0.5)
            expected = pipeline._process_from_geometry(process, phi).terminal_observation(gt)
            row = random_rows[sample * 2 + case]
            assert row["observed_sha256"] == tensor_sha256(expected)
            fixed, _ = sample_fixed_geometry(32, 32, STRENGTHS[0], sample_index=sample,
                                             case_index=case, seed=args.seed)
            expected_fixed = process.operator.degrade(warp_hsi(gt, fixed))
            assert fixed_rows[sample * 2 + case]["observed_sha256"] == tensor_sha256(expected_fixed)
    for row in result["summary"]:
        if row["condition"] == "registered":
            assert row["PSNR_drop_vs_own_registered_db"] == 0
            assert row["SAM_increase_vs_own_registered_deg"] == 0
    for name in ("summary.json", "summary.csv", "comparisons.csv", "geometry.csv"):
        assert (root / name).is_file()


def test_reject_identical_registered_and_adapted_checkpoints(small_setup):
    args, info, process, device = small_setup
    args.registered_checkpoint = args.diffusion_checkpoint
    with pytest.raises(ValueError, match="identical"):
        entry.build_models(args, info, process, device)


def test_base_only_can_resolve_registered_weights_from_init_refiner(small_setup):
    args, info, process, device = small_setup
    registered_path = args.registered_checkpoint
    args.registered_checkpoint = ""
    args.init_refiner_checkpoint = str(Path(args.output_dir).parent / "registered_gigi.pth")
    # Only its provenance is needed in base-only mode; never load GIGI weights.
    torch.save({"extra": {"baseline_checkpoint": registered_path}}, args.init_refiner_checkpoint)
    args.refiner_mode = "off"
    predict, oracle, _, records = entry.build_models(args, info, process, device)
    assert "shared_refiner" not in records
    with torch.no_grad():
        outputs, _, _ = predict(torch.rand(1, 31, 8, 8), torch.rand(1, 3, 32, 32))
    assert all(set(stages) == {"base"} for stages in outputs.values())


def test_gap_signs_and_dry_run_without_data(tmp_path):
    rows = [{"condition": "random", "stage": "base", "path": path, "PSNR": psnr, "SAM": sam}
            for path, psnr, sam in (("registered_backbone", 40, 4), ("adapted_naive", 41, 3.5),
                                     ("estimated", 44, 2), ("oracle", 45, 1.8))]
    gaps = {r["comparison"]: r for r in entry.comparison_rows(rows)}
    assert gaps["geometry_gain"]["PSNR_gain_db"] == 3
    assert gaps["geometry_gain"]["SAM_reduction_deg"] == 1.5
    assert gaps["oracle_gap"]["PSNR_gain_db"] == 1
    result = subprocess.run([sys.executable, str(ROOT / "diagnose_cdrdi_robustness.py"),
                             "--dataset", "CAVE", "--dry_run"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "DRY_RUN" in result.stdout


def test_missing_registered_is_not_silently_replaced(small_setup):
    args, info, process, device = small_setup
    args.registered_checkpoint = ""
    args.init_refiner_checkpoint = str(Path(args.output_dir) / "missing.pth")
    with pytest.raises(ValueError, match="registered_checkpoint"):
        entry.build_models(args, info, process, device)
