"""Endpoint oracle controls, production equivalence, masks and paired exports."""
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
import train_innovation3_gigi_cdrdi as pipeline
from cdrdi_geometry import sample_synthetic_geometry, spectral_project
from innovation1 import reconstruct_from_terminal_lr


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class MSIModel(torch.nn.Module):
    requires_msi = True

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, msi, timestep):
        self.calls += 1
        return state * .8 + msi.mean(1, keepdim=True) * .2 + timestep[:, None, None, None] * .0001


def setup_process(tmp_path, extra=()):
    args = entry.parse_args(["--device", "cpu", "--diffusion_steps", "4", "--cases", "2",
                             "--output_dir", str(tmp_path / "out"), *extra])
    return args, pipeline.build_progressive_process(pipeline._config(args))


def test_oracle_controls_cover_zero_and_nonzero_offset():
    controls = entry.oracle_controls()
    assert all(r["passed"] for r in controls)
    assert controls[0]["delta_T_raw_RMSE"] < 2e-5
    assert controls[1]["delta_T_raw_RMSE"] > 1e-5
    assert controls[1]["state_minus_GT_raw_RMSE"] > 1e-5
    assert controls[1]["prediction_minus_GT_raw_RMSE"] == 0


def test_paired_state_matches_production_with_single_reverse_pass(tmp_path):
    args, base = setup_process(tmp_path)
    gt, msi = torch.rand(1, 4, 64, 64), torch.rand(1, 2, 64, 64)
    rigid = torch.tensor([[1.2, -.5, .7]])
    acquisition = pipeline._process_from_estimate(base, rigid, torch.zeros(1, 2, 64, 64))
    observed = acquisition.terminal_observation(gt)
    model = MSIModel()
    pair = entry.reconstruct_pair(model, acquisition, observed, target_size=(64, 64), hr_msi=msi)
    assert model.calls == args.diffusion_steps
    original = reconstruct_from_terminal_lr(model, acquisition, observed, target_size=(64, 64), hr_msi=msi)
    torch.testing.assert_close(pair["state_x0"], original, atol=0, rtol=0)
    assert pair["gap_identity_max_abs"] < 1e-6
    assert not torch.equal(pair["state_x0"], pair["last_prediction"])


def test_masks_exclude_all_psf_sampling_outside_reference_support(tmp_path):
    _, base = setup_process(tmp_path)
    truth = torch.tensor([[3., -2., 2.]])
    estimate = torch.tensor([[2.5, -1.5, 1.]])
    local = torch.zeros(1, 2, 128, 128)
    hr, lr, margin = entry.valid_masks(base, truth, local, estimate, local)
    assert hr.any() and lr.any() and margin > 0
    assert not hr[:, :, :margin].any()
    generator = torch.Generator().manual_seed(4)
    x = torch.rand(1, 3, 128, 128, generator=generator)
    outside_change = x + (~hr) * torch.rand(1, 3, 128, 128, generator=generator) * 10
    for rigid in (truth, estimate):
        process = pipeline._process_from_estimate(base, rigid, local)
        residual = process.terminal_observation(x) - process.terminal_observation(outside_change)
        assert residual.masked_select(lr.expand_as(residual)).abs().max() < 2e-6


def test_observation_residuals_are_raw_and_normalized(tmp_path):
    args, process = setup_process(tmp_path)
    gt = torch.ones(1, 4, 64, 64) * .5
    pred = gt + 1
    srf = torch.ones(2, 4) / 4
    msi = spectral_project(gt, srf)
    zero = torch.zeros(1, 3)
    local = torch.zeros(1, 2, 64, 64)
    hr, lr, _ = entry.valid_masks(process, zero, local, zero, local)
    metrics = entry.endpoint_metrics(pred, gt, process.terminal_observation(gt), msi,
                                     process, srf, hr, lr, args.scale_ratio)
    assert metrics["E_H"] == pytest.approx(1, abs=1e-5)
    assert metrics["E_M"] == pytest.approx(1)
    assert metrics["raw_RMSE"] == pytest.approx(1)
    assert metrics["RMSE"] == pytest.approx(.5)
    assert metrics["PSNR"] == pytest.approx(6.0205999)


def test_runner_pairs_observations_archives_and_original_refiner_input(tmp_path):
    args, process = setup_process(tmp_path)
    generator = torch.Generator().manual_seed(9)
    gt = torch.rand(4, 128, 128, generator=generator)
    srf = torch.ones(2, 4) / 4
    msi = spectral_project(gt[None], srf)[0]
    dataset = [{"gt": gt, "hr_msi": msi}]
    seen, refiner_inputs = [], []

    def estimate(observed, hr_msi):
        seen.append(entry.tensor_hash(observed))
        return torch.zeros(1, 3), torch.zeros(1, 2, 128, 128)

    def refine(state, hr_msi, observed, acquisition):
        refiner_inputs.append(state.clone())
        return state + .001

    model = MSIModel()
    with torch.no_grad():
        rows = entry.run_diagnostic(args, dataset, process, model, estimate, srf,
                                     torch.device("cpu"), refiner=object(), refine=refine)
    assert len(rows) == 6  # two conditions x three endpoints
    assert model.calls == (1 + args.cases) * args.diffusion_steps
    output = Path(args.output_dir)
    traces = json.loads((output / "traces.json").read_text())
    assert len(traces) == 3
    assert [r["observed_sha256"] for r in traces if r["condition"] == "warp"] == seen
    # Match exact legacy random sampler, unaffected by model RNG or oracle controls.
    geometry_rng = pipeline._make_generator(torch.device("cpu"), args.seed + 70000)
    for index, trace in enumerate(traces):
        with np.load(output / trace["archive"]) as data:
            assert {"state_x0", "last_prediction", "delta_T_diagnostic_only", "state_x0_refined"} <= set(data.files)
            np.testing.assert_array_equal(data["state_x0"], refiner_inputs[index].numpy())
            np.testing.assert_allclose(data["state_minus_prediction"], data["state_x0"] - data["last_prediction"], atol=0)
            if trace["condition"] == "warp":
                truth = sample_synthetic_geometry(128, 128, device=torch.device("cpu"), dtype=torch.float32,
                    generator=geometry_rng, max_translation=4, max_rotation_deg=2,
                    max_local_px=4, control_grid=5, min_jacobian=.5)
                expected = pipeline._process_from_geometry(process, truth).terminal_observation(gt[None])
                assert entry.tensor_hash(expected) == trace["observed_sha256"]
    assert (output / "summary.csv").is_file()
    # Verify paired rows retain identical masks and observation hashes.
    import csv
    with (output / "per_sample.csv").open() as handle:
        detailed = list(csv.DictReader(handle))
    for start in range(0, len(detailed), 3):
        group = detailed[start:start+3]
        assert len({r["observed_sha256"] for r in group}) == 1
        assert len({r["HR_valid_pixels"] for r in group}) == 1
        assert len({r["LR_valid_pixels"] for r in group}) == 1


def test_real_small_architecture_checkpoint_loading(tmp_path):
    args, process = setup_process(tmp_path, ["--geometry_base_channels", "8", "--diffusion_base_channels", "8",
        "--time_dim", "32", "--spectral_hidden", "2", "--geometry_steps", "1", "--conditions", "warp", "--cases", "1"])
    info = {"n_bands": 4, "n_msi_bands": 2, "srf_weights": np.full((2, 4), .25)}
    device = torch.device("cpu")
    geometry, diffusion = pipeline._build_frozen_models(args, info, device, load_weights=False)
    args.geometry_checkpoint = str(tmp_path / "geometry.pth")
    args.diffusion_checkpoint = str(tmp_path / "diffusion.pth")
    for name, model in (("geometry", geometry), ("diffusion", diffusion)):
        torch.save({"model": model.state_dict(), "extra": {"dataset": "PaviaU"}}, getattr(args, name + "_checkpoint"))
        assert len(entry.checkpoint_record(getattr(args, name + "_checkpoint"), "PaviaU")["sha256"]) == 64
    geometry, diffusion = pipeline._build_frozen_models(args, info, device)
    gt = torch.rand(1, 4, 128, 128)
    srf = torch.tensor(info["srf_weights"], dtype=torch.float32)
    msi = spectral_project(gt, srf)

    def estimate(observed, msi):
        return pipeline._estimate_geometry(geometry, y_h=observed, hr_msi=msi,
                   p0=process.operator, srf=srf, steps=args.geometry_steps)

    with torch.no_grad():
        result = entry.run_diagnostic(args, [{"gt": gt[0], "hr_msi": msi[0]}], process,
                                      diffusion, estimate, srf, device)
    assert len(result) == 2
    assert all(np.isfinite(row["PSNR"]) and np.isfinite(row["E_H"]) for row in result)


def test_help_and_dry_run_do_not_load_torch_or_data():
    code = "import sys; import diagnose_diffusion_endpoint as d; d.main(['--dry_run']); assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_complete_entrypoint_with_frozen_refiner_and_manifest(tmp_path, monkeypatch):
    args, _ = setup_process(tmp_path, ["--geometry_base_channels", "8", "--diffusion_base_channels", "8",
        "--time_dim", "32", "--spectral_hidden", "2", "--geometry_steps", "1",
        "--refine_hidden", "8", "--heads", "1", "--cases", "1"])
    info = {"n_bands": 31, "n_msi_bands": 2, "srf_weights": np.full((2, 31), 1/31), "protocol": "synthetic test fixture"}
    device = torch.device("cpu")
    geometry, diffusion = pipeline._build_frozen_models(args, info, device, load_weights=False)
    refiner = pipeline._build_refiner(args, info, device)
    for role, model in (("geometry", geometry), ("diffusion", diffusion), ("refiner", refiner)):
        path = tmp_path / (role + ".pth")
        torch.save({"model": model.state_dict(), "variant": "full", "epoch": 2,
                    "extra": {"dataset": "PaviaU"}}, path)
        setattr(args, role + "_checkpoint", str(path))
    gt = torch.rand(1, 31, 128, 128)
    srf = torch.as_tensor(info["srf_weights"], dtype=torch.float32)
    dataset = [{"gt": gt[0], "hr_msi": spectral_project(gt, srf)[0]}]

    class Loader:
        pass

    loader = Loader()
    loader.dataset = dataset
    monkeypatch.setattr(pipeline, "build_loaders", lambda cfg: (None, loader, info))
    monkeypatch.setattr(entry, "parse_args", lambda argv: args)
    entry.main([])
    manifest = json.loads((Path(args.output_dir) / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert len(manifest["summary"]) == 6
    assert set(manifest["checkpoints"]) == {"geometry", "diffusion", "refiner"}
    assert all(row["passed"] for row in manifest["oracle_controls"])
    # Existing artifacts are protected on a repeated command.
    with pytest.raises(FileExistsError):
        entry.main([])
