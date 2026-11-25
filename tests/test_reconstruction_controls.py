"""Scientific invariants and CPU end-to-end checks for reconstruction controls."""
import hashlib
import inspect
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reconstruction_controls as controls
import train_reconstruction_controls as runner
from models import RawMSIDirectPredictor
from utils import set_seed


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def cfg(experiment="generic_cold", *extra):
    return runner.parse_args(["--experiment", experiment, "--device", "cpu", "--base_channels", "8",
                              "--spectral_hidden", "2", "--time_dim", "32", "--diffusion_steps", "4",
                              "--patch_size", "16", "--test_size", "16", "--stride", "16",
                              "--batch_size", "1", "--epochs", "1", "--eval_interval", "1", *extra])


def info():
    return dict(n_bands=7, n_msi_bands=3, srf_weights=np.full((3, 7), 1 / 7, np.float32),
                protocol="synthetic CPU smoke", train_samples=2, validation_samples=1, test_samples=2)


def batch():
    gt = .05 + .8 * torch.rand(2, 7, 16, 16)
    msi = torch.einsum("mc,bchw->bmhw", torch.as_tensor(info()["srf_weights"]), gt)
    return gt, msi


def test_exact_backbone_initialization_and_default_capacity():
    default = runner.parse_args(["--experiment", "single_step"])
    assert (default.base_channels, default.spectral_hidden, default.diffusion_steps,
            default.epochs, default.batch_size, default.lr) == (64, 8, 12, 200, 4, 1e-4)
    states = []
    for experiment in controls.EXPERIMENTS:
        set_seed(10)
        model = controls.build_model(cfg(experiment), info(), "cpu")
        assert type(model) is RawMSIDirectPredictor
        states.append(model.state_dict())
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        for key in state:
            torch.testing.assert_close(state[key], states[0][key], atol=0, rtol=0)


def test_generic_has_no_sensor_operator_and_both_cold_variants_share_actual_endpoint(monkeypatch):
    gt, _ = batch()
    generic, physical = (controls.build_process(cfg(mode)) for mode in ("generic_cold", "physical_cold"))
    observed = physical.terminal_observation(gt)
    torch.testing.assert_close(generic.terminal_observation(gt), observed, atol=0, rtol=0)
    terminal = physical.terminal_state(observed, target_size=(16, 16))
    torch.testing.assert_close(generic.terminal_state(observed, target_size=(16, 16)), terminal, atol=0, rtol=0)
    torch.testing.assert_close(generic.state_at(gt, 4), physical.state_at(gt, 4), atol=0, rtol=0)
    assert not torch.allclose(generic.state_at(gt, 2), physical.state_at(gt, 2))
    times = torch.full((2,), 4, dtype=torch.long)
    for process in (generic, physical):
        torch.testing.assert_close(process.training_state(gt, observed, times), terminal, atol=0, rtol=0)
    assert not torch.allclose(generic.trajectory.terminal_observation(gt), observed)
    delta = terminal - generic.state_at(gt, 4)
    times = torch.tensor([1, 3])
    actual = generic.training_state(gt, observed, times)
    torch.testing.assert_close(actual, torch.cat([generic.state_at(gt[:1], 1), generic.state_at(gt[1:], 3)]) + delta)
    def forbidden(*args, **kwargs):
        raise AssertionError("Generic intermediate trajectory used a sensor operator")
    monkeypatch.setattr(generic.observation_process.operator, "degrade_at", forbidden)
    monkeypatch.setattr(generic.observation_process.operator, "adjoint_at", forbidden)
    assert generic.trajectory.operator.mode == "gaussian_bicubic"
    assert not hasattr(generic.trajectory.operator, "adjoint_at")
    assert torch.isfinite(generic.state_at(gt, 2)).all()
    torch.testing.assert_close(generic.state_at(gt, 0), gt, atol=0, rtol=0)


def test_direct_calls_once_and_never_constructs_a_diffusion_state(monkeypatch):
    args = cfg("single_step")
    process = controls.build_process(args)
    gt, msi = batch()
    lr = process.terminal_observation(gt)
    terminal = process.terminal_state(lr, target_size=(16, 16))
    calls = []
    class Model(torch.nn.Module):
        requires_msi = True
        def forward(self, x, observed_msi, t):
            calls.append((x.clone(), observed_msi.clone(), t.clone()))
            return x + .01
    def forbidden(*args, **kwargs):
        raise AssertionError("Direct control executed diffusion")
    monkeypatch.setattr(process, "training_state", forbidden)
    monkeypatch.setattr(process, "sample_timesteps", forbidden)
    monkeypatch.setattr(process, "reverse_update", forbidden)
    monkeypatch.setattr(process.trajectory, "state_at", forbidden)
    pred, train_lr = controls.training_prediction(Model(), process, gt, msi, args, random.Random(1))
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], terminal, atol=0, rtol=0)
    torch.testing.assert_close(train_lr, lr, atol=0, rtol=0)
    assert (calls[0][2] == 4).all()
    torch.testing.assert_close(controls.predict(Model(), process, lr, msi, args), pred, atol=0, rtol=0)
    assert len(calls) == 2
    assert "gt" not in inspect.signature(controls.predict).parameters


@pytest.mark.parametrize("experiment", ["generic_cold", "physical_cold"])
@pytest.mark.parametrize("steps", [4, 12])
def test_oracle_nonzero_residual_endpoint_and_call_count(experiment, steps):
    args = cfg(experiment, "--diffusion_steps", str(steps))
    process = controls.build_process(args)
    gt, msi = batch()
    # Deliberate observation mismatch, also for the physical control.
    lr = process.terminal_observation(gt) + .02
    calls = []
    class Oracle(torch.nn.Module):
        requires_msi = True
        def forward(self, state, observed_msi, t):
            calls.append(int(t[0]))
            return gt
    corrected = controls.predict(Oracle(), process, lr, msi, args)
    torch.testing.assert_close(corrected, gt, atol=5e-7, rtol=0)
    assert calls == list(range(steps, 0, -1))
    args.diffusion_output = "state"
    raw = controls.predict(Oracle(), process, lr, msi, args)
    delta = process.terminal_state(lr, target_size=(16, 16)) - process.state_at(gt, steps)
    assert delta.abs().mean() > 1e-4
    torch.testing.assert_close(raw, gt + delta, atol=5e-7, rtol=0)


def test_timestep_stream_does_not_change_shared_augmentations(monkeypatch):
    class Augmented(torch.utils.data.Dataset):
        def __len__(self): return 3
        def __getitem__(self, i):
            gt = torch.full((7, 16, 16), .1 + random.random() * .8)
            return {"gt": gt, "hr_msi": gt[:3]}
    original = controls.training_prediction
    fingerprints = []
    def record(*args):
        fingerprints[-1].append(hashlib.sha256(args[2].numpy().tobytes()).hexdigest())
        return original(*args)
    monkeypatch.setattr(controls, "training_prediction", record)
    for experiment in controls.EXPERIMENTS:
        fingerprints.append([])
        args = cfg(experiment)
        set_seed(10)
        model = controls.build_model(args, info(), "cpu")
        loader = torch.utils.data.DataLoader(Augmented(), batch_size=1, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        set_seed(1000013)
        controls.train_one_epoch(model, loader, optimizer, controls.build_process(args), args, "cpu", epoch=1)
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_full_training_validation_test_summary_and_checkpoint_guards(tmp_path):
    gt, msi = batch()
    examples = [{"gt": x, "hr_msi": y} for x, y in zip(gt, msi)]
    loader = torch.utils.data.DataLoader(examples, batch_size=1)
    validation = torch.utils.data.DataLoader(examples[:1], batch_size=1)
    rows = []
    for experiment in controls.EXPERIMENTS:
        args = cfg(experiment, "--checkpoint_root", str(tmp_path / "weights"),
                   "--log_root", str(tmp_path / "logs"), "--output_root", str(tmp_path / "out"))
        set_seed(10)
        initial = controls.build_model(args, info(), "cpu").state_dict()
        paths = runner.train(args, loader, validation, info(), "cpu")
        saved = torch.load(paths["last"], weights_only=False)
        assert saved["epoch"] == 1 and saved["updates"] == 2
        assert saved["validation"]["test_patches"] == 1
        assert any(not torch.equal(value, initial[key]) for key, value in saved["model"].items())
        rows.append(runner.test(args, loader, info(), "cpu"))
        with pytest.raises(FileExistsError):
            runner.train(args, loader, validation, info(), "cpu")
        if experiment != "single_step":
            args.diffusion_output = "state"
            with pytest.raises(ValueError, match="protocol mismatch"):
                runner.load_state(paths["best"], runner.protocol_signature(args, info()))
    assert len({row["input_sha256"] for row in rows}) == 1
    assert len({row["params"] for row in rows}) == 1
    assert [row["inference_network_calls"] for row in rows] == [1, 4, 4]
    assert len((tmp_path / "out/reconstruction_controls/summary.csv").read_text(encoding="utf-8-sig").splitlines()) == 4


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path):
    gt, msi = batch()
    loader = torch.utils.data.DataLoader([{"gt": x, "hr_msi": y} for x, y in zip(gt, msi)], batch_size=1, shuffle=True)
    common = ["--log_root", str(tmp_path / "logs"), "--output_root", str(tmp_path / "outputs")]
    continuous = cfg("generic_cold", *common, "--epochs", "2", "--checkpoint_root", str(tmp_path / "continuous"), "--save_name", "continuous")
    full = runner.train(continuous, loader, loader, info(), "cpu")
    split = cfg("generic_cold", *common, "--checkpoint_root", str(tmp_path / "split"), "--save_name", "split")
    first = runner.train(split, loader, loader, info(), "cpu")
    split.resume, split.epochs = str(first["last"]), 2
    resumed = runner.train(split, loader, loader, info(), "cpu")
    a = torch.load(full["last"], weights_only=False)
    b = torch.load(resumed["last"], weights_only=False)
    assert a["updates"] == b["updates"] == 4
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
