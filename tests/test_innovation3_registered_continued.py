"""CPU/static tests for the registered-continuation control experiment."""
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from run_innovation3_registered_continued import command
from train_innovation3_gigi_cdrdi import _checkpoint_path, _monitor_value


def _launcher_args(root):
    return SimpleNamespace(
        geometry_root=root + "/g",
        diffusion_root=root + "/d",
        registered_gigi_root=root + "/r",
        checkpoint_root=root + "/out",
        log_root=root + "/logs",
        epochs=100,
        batch_size=2,
        seed=10,
        device="cuda",
    )


def test_launcher_controls_only_geometry_distribution():
    with TemporaryDirectory() as root:
        args = _launcher_args(root)
        cmd = command("PaviaU", args, stage="train")
        joined = " ".join(cmd)
        assert "--train_geometry_mode registered" in joined
        assert "--monitor registered_sam_high" in joined
        assert "--epochs 100" in joined
        assert "--lr 2e-4" in joined
        assert "--lambda_l1 1.0" in joined
        assert "--lambda_ang 0.1" in joined
        assert "--lambda_phy 0.1" in joined
        assert "--lambda_msi 0.1" in joined
        assert "PaviaU_recursive_k6_finalonly_300ep_lr1e4.pth" in joined
        assert "PaviaU_estimated_deform_diffusion_k9_stage2d_A.pth" in joined
        assert "PaviaU_innovation3_gigi_full.pth" in joined


def test_registered_control_checkpoint_name_cannot_overwrite_mixed():
    with TemporaryDirectory() as root:
        common = dict(
            checkpoint_root=root,
            save_name="",
            dataset="PaviaU",
            variant="full",
            geometry_steps=9,
        )
        registered = _checkpoint_path(
            SimpleNamespace(**common, train_geometry_mode="registered")
        )
        mixed = _checkpoint_path(
            SimpleNamespace(**common, train_geometry_mode="mixed")
        )
        assert registered != mixed
        assert "registered_continued" in registered
        assert "gigi_cdrdi" in mixed


def test_registered_monitor_reads_registered_metrics():
    metrics = {
        "registered_refined": {"SAM_HIGH": 1.2, "SAM": 1.0, "PSNR": 42.0},
        "warp_refined": {"SAM_HIGH": 4.0, "SAM": 3.5, "PSNR": 31.0},
    }
    assert _monitor_value(metrics, "registered_sam_high") == 1.2
    assert _monitor_value(metrics, "registered_sam") == 1.0
    assert _monitor_value(metrics, "registered_psnr") == 42.0
    assert _monitor_value(metrics, "warp_sam_high") == 4.0
