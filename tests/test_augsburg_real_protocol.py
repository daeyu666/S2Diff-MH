import json

import numpy as np
import torch

from augsburg_real import AugsburgRealDataset, _partition_tiles
from augsburg_real_process import build_augsburg_real_process
from degradations.deformation_aware import DeformationAwareProgressiveDegradation
from degradations.effective_gaussian import EffectiveGaussianDegradation


def test_real_progressive_schedule_is_1_2_3():
    process = build_augsburg_real_process(effective_sigma=1.2, diffusion_steps=12)
    scales = [process.state(t).scale for t in range(1, 13)]
    assert scales == [1] * 4 + [2] * 4 + [3] * 4
    assert process.stages == [1, 2, 3]


def test_effective_operator_x3_shape_and_constant_lift():
    op = EffectiveGaussianDegradation(scale_ratio=3, terminal_sigma=1.2)
    x = torch.ones(1, 5, 96, 96)
    y = op.degrade(x)
    assert y.shape == (1, 5, 32, 32)
    lifted = op.lift(
        y,
        scale=3,
        strength=1.0,
        lift_mode="normalized_adjoint",
        target_size=(96, 96),
    )
    assert lifted.shape == x.shape
    assert torch.allclose(lifted, x, atol=1e-5, rtol=1e-5)


def test_dataset_returns_exact_x3_pairs(tmp_path):
    for split in ("train", "validation", "test"):
        d = tmp_path / split
        d.mkdir(parents=True)
        gt = np.random.default_rng(1).random((192, 192, 8), dtype=np.float32)
        lr = np.random.default_rng(2).random((64, 64, 8), dtype=np.float32)
        msi = np.random.default_rng(3).random((192, 192, 4), dtype=np.float32)
        mask = np.ones((192, 192), dtype=np.uint8)
        np.save(d / "gt.npy", gt)
        np.save(d / "lr_hsi.npy", lr)
        np.save(d / "hr_msi.npy", msi)
        np.save(d / "valid_mask.npy", mask)
        with open(d / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"split": split}, f)

    ds = AugsburgRealDataset(
        str(tmp_path),
        "train",
        train_patch_size=96,
        train_stride=48,
        eval_patch_size=192,
        augment=False,
    )
    sample = ds[0]
    assert sample["gt"].shape == (8, 96, 96)
    assert sample["lr_hsi"].shape == (8, 32, 32)
    assert sample["hr_msi"].shape == (4, 96, 96)
    assert sample["valid_mask"].shape == (1, 96, 96)


def test_dataset_rejects_non_x3_patch(tmp_path):
    d = tmp_path / "train"
    d.mkdir(parents=True)
    np.save(d / "gt.npy", np.zeros((96, 96, 2), np.float32))
    np.save(d / "lr_hsi.npy", np.zeros((32, 32, 2), np.float32))
    np.save(d / "hr_msi.npy", np.zeros((96, 96, 4), np.float32))
    np.save(d / "valid_mask.npy", np.ones((96, 96), np.uint8))
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump({}, f)
    try:
        AugsburgRealDataset(str(tmp_path), "train", train_patch_size=64, train_stride=48)
    except ValueError as exc:
        assert "divisible by 3" in str(exc)
    else:
        raise AssertionError("expected x3 patch validation failure")


def test_effective_operator_works_with_deformation_aware_x3_process():
    base = build_augsburg_real_process(effective_sigma=1.2, diffusion_steps=12)
    rigid = torch.zeros(1, 3)
    local = torch.zeros(1, 2, 96, 96)
    process = DeformationAwareProgressiveDegradation(
        base,
        rigid=rigid,
        local_field=local,
    )
    x = torch.rand(1, 7, 96, 96)
    y = process.terminal_observation(x)
    assert y.shape == (1, 7, 32, 32)
    lifted = process.terminal_state(y, target_size=(96, 96))
    assert lifted.shape == x.shape
    process.assert_terminal_closure(x)


def test_eval_tiles_cover_full_region_once():
    tiles = _partition_tiles(300, 360, 192)
    assert tiles == [
        (0, 0, 192, 192),
        (0, 192, 192, 168),
        (192, 0, 108, 192),
        (192, 192, 108, 168),
    ]
    canvas = np.zeros((300, 360), dtype=np.int32)
    for top, left, ph, pw in tiles:
        assert ph % 3 == 0 and pw % 3 == 0
        canvas[top:top+ph, left:left+pw] += 1
    assert np.all(canvas == 1)


def test_validation_edge_tile_is_padded_to_multiple_of_six(tmp_path):
    d = tmp_path / "validation"
    d.mkdir(parents=True)
    h, w = 192, 255
    np.save(d / "gt.npy", np.ones((h, w, 8), np.float32))
    np.save(d / "lr_hsi.npy", np.ones((h // 3, w // 3, 8), np.float32))
    np.save(d / "hr_msi.npy", np.ones((h, w, 4), np.float32))
    np.save(d / "valid_mask.npy", np.ones((h, w), np.uint8))
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump({}, f)

    ds = AugsburgRealDataset(
        str(tmp_path),
        "validation",
        train_patch_size=96,
        train_stride=48,
        eval_patch_size=192,
        augment=False,
    )
    assert ds.samples[-1] == (0, 192, 192, 63)
    sample = ds[-1]
    assert sample["gt"].shape == (8, 192, 66)
    assert sample["hr_msi"].shape == (4, 192, 66)
    assert sample["lr_hsi"].shape == (8, 64, 22)
    assert sample["valid_mask"].shape == (1, 192, 66)
    assert float(sample["valid_mask"][:, :, :63].min()) == 1.0
    assert float(sample["valid_mask"][:, :, 63:].max()) == 0.0


def test_identity_real_d2_eval_avoids_geometry_model():
    """Smoke-test identity path with no Real-C checkpoint or deformation field."""
    import math
    from types import SimpleNamespace

    from train_augsburg_real_diffusion import evaluate

    class ZeroPredictor(torch.nn.Module):
        requires_msi = False

        def forward(self, x, timesteps):
            return x

    base = build_augsburg_real_process(effective_sigma=0.8, diffusion_steps=6)
    gt = torch.rand(1, 8, 12, 12) * 0.5 + 0.1
    lr = base.terminal_observation(gt)
    msi = torch.rand(1, 4, 12, 12) * 0.5 + 0.1
    srf = torch.rand(4, 8)
    srf = srf / srf.sum(dim=1, keepdim=True)
    batch = {
        "gt": gt,
        "lr_hsi": lr,
        "hr_msi": msi,
        "valid_mask": torch.ones(1, 1, 12, 12),
    }
    metrics = evaluate(
        ZeroPredictor(),
        None,
        [batch],
        base_process=base,
        p0=base.operator,
        srf=srf,
        radiometry=None,
        args=SimpleNamespace(geometry_mode="identity"),
        device=torch.device("cpu"),
    )
    for name in ("ref_psnr", "ref_sam", "phy", "msi"):
        assert math.isfinite(metrics[name]), name


def test_sim_control_selects_canonical_s2_band_indices():
    from prepare_augsburg_sim_control import _select_sim_indexes
    assert _select_sim_indexes(4) == [1, 2, 3, 4]
    assert _select_sim_indexes(12) == [2, 3, 4, 8]
    try:
        _select_sim_indexes(5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid simulated-MSI band count to fail")


def test_sim_control_product_paths_match_official_split_files():
    from prepare_augsburg_sim_control import _simulated_path
    assert _simulated_path(
        "/data/sr_deep_model_data/EeteS_EnMAP_10m_deep_valid.tif"
    ) == "/data/sr_deep_model_data/EeteS_Sentinel_2_10m_deep_valid.tif"
    assert _simulated_path(
        "/data/sub_area_1/EeteS_EnMAP_10m_sub_area1.tif"
    ) == "/data/sub_area_1/EeteS_Sentinel_2_10m_sub_area1.tif"


def test_sim_control_hardlinks_shared_hsi_arrays(tmp_path):
    from prepare_augsburg_sim_control import _shared
    src = tmp_path / "real" / "gt.npy"
    src.parent.mkdir()
    np.save(src, np.ones((6, 6, 2), np.float32))
    dst = tmp_path / "simulated" / "gt.npy"
    _shared(str(src), str(dst), mode="hardlink", overwrite=False)
    assert np.array_equal(np.load(src), np.load(dst))
    assert src.stat().st_ino == dst.stat().st_ino
    assert src.stat().st_nlink >= 2


def test_augsburg2_wald_resolutions_and_no_enmap10_required():
    from prepare_augsburg2_wald import downsample_hsi_wald, mean_downsample_msi

    hsi30 = np.ones((48, 72, 7), dtype=np.float32)
    s210 = np.ones((144, 216, 4), dtype=np.float32)
    lr90 = downsample_hsi_wald(hsi30, sigma=1.2)
    msi30 = mean_downsample_msi(s210)
    assert lr90.shape == (16, 24, 7)
    assert msi30.shape == (48, 72, 4)
    assert float(msi30.min()) == 1.
    assert np.isfinite(lr90).all()


def test_augsburg2_full_tiles_cover_region2_and_align_to_x3_grid():
    from infer_augsburg2_wald import positions

    y_starts = positions(300, 96, 48)
    x_starts = positions(360, 96, 48)
    assert y_starts[-1] == 204
    assert x_starts[-1] == 264
    assert all(v % 3 == 0 for v in y_starts + x_starts)
    canvas = np.zeros((300, 360), dtype=np.int32)
    for y in y_starts:
        for x in x_starts:
            canvas[y:y + 96, x:x + 96] += 1
    assert canvas.min() >= 1
