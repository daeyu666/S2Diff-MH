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
