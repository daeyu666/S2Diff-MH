"""CPU-only guard tests for strictly Wald-trained Augsburg-2 D2 A/B/C.

Does not read old EnMAP10 labels or require any real Augsburg arrays.
Run: python -m unittest discover -s tests -p 'test_augsburg2_wald_d2_abc.py'
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from augsburg_real_process import build_augsburg_real_process
from cdrdi_geometry import forward_warp
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from train_augsburg_real_diffusion import (
    _batch_geometry,
    _d2_checkpoint_metadata,
    _estimated_process,
    _fixed_wald_geometry,
    _geometry_required,
    _wald_d2_extra_guard,
    _wald_d2_provenance,
)


def _args(root, **changes):
    options = {
        "cache_root": root,
        "stage": "train",
        "geometry_mode": "identity",
        "init_checkpoint": "",
        "resume": "",
        "from_scratch": True,
        "geometry_checkpoint": "",
        "fixed_dx_px": -0.5,
        "fixed_dy_px": 0.0,
        "guard_window": 5,
        "guard_min_jac": 0.5,
        "guard_relative_gain": 1e-4,
        "geometry_steps": 9,
        "radiometry_json": "./data/calibration/Augsburg2_Wald_radiometry.json",
        "monitor": "ref_psnr",
    }
    options.update(changes)
    return SimpleNamespace(**options)


class WaldABCTests(unittest.TestCase):
    @staticmethod
    def _metadata(root):
        for split in ("train", "validation", "test"):
            path = os.path.join(root, split)
            os.makedirs(path)
            with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "msi_source": "real_Sentinel_2_Wald_30m",
                    "target": "30m_EnMAP_like",
                    "gt_source": "observed_30m_HSI_only",
                    "scale_ratio": 3,
                }, f)

    def test_wald_only_modes_and_scratch_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            self._metadata(root)
            _wald_d2_provenance(_args(root), sigma=1.2)
            _wald_d2_provenance(_args(root, geometry_mode="wald_fixed"), sigma=1.2)
            for bad in (
                _args(root, geometry_mode="estimated"),
                _args(root, init_checkpoint="synthetic.pth"),
                _args(root, geometry_mode="wald_fixed", fixed_dx_px=2.0),
                _args(root, from_scratch=False),
            ):
                with self.assertRaises(ValueError):
                    _wald_d2_provenance(bad, sigma=1.2)

    def test_fixed_shift_is_physical_warp_in_forward_degradation(self):
        torch.manual_seed(11)
        msi = torch.rand(1, 4, 48, 48)
        rigid, local = _fixed_wald_geometry(msi, dx=-0.5, dy=0.0)
        self.assertEqual(tuple(rigid.shape), (1, 3))
        self.assertEqual(tuple(local.shape), (1, 2, 48, 48))
        self.assertAlmostEqual(rigid[0, 0].item(), -0.5)
        self.assertEqual(float(local.abs().max()), 0)
        base = build_augsburg_real_process(effective_sigma=1.2, diffusion_steps=12)
        process = _estimated_process(base, rigid, local)
        full = process.terminal_observation(msi)
        expected = base.operator.degrade(
            forward_warp(msi, rigid[:, 0], rigid[:, 1], rigid[:, 2], local)
        )
        torch.testing.assert_close(full, expected, atol=1e-6, rtol=0)
        self.assertTrue(_geometry_required(_args(".", geometry_mode="wald_fixed")))
        self.assertFalse(_geometry_required(_args(".")))

    def test_frozen_guarded_cdrdi_returns_rigid_only(self):
        torch.manual_seed(9)
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        msi = torch.rand(1, 4, 48, 48)
        base = build_augsburg_real_process(effective_sigma=1.2, diffusion_steps=12)
        p0 = base.operator
        zero = torch.zeros(1)
        field = torch.zeros(1, 2, 48, 48)
        observed_lr = p0.degrade(
            forward_warp(msi, torch.tensor([-0.5]), zero, zero, field)
        )
        model = LearnedPhysicalResidualSolver(
            4, base_channels=8, control_grid=5,
            max_translation=1., max_rotation_deg=0.5,
            max_local_px=0.5, initial_dx_px=-0.5
        ).eval()
        mask = torch.ones(1, 1, 48, 48)
        args = _args(".", geometry_mode="wald_cdrdi", geometry_steps=3)
        with torch.no_grad():
            rigid, local = _batch_geometry(
                args, model, observed_lr, msi, mask, p0=p0,
                srf=torch.eye(4)
            )
        self.assertEqual(tuple(rigid.shape), (1, 3))
        self.assertEqual(float(local.abs().amax()), 0.0)
        self.assertGreaterEqual(float(rigid[0, 0]), -1.0)
        self.assertLessEqual(float(rigid[0, 0]), 1.0)

    def test_wald_checkpoint_cannot_be_reused_across_branches(self):
        args = _args(".", geometry_mode="wald_fixed")
        extra = _d2_checkpoint_metadata(
            args, msi_source="real_Sentinel_2_Wald_30m",
            sigma=1.2, reference_frame="Wald_EnMAP30", kind="best"
        )
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, "model.pth")
            torch.save({"extra": extra}, checkpoint)
            _wald_d2_extra_guard(
                checkpoint, args, msi_source="real_Sentinel_2_Wald_30m", sigma=1.2
            )
            with self.assertRaises(ValueError):
                _wald_d2_extra_guard(
                    checkpoint, _args(".", geometry_mode="identity"),
                    msi_source="real_Sentinel_2_Wald_30m", sigma=1.2
                )


if __name__ == "__main__":
    unittest.main()
