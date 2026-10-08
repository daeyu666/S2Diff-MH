"""Unit checks for strict Augsburg-2 Wald CDRDI metadata and geometry guards.

Run: python -m unittest discover -s tests -p 'test_augsburg2_wald_cdrdi.py'
No Augsburg data and no CUDA device required.
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from cdrdi_geometry import forward_warp
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from train_augsburg_real_cdrdi import (
    _assert_wald_checkpoint_settings,
    _selection_score,
    _wald_geometry_accepted,
    _wald_metadata,
)


class WaldCDRDITests(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            base_channels=32,
            control_grid=5,
            max_translation=1.0,
            max_rotation_deg=0.5,
            max_local_px=0.5,
            initial_dx_px=-0.5,
            initial_dy_px=0.0,
            selection_lambda_geometry=0.005,
            selection_min_jac=0.5,
            selection_max_motion_fraction=0.9,
            radiometry_json="./data/calibration/Augsburg2_Wald_radiometry.json",
        )

    @staticmethod
    def _metrics():
        return {
            "norm": 0.1, "min_jac": 0.9,
            "dx_abs": 0.5, "dy_abs": 0.1,
            "dx_residual_abs": 0.0, "dy_residual_abs": 0.1,
            "theta_abs": 0.1, "local_mean": 0.05,
        }

    def test_provenance_rejects_non_wald_split(self):
        with tempfile.TemporaryDirectory() as root:
            for split in ("train", "validation", "test"):
                os.makedirs(os.path.join(root, split))
                metadata = {
                    "msi_source": "real_Sentinel_2_Wald_30m",
                    "target": "30m_EnMAP_like",
                    "gt_source": "observed_30m_HSI_only",
                    "scale_ratio": 3,
                }
                with open(os.path.join(root, split, "meta.json"), "w", encoding="utf-8") as file:
                    json.dump(metadata, file)
            self.assertTrue(_wald_metadata(root))
            target = os.path.join(root, "validation", "meta.json")
            with open(target, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            metadata["gt_source"] = "EnMAP10"
            with open(target, "w", encoding="utf-8") as file:
                json.dump(metadata, file)
            with self.assertRaisesRegex(ValueError, "Invalid Wald-only provenance"):
                _wald_metadata(root)

    def test_geometric_checkpoint_selection_has_hard_limits(self):
        args = self._args()
        baseline = self._metrics()
        self.assertTrue(_wald_geometry_accepted(baseline, args))
        self.assertGreater(_selection_score(baseline, args), baseline["norm"])
        bad = dict(baseline, min_jac=0.3)
        self.assertFalse(_wald_geometry_accepted(bad, args))
        bad = dict(baseline, dx_abs=0.95)
        self.assertFalse(_wald_geometry_accepted(bad, args))

    def test_checkpoint_settings_must_match_physical_scale(self):
        args = self._args()
        extra = {
            "geometry_config": {
                "base_channels": 32, "control_grid": 5,
                "max_translation": 1.0,
                "max_rotation_deg": 0.5, "max_local_px": 0.5,
                "initial_dx_px": -0.5, "initial_dy_px": 0.0,
            },
            "effective_sigma": 1.2,
            "radiometry_json": args.radiometry_json,
        }
        _assert_wald_checkpoint_settings(extra, args, sigma=1.2)
        with self.assertRaisesRegex(ValueError, "max_translation"):
            _assert_wald_checkpoint_settings(extra, SimpleNamespace(
                **{**vars(args), "max_translation": 0.5}
            ), sigma=1.2)
        with self.assertRaisesRegex(ValueError, "PSF"):
            _assert_wald_checkpoint_settings(extra, args, sigma=1.3)
        with self.assertRaisesRegex(ValueError, "initial_dx_px"):
            _assert_wald_checkpoint_settings(extra, SimpleNamespace(
                **{**vars(args), "initial_dx_px": 0.0}
            ), sigma=1.2)

    def test_initial_minus_half_pixel_shift_matches_sampler_sign(self):
        torch.manual_seed(5)
        msi = torch.rand(1, 4, 18, 18)
        delta_x = torch.tensor([-0.5])
        zeros = torch.zeros(1)
        local = torch.zeros(1, 2, 18, 18)

        class SimplePool:
            @staticmethod
            def degrade(x):
                return F.avg_pool2d(x, 3, 3)

        target = SimplePool.degrade(forward_warp(msi, delta_x, zeros, zeros, local))
        solver = LearnedPhysicalResidualSolver(
            4, base_channels=8, max_translation=1.0,
            max_rotation_deg=0.5, max_local_px=0.5,
            initial_dx_px=-0.5,
        ).eval()
        with torch.no_grad():
            result = solver(target, msi, SimplePool(), steps=1)
        initial_error = (result["initial_prediction"] - target).abs().max().item()
        zero_error = (result["unaligned_prediction"] - target).abs().mean().item()
        self.assertLess(initial_error, 1e-6)
        self.assertGreater(zero_error, 1e-4)
        self.assertAlmostEqual(result["final_rigid"][0, 0].item(), -0.5, delta=0.1)

    def test_legacy_default_starts_at_zero(self):
        model = LearnedPhysicalResidualSolver(4, base_channels=8)
        self.assertEqual(model.initial_dx_px, 0.0)
        self.assertEqual(model.initial_dy_px, 0.0)


if __name__ == "__main__":
    unittest.main()
