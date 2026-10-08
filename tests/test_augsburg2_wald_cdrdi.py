"""Unit checks for strict Augsburg-2 Wald CDRDI metadata and geometry guards.

Run: python -m unittest discover -s tests -p 'test_augsburg2_wald_cdrdi.py'
No Augsburg data and no CUDA device required.
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
