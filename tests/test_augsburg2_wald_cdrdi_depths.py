"""Synthetic CPU tests for the Augsburg-2 Wald CDRDI depth diagnostic.

No real Augsburg arrays, EnMAP10 labels, CUDA, or checkpoints are required.
"""
import unittest

import torch

from cdrdi_geometry import forward_warp
from degradations.effective_gaussian import EffectiveGaussianDegradation
from diagnose_augsburg2_wald_cdrdi_depths import (
    collect_depth_metrics,
    parse_depths,
)
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver


class WaldDepthDiagnosticTests(unittest.TestCase):
    def test_depth_parser(self):
        self.assertEqual(parse_depths("9,3,0,1,6,3"), [0, 1, 3, 6, 9])
        with self.assertRaises(ValueError):
            parse_depths("-1,0")
        with self.assertRaises(ValueError):
            parse_depths("")

    def test_common_input_all_steps_and_seed_better_than_identity(self):
        torch.manual_seed(8)
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        device = torch.device("cpu")
        msi = torch.rand(1, 4, 48, 48)
        operator = EffectiveGaussianDegradation(
            scale_ratio=3, terminal_sigma=1.2
        )
        zero = torch.zeros(1)
        field = torch.zeros(1, 2, 48, 48)
        lr = operator.degrade(forward_warp(msi, torch.tensor([-0.5]), zero, zero, field))
        sample = {
            "lr_hsi": lr,
            "hr_msi": msi,
            "valid_mask": torch.ones(1, 1, 48, 48),
        }
        solver = LearnedPhysicalResidualSolver(
            4, base_channels=8, control_grid=5, max_translation=1.0,
            max_rotation_deg=0.5, max_local_px=0.5,
            initial_dx_px=-0.5, initial_dy_px=0.0,
        ).eval()
        kwargs = {
            "p0": operator, "srf": torch.eye(4),
            "radiometry": None, "device": device, "local_window": 5,
        }
        depths = [0, 1, 3, 6, 9]
        result, tiles = collect_depth_metrics(
            solver, [sample], depths=depths, **kwargs
        )
        self.assertEqual(tiles, [])
        self.assertEqual(set(result), {"identity", *depths})
        self.assertLess(result[0]["norm"], 1e-7)
        self.assertGreater(result["identity"]["norm"], 1e-4)
        self.assertAlmostEqual(result[0]["dx"], -0.5)
        self.assertAlmostEqual(result[0]["min_jac"], 1.0)
        self.assertEqual(result[9]["n_tiles"], 1)
        for depth in result:
            self.assertEqual(
                result[depth]["valid_lr_pixels"], 16 * 16
            )

        # Extracting intermediate depth 3 from a nine-step run must match
        # stopping the same deterministic solver at step 3.
        short, _ = collect_depth_metrics(
            solver, [sample], depths=[0, 1, 3], **kwargs
        )
        for depth in (0, 1, 3):
            self.assertAlmostEqual(result[depth]["norm"], short[depth]["norm"], places=6)
            self.assertAlmostEqual(result[depth]["dx"], short[depth]["dx"], places=6)


if __name__ == "__main__":
    unittest.main()
