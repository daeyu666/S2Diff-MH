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
    parse_modes,
    parse_policies,
)
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver


class WaldDepthDiagnosticTests(unittest.TestCase):
    def test_depth_parser(self):
        self.assertEqual(parse_depths("9,3,0,1,6,3"), [0, 1, 3, 6, 9])
        with self.assertRaises(ValueError):
            parse_depths("-1,0")
        with self.assertRaises(ValueError):
            parse_depths("")

    def test_mode_parser(self):
        self.assertEqual(
            parse_modes("rigid_only,both,local_only,rigid_only,seed_only"),
            ["rigid_only", "both", "local_only", "seed_only"]
        )
        for bad in ("", "both,garbage", "none"):
            with self.assertRaises(ValueError):
                parse_modes(bad)

    def test_policy_parser(self):
        self.assertEqual(
            parse_policies("plain,closure_backtrack,plain"),
            ["plain", "closure_backtrack"]
        )
        with self.assertRaises(ValueError):
            parse_policies("nothing")

    def test_closure_guard_rejects_known_harmful_corrections(self):
        torch.manual_seed(32)
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        msi = torch.rand(1, 4, 48, 48)
        psf = EffectiveGaussianDegradation(scale_ratio=3, terminal_sigma=1.2)
        zero = torch.zeros(1)
        local = torch.zeros(1, 2, 48, 48)
        target = psf.degrade(
            forward_warp(msi, torch.tensor([-0.5]), zero, zero, local)
        )
        model = LearnedPhysicalResidualSolver(
            4, base_channels=8, control_grid=5,
            max_translation=1.0, max_rotation_deg=0.5, max_local_px=0.5,
            initial_dx_px=-0.5
        ).eval()
        # Make the network propose a deterministic WRONG rigid x step
        # moving the already correct -0.5 offset back towards zero.
        with torch.no_grad():
            model.update_net.rigid_head[-1].weight.zero_()
            model.update_net.rigid_head[-1].bias.zero_()
            model.update_net.rigid_head[-1].bias[0] = 0.3
            mask = torch.ones(1, 1, 16, 16)
            plain = model(
                target, msi, psf, steps=3, update_mode="rigid_only"
            )
            guarded = model(
                target, msi, psf, steps=3, update_mode="rigid_only",
                update_policy="closure_backtrack", acceptance_mask=mask
            )
        for pred in guarded["predictions"]:
            torch.testing.assert_close(
                pred, guarded["initial_prediction"], atol=1e-7, rtol=0
            )
        for step_accepted, step_alpha in zip(
            guarded["accepted_steps"], guarded["accepted_alphas"]
        ):
            self.assertFalse(bool(step_accepted.any()))
            self.assertEqual(float(step_alpha.max()), 0.0)
        self.assertGreater(float(plain["final_rigid"][0, 0]), -0.49)
        self.assertAlmostEqual(float(guarded["final_rigid"][0, 0]), -0.5, places=6)

    def test_closure_guard_accepts_beneficial_corrections(self):
        torch.manual_seed(42)
        msi = torch.rand(1, 4, 48, 48)
        psf = EffectiveGaussianDegradation(scale_ratio=3, terminal_sigma=1.2)
        zero = torch.zeros(1)
        local = torch.zeros(1, 2, 48, 48)
        target = psf.degrade(
            forward_warp(msi, torch.tensor([-0.75]), zero, zero, local)
        )
        model = LearnedPhysicalResidualSolver(
            4, base_channels=8, control_grid=5,
            max_translation=1.0, max_rotation_deg=0.5, max_local_px=0.5,
            initial_dx_px=-0.5
        ).eval()
        with torch.no_grad():
            model.update_net.rigid_head[-1].weight.zero_()
            model.update_net.rigid_head[-1].bias.zero_()
            model.update_net.rigid_head[-1].bias[0] = -0.12
            mask = torch.ones(1, 1, 16, 16)
            guarded = model(
                target, msi, psf, steps=3, update_mode="rigid_only",
                update_policy="closure_backtrack", acceptance_mask=mask
            )
            from train_augsburg_real_cdrdi import _local_standardize, _charbonnier
            scores = [
                float(_charbonnier(
                    _local_standardize(target, 5) -
                    _local_standardize(p, 5), mask
                ).item())
                for p in [guarded["initial_prediction"], *guarded["predictions"]]
            ]
        self.assertTrue(bool(guarded["accepted_steps"][0].all()))
        self.assertLess(scores[1], scores[0])
        for a, b in zip(scores, scores[1:]):
            self.assertLessEqual(b, a + 1e-7)

    def test_ablation_masks_updates_during_each_iteration(self):
        torch.manual_seed(21)
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        msi = torch.rand(1, 4, 24, 24)
        operator = EffectiveGaussianDegradation(scale_ratio=3, terminal_sigma=1.2)
        target = operator.degrade(msi)
        solver = LearnedPhysicalResidualSolver(
            4, base_channels=8, control_grid=5, max_translation=1.0,
            max_rotation_deg=0.5, max_local_px=0.5,
            initial_dx_px=-0.5
        ).eval()
        with torch.no_grad():
            full = solver(target, msi, operator, steps=3)
            both = solver(target, msi, operator, steps=3, update_mode="both")
            rigid = solver(target, msi, operator, steps=3, update_mode="rigid_only")
            local = solver(target, msi, operator, steps=3, update_mode="local_only")
            seed = solver(target, msi, operator, steps=3, update_mode="seed_only")
            for a, b in zip(full["predictions"], both["predictions"]):
                torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
            torch.testing.assert_close(
                full["final_rigid"], both["final_rigid"], rtol=0, atol=1e-7
            )
            for control in rigid["local_fields"]:
                self.assertLess(float(control.abs().max()), 1e-8)
            for state in local["rigid_states"]:
                self.assertAlmostEqual(float(state[0, 0]), -0.5, places=6)
                self.assertLess(float(state[0, 1:].abs().max()), 1e-8)
            for pred, state, control in zip(
                seed["predictions"], seed["rigid_states"], seed["local_fields"]
            ):
                torch.testing.assert_close(
                    pred, seed["initial_prediction"], rtol=0, atol=1e-7
                )
                self.assertAlmostEqual(float(state[0, 0]), -0.5, places=6)
                self.assertLess(float(control.abs().max()), 1e-8)
            with self.assertRaisesRegex(ValueError, "update_mode"):
                solver(target, msi, operator, steps=1, update_mode="unknown")

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

        for mode in ("rigid_only", "local_only", "seed_only"):
            alternative, _ = collect_depth_metrics(
                solver, [sample], depths=depths, update_mode=mode, **kwargs
            )
            self.assertAlmostEqual(
                alternative["identity"]["norm"], result["identity"]["norm"], places=7
            )
            self.assertAlmostEqual(
                alternative[0]["norm"], result[0]["norm"], places=7
            )
            if mode == "rigid_only":
                self.assertLess(alternative[9]["local_mean"], 1e-8)
            elif mode == "local_only":
                self.assertAlmostEqual(alternative[9]["dx"], -0.5, places=6)
            else:
                for depth in depths:
                    self.assertAlmostEqual(
                        alternative[depth]["norm"], alternative[0]["norm"], places=7
                    )
                    self.assertAlmostEqual(
                        alternative[depth]["reduction_vs_seed_pct"], 0, places=6
                    )

        guarded, _ = collect_depth_metrics(
            solver, [sample], depths=depths,
            update_mode="both", update_policy="closure_backtrack", **kwargs
        )
        for d in depths:
            self.assertLessEqual(guarded[d]["norm"], guarded[0]["norm"] + 1e-7)
        for a, b in zip(depths, depths[1:]):
            self.assertLessEqual(guarded[b]["norm"], guarded[a]["norm"] + 1e-7)



if __name__ == "__main__":
    unittest.main()
