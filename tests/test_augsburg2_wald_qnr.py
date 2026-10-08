"""Regression tests for standard-form Augsburg-2 QNR with explicit PAN.

Synthetic HSI/MSI data only. No EnMAP10 reference, checkpoints or CUDA.
"""
import json
import os
import tempfile
import unittest

import numpy as np

from augsburg2_wald_qnr import standard_qnr, evaluate_cache, _masked_uiqi


def example(seed=10):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(0.1, 0.6, size=(32, 32, 242)).astype(np.float32)
    hi = lo.repeat(3, axis=0).repeat(3, axis=1)
    srf = np.zeros((4, 242), dtype=np.float32)
    for i in range(4):
        srf[i, i * 4:(i + 1) * 4] = 0.25
    msi = hi @ srf.T
    mask = np.ones((96, 96), dtype=bool)
    return hi, lo, msi, mask, srf


class WaldStandardFormQNRTests(unittest.TestCase):
    def test_exact_agreement_is_unit_qnr(self):
        fused, lr, msi, mask, srf = example()
        out = standard_qnr(fused, lr, msi, mask, srf)
        self.assertAlmostEqual(out["QNR"], 1., places=5)
        self.assertAlmostEqual(out["Dlambda"], 0., places=5)
        self.assertAlmostEqual(out["Ds"], 0., places=5)
        self.assertEqual(out["spectral_pair_count"], 6)
        self.assertEqual(out["spatial_pair_count"], 4)
        self.assertEqual(out["high_window"], 48)
        self.assertEqual(out["low_window"], 16)
        self.assertFalse(out["is_genuine_pan"])
        self.assertFalse(out["full_HR_HSI_ground_truth_used"])
        self.assertIn("synthetic", out["pan_origin"])

    def test_optional_supplied_pan_has_exact_same_standard_formula(self):
        fused, lr, msi, mask, srf = example()
        p = msi.mean(axis=-1)
        p_low = p.reshape(32, 3, 32, 3).mean(axis=(1, 3))
        proxy = standard_qnr(fused, lr, msi, mask, srf)
        supplied = standard_qnr(fused, lr, msi, mask, srf, pan_hr=p, pan_lr=p_low)
        self.assertAlmostEqual(proxy["QNR"], supplied["QNR"], places=6)
        self.assertAlmostEqual(proxy["Ds"], supplied["Ds"], places=6)
        self.assertTrue(supplied["is_genuine_pan"])
        self.assertIn("external", supplied["pan_origin"])
        with self.assertRaisesRegex(ValueError, "both pan_hr and pan_lr"):
            standard_qnr(fused, lr, msi, mask, srf, pan_hr=p)

    def test_only_four_corresponding_pan_terms(self):
        fused, lr, msi, mask, srf = example()
        altered = fused.copy()
        altered[:, :, :4] *= 0.4
        out = standard_qnr(altered, lr, msi, mask, srf)
        self.assertLess(out["QNR"], 1.)
        self.assertGreater(out["Dlambda"], 0.)
        self.assertEqual(out["spatial_pair_count"], 4)

    def test_train_only_radiometry_and_invalid_windows(self):
        fused, lr, msi, mask, srf = example()
        gain = np.array([0.8, 1.1, 0.95, 0.9], dtype=np.float32)
        bias = np.array([0.02, 0.03, -0.01, 0.015], dtype=np.float32)
        raw_msi = (msi - bias) / gain
        corrected = standard_qnr(fused, lr, raw_msi, mask, srf, gains=gain, biases=bias)
        uncorrected = standard_qnr(fused, lr, raw_msi, mask, srf)
        self.assertGreater(uncorrected["Ds"], corrected["Ds"])
        with self.assertRaisesRegex(ValueError, "No valid UIQI windows"):
            standard_qnr(fused, lr, msi, np.zeros_like(mask), srf)
        with self.assertRaisesRegex(ValueError, "window_hr"):
            standard_qnr(fused, lr, msi, mask, srf, window_hr=50)

    def test_cache_matches_direct_and_checks_provenance(self):
        fused, lr, msi, mask, srf = example()
        with tempfile.TemporaryDirectory() as root:
            for split in ("train", "validation", "test"):
                path = os.path.join(root, split)
                os.makedirs(path)
                with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as handle:
                    json.dump({
                        "msi_source": "real_Sentinel_2_Wald_30m",
                        "target": "30m_EnMAP_like",
                        "gt_source": "observed_30m_HSI_only",
                        "scale_ratio": 3,
                    }, handle)
            full = os.path.join(root, "full")
            os.makedirs(full)
            with open(os.path.join(full, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump({"region": "sub_area_2"}, handle)
            with open(os.path.join(root, "wald_psf.json"), "w", encoding="utf-8") as handle:
                json.dump({"scale_ratio": 3, "terminal_sigma_hr_pixels": 1.2}, handle)
            cal = os.path.join(root, "calibration.json")
            with open(cal, "w", encoding="utf-8") as handle:
                json.dump({"dataset": "Augsburg-2-Wald", "uses_EnMAP10_reference": False,
                           "gain": [1, 1, 1, 1], "bias": [0, 0, 0, 0]}, handle)
            np.save(os.path.join(root, "srf_weights.npy"), srf)
            np.save(os.path.join(full, "lr_hsi.npy"), lr)
            np.save(os.path.join(full, "hr_msi.npy"), msi)
            np.save(os.path.join(full, "valid_mask.npy"), mask)
            filename = os.path.join(root, "fused.npy")
            np.save(filename, fused)
            direct = standard_qnr(fused, lr, msi, mask, srf)
            cached = evaluate_cache(root, filename, cal)
            for k in ("QNR", "Dlambda", "Ds"):
                self.assertAlmostEqual(direct[k], cached[k], places=7)
            with open(cal, "w", encoding="utf-8") as handle:
                json.dump({"dataset": "Augsburg-2-Wald", "uses_EnMAP10_reference": True,
                           "gain": [1, 1, 1, 1], "bias": [0, 0, 0, 0]}, handle)
            with self.assertRaisesRegex(ValueError, "train-only"):
                evaluate_cache(root, filename, cal)

    def test_uiqi_ignores_pixels_outside_valid_mask(self):
        a = np.arange(48 * 48, dtype=np.float64).reshape(48, 48)
        bad = a.copy()
        valid = np.ones((48, 48), dtype=bool)
        valid[:3] = False
        bad[:3] += 1e5
        self.assertAlmostEqual(_masked_uiqi(a, bad, valid, 48), 1., places=7)


if __name__ == "__main__":
    unittest.main()
