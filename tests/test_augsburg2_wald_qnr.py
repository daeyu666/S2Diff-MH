"""Regression checks for UAFL-compatible Augsburg-2 MSI-projected QNR.

No network, checkpoints, real Augsburg files or 10m HSI ground truth needed.
Run: python -m unittest discover -s tests -p test_augsburg2_wald_qnr.py
"""
import json
import os
import tempfile
import unittest
import numpy as np
from augsburg2_wald_qnr import projected_qnr, _masked_uiqi, evaluate_cache
from compare_augsburg2_wald_qnr import compare_qnr


def _inputs(seed=13):
    rng = np.random.default_rng(seed)
    lr_hsi = rng.uniform(0.04, 0.65, (32, 32, 242)).astype(np.float32)
    srf = np.zeros((4, 242), dtype=np.float64)
    # A deterministic four-band SRF, sufficient for independent metric tests.
    for index in range(4):
        srf[index, index * 4:(index + 1) * 4] = 0.25
    fused = np.repeat(np.repeat(lr_hsi, 3, axis=0), 3, axis=1)
    msi = fused @ srf.T.astype(np.float32)
    mask = np.ones((96, 96), dtype=bool)
    return fused, lr_hsi, msi, mask, srf


class WaldProjectedQNRTests(unittest.TestCase):
    def test_perfect_consistency_qnr_is_one(self):
        fused, lr, msi, mask, srf = _inputs()
        metrics = projected_qnr(fused, lr, msi, mask, srf)
        self.assertAlmostEqual(metrics["QNR"], 1.0, places=5)
        self.assertLess(metrics["Dlambda"], 1e-5)
        self.assertLess(metrics["Ds"], 1e-5)
        self.assertEqual(metrics["spectral_pair_count"], 6)
        self.assertEqual(metrics["spatial_pair_count"], 16)
        self.assertEqual(metrics["high_window"], 48)
        self.assertEqual(metrics["low_window"], 16)
        self.assertFalse(metrics["full_HR_HSI_ground_truth_used"])

    def test_prespecified_radiometry_applied_to_real_msi(self):
        fused, lr, msi, mask, srf = _inputs()
        gains = np.array([0.8, 0.9, 1.1, 0.85])
        biases = np.array([0.02, -0.01, 0.015, 0.03])
        raw_msi = (msi - biases.reshape(1, 1, 4)) / gains.reshape(1, 1, 4)
        metrics = projected_qnr(
            fused, lr, raw_msi, mask, srf, gains=gains, biases=biases
        )
        self.assertAlmostEqual(metrics["QNR"], 1.0, places=5)
        self.assertLess(metrics["Ds"], 1e-5)
        uncorrected = projected_qnr(fused, lr, raw_msi, mask, srf)
        self.assertGreater(uncorrected["Ds"], metrics["Ds"])

    def test_spectral_distortion_affects_score_without_gt(self):
        fused, lr, msi, mask, srf = _inputs()
        altered = fused.copy()
        altered[:, :, 0:4] *= 0.4
        metrics = projected_qnr(altered, lr, msi, mask, srf)
        self.assertGreater(metrics["Dlambda"], 0.0)
        self.assertLess(metrics["QNR"], 1.0)
        self.assertEqual(metrics["source_HSI"], "observed 30m HSI")

    def test_validity_mask_and_window_fail_closed(self):
        fused, lr, msi, mask, srf = _inputs()
        with self.assertRaisesRegex(ValueError, "No valid"):
            projected_qnr(fused, lr, msi, np.zeros_like(mask), srf)
        wrong_srf = srf.copy()
        wrong_srf[0] *= 0.9
        with self.assertRaisesRegex(ValueError, "Invalid fixed"):
            projected_qnr(fused, lr, msi, mask, wrong_srf)
        with self.assertRaisesRegex(ValueError, "window_hr"):
            projected_qnr(fused, lr, msi, mask, srf, window_hr=50)

    def test_identical_metric_reports_compare_to_zero(self):
        fused, lr, msi, mask, srf = _inputs()
        m = projected_qnr(fused, lr, msi, mask, srf)
        self.assertEqual(compare_qnr(m, dict(m)), {
            "QNR": 0.0, "Dlambda": 0.0, "Ds": 0.0
        })
        bad = dict(m)
        bad["high_window"] = 96
        with self.assertRaisesRegex(ValueError, "high_window"):
            compare_qnr(m, bad)

    def test_cache_matches_direct_eval_and_checks_provenance(self):
        fused, lr, msi, mask, srf = _inputs()
        with tempfile.TemporaryDirectory() as directory:
            for name in ("train", "validation", "test"):
                root = os.path.join(directory, name)
                os.makedirs(root)
                with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "msi_source": "real_Sentinel_2_Wald_30m",
                        "target": "30m_EnMAP_like",
                        "gt_source": "observed_30m_HSI_only",
                        "scale_ratio": 3,
                    }, f)
            full = os.path.join(directory, "full")
            os.makedirs(full)
            with open(os.path.join(full, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"region": "sub_area_2"}, f)
            with open(os.path.join(directory, "wald_psf.json"), "w", encoding="utf-8") as f:
                json.dump({"scale_ratio": 3, "terminal_sigma_hr_pixels": 1.2}, f)
            calibration = os.path.join(directory, "calibration.json")
            with open(calibration, "w", encoding="utf-8") as f:
                json.dump({"dataset": "Augsburg-2-Wald",
                           "uses_EnMAP10_reference": False,
                           "gain": [1, 1, 1, 1],
                           "bias": [0, 0, 0, 0]}, f)
            np.save(os.path.join(directory, "srf_weights.npy"), srf)
            np.save(os.path.join(full, "lr_hsi.npy"), lr)
            np.save(os.path.join(full, "hr_msi.npy"), msi)
            np.save(os.path.join(full, "valid_mask.npy"), mask)
            out = os.path.join(directory, "fused.npy")
            np.save(out, fused)
            direct = projected_qnr(fused, lr, msi, mask, srf)
            cached = evaluate_cache(directory, out, calibration)
            for k in ("QNR", "Dlambda", "Ds"):
                self.assertAlmostEqual(direct[k], cached[k], places=7)
            with open(calibration, "w", encoding="utf-8") as f:
                json.dump({"dataset": "Augsburg-2-Wald",
                           "uses_EnMAP10_reference": True,
                           "gain": [1, 1, 1, 1],
                           "bias": [0, 0, 0, 0]}, f)
            with self.assertRaisesRegex(ValueError, "train-only"):
                evaluate_cache(directory, out, calibration)

    def test_uiqi_ignores_invalid_samples(self):
        rng = np.random.default_rng(4)
        a = rng.random((48, 48))
        mask = np.ones_like(a, dtype=bool)
        mask[:4, :] = False
        altered = a.copy()
        altered[:4, :] = 100.0
        self.assertAlmostEqual(
            _masked_uiqi(a, altered, mask, window=48),
            1.0, places=8
        )


if __name__ == "__main__":
    unittest.main()
