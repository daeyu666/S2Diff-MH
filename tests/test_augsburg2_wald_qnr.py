"""CPU synthetic regression tests for QNR using original LR-HSI and HR-MSI.

Tests 242-band spectral pair coverage, MSI cross-scale spatial consistency,
absence of PAN/proxy/SRF projection, and strict Wald data provenance.
"""
import json
import os
import tempfile
import unittest

import numpy as np

from augsburg2_wald_qnr import hsi_msi_qnr, evaluate_cache, _windowed_uiqi_matrix


def _inputs(seed=18):
    rng = np.random.default_rng(seed)
    hsi = rng.uniform(0.07, 0.8, (16, 16, 242)).astype(np.float32)
    fused = np.repeat(np.repeat(hsi, 3, axis=0), 3, axis=1)
    srf = np.zeros((4, 242), dtype=np.float32)
    for j in range(4):
        srf[j, j * 6:j * 6 + 6] = 1. / 6.
    msi = fused @ srf.T
    mask = np.ones((48, 48), dtype=np.uint8)
    return fused, hsi, msi, mask, srf


class HsiMsiQNRTests(unittest.TestCase):
    def test_perfect_native_hsi_msi_consistency(self):
        fused, hsi, msi, mask, srf = _inputs()
        quality = hsi_msi_qnr(fused, hsi, msi, mask, srf)
        self.assertAlmostEqual(quality["QNR"], 1., places=5)
        self.assertAlmostEqual(quality["Dlambda"], 0., places=5)
        self.assertAlmostEqual(quality["Ds"], 0., places=5)
        self.assertEqual(quality["spectral_pair_count"], 29161)
        self.assertEqual(quality["spatial_pair_count"], 24)
        self.assertEqual(quality["spatial_support_counts"], [6, 6, 6, 6])
        self.assertEqual(quality["spectral_reference"], "all_242_bands_of_observed_30m_HSI")
        self.assertFalse(quality["pan_used"])
        self.assertFalse(quality["srf_projection_used"])
        self.assertFalse(quality["full_HR_HSI_ground_truth_used"])

    def test_dlambda_sensitive_to_hsi_band_outside_msi_srf(self):
        fused, hsi, msi, mask, srf = _inputs()
        altered = fused.copy()
        altered[:, :, 200] *= 0.25
        q = hsi_msi_qnr(altered, hsi, msi, mask, srf)
        self.assertGreater(q["Dlambda"], 0.)
        # Spatial difference excludes 200, which is outside S2 B2/B3/B4/B8.
        self.assertAlmostEqual(q["Ds"], 0., places=6)

    def test_ds_sensitive_to_msi_observations(self):
        fused, hsi, msi, mask, srf = _inputs()
        wrong = msi.copy()
        # A spatially varying distortion: high MSI changes but its low-res
        # 3x3 block-average reference is largely preserved.
        wrong[::3, ::3, 0] *= 0.45
        q = hsi_msi_qnr(fused, hsi, wrong, mask, srf)
        self.assertGreater(q["Ds"], 0.)
        self.assertAlmostEqual(q["Dlambda"], 0., places=6)

    def test_train_only_radiometry_is_used(self):
        fused, hsi, msi, mask, srf = _inputs()
        gain = np.array([0.9, 1.1, 0.8, 0.95], dtype=np.float32)
        bias = np.array([0.015, -0.02, 0.02, 0.005], dtype=np.float32)
        raw = (msi - bias) / gain
        corrected = hsi_msi_qnr(fused, hsi, raw, mask, srf, gains=gain, biases=bias)
        uncorrected = hsi_msi_qnr(fused, hsi, raw, mask, srf)
        self.assertLess(corrected["Ds"], uncorrected["Ds"])
        self.assertAlmostEqual(corrected["QNR"], 1.0, places=5)

    def test_mask_and_srf_fail_closed(self):
        fused, hsi, msi, mask, srf = _inputs()
        with self.assertRaisesRegex(ValueError, "No valid"):
            hsi_msi_qnr(fused, hsi, msi, np.zeros_like(mask), srf)
        bad_srf = srf.copy()
        bad_srf[0] *= 0.5
        with self.assertRaisesRegex(ValueError, "Invalid fixed"):
            hsi_msi_qnr(fused, hsi, msi, mask, bad_srf)
        with self.assertRaisesRegex(ValueError, "window_hr"):
            hsi_msi_qnr(fused, hsi, msi, mask, srf, window_hr=49)

    def test_matrix_uiqi_has_diagonal_one_for_identical_images(self):
        rng = np.random.default_rng(13)
        x = rng.uniform(0.1, 0.9, size=(48, 48, 4))
        q = _windowed_uiqi_matrix(x, x, np.ones((48, 48), dtype=bool), 48)
        np.testing.assert_allclose(q, q.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(q), np.ones(4), atol=1e-9)

    def test_cache_consistency_and_provenance(self):
        fused, hsi, msi, mask, srf = _inputs()
        with tempfile.TemporaryDirectory() as root:
            for split in ("train", "validation", "test"):
                sub = os.path.join(root, split)
                os.makedirs(sub)
                with open(os.path.join(sub, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "msi_source": "real_Sentinel_2_Wald_30m",
                        "target": "30m_EnMAP_like",
                        "gt_source": "observed_30m_HSI_only",
                        "scale_ratio": 3,
                    }, f)
            full = os.path.join(root, "full")
            os.makedirs(full)
            with open(os.path.join(full, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"region": "sub_area_2"}, f)
            with open(os.path.join(root, "wald_psf.json"), "w", encoding="utf-8") as f:
                json.dump({"scale_ratio": 3}, f)
            cal = os.path.join(root, "calibration.json")
            with open(cal, "w", encoding="utf-8") as f:
                json.dump({"dataset": "Augsburg-2-Wald",
                           "uses_EnMAP10_reference": False,
                           "gain": [1,1,1,1], "bias": [0,0,0,0]}, f)
            np.save(os.path.join(full, "lr_hsi.npy"), hsi)
            np.save(os.path.join(full, "hr_msi.npy"), msi)
            np.save(os.path.join(full, "valid_mask.npy"), mask)
            np.save(os.path.join(root, "srf_weights.npy"), srf)
            fp = os.path.join(root, "fused.npy")
            np.save(fp, fused)
            direct = hsi_msi_qnr(fused, hsi, msi, mask, srf)
            via_cache = evaluate_cache(root, fp, cal)
            for k in ("QNR","Dlambda","Ds"):
                self.assertAlmostEqual(direct[k], via_cache[k], places=7)


if __name__ == "__main__":
    unittest.main()
