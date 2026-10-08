"""Checks for Augsburg-2 Wald radiometry fit helpers."""

import numpy as np

from calibrate_augsburg2_wald_radiometry import sample_shift
from diagnose_augsburg2_srf import _fit_affine


def test_wald_radiometry_shift_is_diagnostic_only_and_has_known_sign():
    img = np.broadcast_to(
        np.arange(12, dtype=np.float32)[None, :, None],
        (8, 12, 4)
    ).copy()
    original = img.copy()
    aligned = sample_shift(img, 0.0, -0.5)
    assert aligned.shape == img.shape
    assert np.array_equal(img, original)
    assert np.allclose(aligned[:, 3:-3, :], img[:, 3:-3, :] - .5)


def test_wald_radiometry_bandwise_gain_is_identifiable_without_enmap10():
    x = np.linspace(.01, .5, 100, dtype=np.float32)
    hsi30_projection = .81 * x + .015
    gain, bias = _fit_affine(x, hsi30_projection)
    assert np.isclose(gain, .81, atol=1e-6)
    assert np.isclose(bias, .015, atol=1e-6)
