"""Unit checks for non-destructive Augsburg-2 SRF diagnostics."""

import numpy as np
import pandas as pd

from diagnose_augsburg2_srf import (
    _fit_affine, _geometry_lag_diagnostics, _metrics,
    _stable_masks, _warped_srf, _heldout_fractional_lag_test,
)
from srf_utils import estimate_band_widths


def test_srf_shift_width_only_modifies_smooth_nonnegative_row():
    wavelengths = np.linspace(400, 1000, 242, dtype=np.float32)
    wl = np.arange(400, 1001, dtype=np.float32)
    response = np.exp(-0.5 * ((wl - 550.) / 24.) ** 2)
    table = pd.DataFrame({"WL(nm)": wl, "S2B B3": response})
    widths = estimate_band_widths(wavelengths)
    a = _warped_srf(table, "S2B B3", wavelengths, widths, 550., 0., 1.)
    b = _warped_srf(table, "S2B B3", wavelengths, widths, 550., 4., 1.04)
    assert a.shape == (242,)
    assert np.min(a) >= 0
    assert np.min(b) >= 0
    assert np.isclose(a.sum(), 1., atol=1e-6)
    assert np.isclose(b.sum(), 1., atol=1e-6)
    assert not np.allclose(a, b)


def test_spatial_holdout_disjoint_and_sufficient():
    rng = np.random.default_rng(10)
    cube = rng.uniform(0.1, 0.5, size=(30, 40, 6)).astype(np.float32)
    msi = np.stack([cube[:, :, 0], cube[:, :, 2], cube[:, :, 3], cube[:, :, 5]], axis=-1)
    valid = np.ones((30, 40), dtype=bool)
    tr, te, boundary, _ = _stable_masks(cube, msi, valid, .7, .75)
    assert 2 <= boundary <= 38
    assert tr.sum() >= 100
    assert te.sum() >= 100
    assert not np.any(tr & te)
    assert not tr[:, boundary - 1:].any()
    assert not te[:, :boundary + 2].any()


def test_affine_fit_corrects_radiometry_without_editing_srf():
    x = np.linspace(.05, .6, 1000)
    y = 1.08 * x + .015
    gain, bias = _fit_affine(x, y)
    assert abs(gain - 1.08) < 1e-10
    assert abs(bias - .015) < 1e-10
    assert _metrics(x, y, gain, bias)["rmse_after_affine"] < 1e-10


def test_spatial_lag_diagnostic_flags_displacement():
    rng = np.random.default_rng(10)
    base = rng.normal(size=(30, 30, 4)).astype(np.float32)
    shift = np.roll(base, 1, axis=1)
    valid = np.ones((30, 30), dtype=bool)
    lags = _geometry_lag_diagnostics(base, shift, valid)
    assert len(lags) == 4
    assert all(row["lag_corr_gain"] > .8 for row in lags)
    assert all(abs(row["best_lag"]["dx"]) == 1 for row in lags)


def test_fractional_lag_is_selected_from_train_and_confirmed_on_holdout():
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(10)
    ref = gaussian_filter(
        rng.normal(size=(40, 52, 4)).astype(np.float32),
        sigma=(1.0, 1.0, 0.0),
    )
    msi = np.roll(ref, shift=1, axis=1)
    valid = np.ones((40, 52), dtype=bool)
    train = np.zeros((40, 52), dtype=bool)
    holdout = np.zeros((40, 52), dtype=bool)
    train[:, :34] = True
    holdout[:, 37:] = True

    out = _heldout_fractional_lag_test(ref, msi, train, holdout, valid, step=.5)
    assert out["status"] == "ok"
    assert out["best_global_shift_lr_pixels"]["dx"] == -1.0
    assert abs(out["best_global_shift_lr_pixels"]["dy"]) < 1e-6
    assert all(row["holdout_corr_gain"] > .2 for row in out["bands"])
    assert all(row["holdout_rmse_reduction"] > .1 for row in out["bands"])
