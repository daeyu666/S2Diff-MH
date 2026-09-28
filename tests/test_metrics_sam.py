import math

import torch

from metrics import calc_sam


def test_calc_sam_identical_low_energy_spectra_are_near_zero():
    spectrum = torch.tensor([1e-4, 2e-4, 3e-4, 4e-4], dtype=torch.float32)
    target = spectrum.view(1, 4, 1, 1)
    pred = target.clone()
    assert calc_sam(pred, target) < 0.05


def test_calc_sam_orthogonal_spectra_are_ninety_degrees():
    pred = torch.tensor([[[[1.0]], [[0.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.0]], [[1.0]]]], dtype=torch.float32)
    assert math.isclose(calc_sam(pred, target), 90.0, abs_tol=1e-5)


def test_calc_sam_all_zero_pair_is_defined_as_zero():
    pred = torch.zeros((1, 8, 2, 2), dtype=torch.float32)
    target = torch.zeros_like(pred)
    assert calc_sam(pred, target) == 0.0
