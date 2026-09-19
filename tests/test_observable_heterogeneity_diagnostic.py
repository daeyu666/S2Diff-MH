import torch

from diagnose_observable_heterogeneity import (
    _rank01_map,
    observable_heterogeneity,
    observable_risk_maps,
)


def test_rank01_preserves_nan_and_range():
    x = torch.tensor([[[float("nan"), 3.0, 1.0], [2.0, 4.0, float("nan")]]])
    r = _rank01_map(x)
    assert torch.isnan(r[0, 0, 0])
    assert torch.isnan(r[0, 1, 2])
    finite = r[torch.isfinite(r)]
    assert float(finite.min()) == 0.0
    assert float(finite.max()) == 1.0


def test_observable_heterogeneity_detects_spectral_transition():
    h = w = 16
    c = 4
    left = torch.tensor([1.0, 0.2, 0.1, 0.1])
    right = torch.tensor([0.1, 0.1, 0.2, 1.0])
    x = torch.empty(1, c, h, w)
    x[:, :, :, : w // 2] = left.view(1, c, 1, 1)
    x[:, :, :, w // 2 :] = right.view(1, c, 1, 1)

    hetero = observable_heterogeneity(x)[0]
    transition = hetero[2:-2, 7:9].nanmean()
    interior = torch.cat(
        [
            hetero[2:-2, 2:5].reshape(-1),
            hetero[2:-2, 11:14].reshape(-1),
        ]
    )
    interior = interior[torch.isfinite(interior)].mean()
    assert float(transition) > float(interior) + 1e-3


def test_risk_high_when_msi_transition_missing_from_xt():
    h = w = 16
    msi = torch.empty(1, 4, h, w)
    left = torch.tensor([1.0, 0.1, 0.1, 0.1])
    right = torch.tensor([0.1, 0.1, 0.1, 1.0])
    msi[:, :, :, : w // 2] = left.view(1, 4, 1, 1)
    msi[:, :, :, w // 2 :] = right.view(1, 4, 1, 1)

    # Current HSI state misses the material transition and is spectrally uniform.
    xt = torch.ones(1, 12, h, w)

    maps = observable_risk_maps(msi, xt)
    risk = maps["RISK"].reshape(h, w)
    transition = risk[2:-2, 7:9].nanmean()
    interior = torch.cat(
        [
            risk[2:-2, 2:5].reshape(-1),
            risk[2:-2, 11:14].reshape(-1),
        ]
    )
    interior = interior[torch.isfinite(interior)].mean()

    assert float(transition) > float(interior) + 0.2
