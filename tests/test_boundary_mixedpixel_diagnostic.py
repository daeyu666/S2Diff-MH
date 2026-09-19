import torch

from diagnose_boundary_mixedpixel import (
    _proxy_summary,
    spatial_material_proxies,
)


def _two_material_cube(h=16, w=16, c=12):
    left = torch.linspace(0.2, 0.8, c)
    right = torch.linspace(0.8, 0.2, c)
    cube = torch.empty(1, c, h, w)
    cube[:, :, :, : w // 2] = left.view(1, c, 1, 1)
    cube[:, :, :, w // 2 :] = right.view(1, c, 1, 1)
    return cube


def test_boundary_proxy_peaks_at_material_transition():
    gt = _two_material_cube()
    p = spatial_material_proxies(gt)
    h = w = 16
    boundary = p["boundary"].reshape(h, w)
    hetero = p["heterogeneity"].reshape(h, w)

    transition = boundary[2:-2, 7:9].nanmean()
    interior = torch.cat(
        [
            boundary[2:-2, 2:5].reshape(-1),
            boundary[2:-2, 11:14].reshape(-1),
        ]
    )
    interior = interior[torch.isfinite(interior)].mean()
    assert float(transition) > float(interior) + 1e-3

    transition_h = hetero[2:-2, 7:9].nanmean()
    interior_h = torch.cat(
        [
            hetero[2:-2, 2:5].reshape(-1),
            hetero[2:-2, 11:14].reshape(-1),
        ]
    )
    interior_h = interior_h[torch.isfinite(interior_h)].mean()
    assert float(transition_h) > float(interior_h) + 1e-3


def test_proxy_summary_detects_hard_pixel_concentration():
    n = 400
    proxy = torch.linspace(0.0, 1.0, n)
    sam = 0.5 + 2.0 * proxy
    valid = torch.ones(n, dtype=torch.bool)
    spectral_norm = torch.ones(n)
    intensity_grad = torch.zeros(n)
    fl = 0.4 + 0.2 * proxy
    c4l = 0.5 + 0.1 * proxy
    remainder = 0.7 + 0.1 * proxy

    out = _proxy_summary(
        proxy,
        sam,
        valid,
        spectral_norm=spectral_norm,
        intensity_gradient=intensity_grad,
        fl=fl,
        c4l_within_low=c4l,
        continuum_remainder=remainder,
        proxy_quantile=0.25,
        hard_fraction=0.10,
    )

    assert out["rho_proxy_sam"] > 0.99
    assert out["delta_sam_high_minus_low"] > 1.0
    assert out["hard_concentration_enrichment"] > 3.0
    assert out["delta_FL_high_minus_low"] > 0.0
    assert out["delta_C4L_within_low_high_minus_low"] > 0.0
    assert out["delta_continuum_remainder_high_minus_low"] > 0.0
