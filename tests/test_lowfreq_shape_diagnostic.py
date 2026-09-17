import torch

from diagnose_lowfreq_shape import _continuum_basis, decompose_lowfreq_error
from diagnose_spectral_shape import _dct_matrix


def test_continuum_basis_is_orthonormal():
    q = _continuum_basis(31, device=torch.device("cpu"), dtype=torch.float32)
    eye = torch.eye(3)
    assert q.shape == (31, 3)
    assert torch.allclose(q.T @ q, eye, atol=1e-6, rtol=1e-6)


def test_low_dct_subfractions_sum_to_one_inside_low():
    torch.manual_seed(3)
    b, c, h, w = 1, 15, 3, 4
    target = torch.rand(b, c, h, w) + 0.2
    s = target.permute(0, 2, 3, 1).reshape(-1, c)

    raw = torch.randn_like(s)
    proj = (raw * s).sum(dim=1, keepdim=True) / s.square().sum(dim=1, keepdim=True)
    e_perp = raw - proj * s
    e_perp = 0.03 * e_perp / torch.linalg.vector_norm(e_perp, dim=1, keepdim=True).clamp_min(1e-8)
    pred = 1.05 * s + e_perp
    pred = pred.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    dct = _dct_matrix(c, device=torch.device("cpu"), dtype=torch.float32)
    basis = _continuum_basis(c, device=torch.device("cpu"), dtype=torch.float32)
    out = decompose_lowfreq_error(
        pred,
        target,
        dct=dct,
        continuum_basis=basis,
        wavelengths=None,
        eps=1e-8,
        a_min=1e-6,
    )
    mask = out["valid"]
    low_sum = (
        out["FL_C0"][mask]
        + out["FL_C1"][mask]
        + out["FL_C23"][mask]
        + out["FL_C4L"][mask]
    )
    continuum_sum = (
        out["FC_OFFSET"][mask]
        + out["FC_SLOPE"][mask]
        + out["FC_CURVATURE"][mask]
        + out["FC_REMAINDER"][mask]
    )

    assert torch.allclose(low_sum, torch.ones_like(low_sum), atol=1e-6, rtol=1e-6)
    assert torch.allclose(continuum_sum, torch.ones_like(continuum_sum), atol=2e-6, rtol=2e-6)
