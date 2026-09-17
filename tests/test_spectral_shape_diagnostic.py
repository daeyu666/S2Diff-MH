import math

import torch

from diagnose_spectral_shape import _dct_matrix, decompose_spectral_error


def test_orthonormal_dct_parseval():
    c = 12
    dct = _dct_matrix(c, device=torch.device("cpu"), dtype=torch.float32)
    eye = torch.eye(c)
    assert torch.allclose(dct @ dct.T, eye, atol=1e-6, rtol=1e-6)


def test_q_equals_tan_sam_and_fractions_sum_to_one():
    torch.manual_seed(0)
    b, c, h, w = 1, 12, 3, 4
    target = torch.rand(b, c, h, w) + 0.2
    s = target.permute(0, 2, 3, 1).reshape(-1, c)

    raw = torch.randn_like(s)
    proj = (raw * s).sum(dim=1, keepdim=True) / s.square().sum(dim=1, keepdim=True)
    e_perp = raw - proj * s
    e_perp = 0.05 * e_perp / torch.linalg.vector_norm(e_perp, dim=1, keepdim=True).clamp_min(1e-8)
    pred = 1.1 * s + e_perp
    pred = pred.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    dct = _dct_matrix(c, device=torch.device("cpu"), dtype=torch.float32)
    out = decompose_spectral_error(
        pred,
        target,
        dct=dct,
        wavelengths=None,
        eps=1e-8,
        a_min=1e-6,
    )
    mask = out["valid"]
    q = out["q"][mask]
    sam = out["sam"][mask] * math.pi / 180.0
    frac_sum = out["FL"][mask] + out["FM"][mask] + out["FH"][mask]

    assert torch.allclose(q, torch.tan(sam), atol=2e-5, rtol=2e-5)
    assert torch.allclose(frac_sum, torch.ones_like(frac_sum), atol=1e-6, rtol=1e-6)
    assert float(out["parseval_rel"][mask].max().item()) < 1e-5
