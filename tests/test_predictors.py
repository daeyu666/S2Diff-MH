import torch

from models import RawMSIDirectPredictor, SpectralSpatialCleanHSIPredictor
from models.predictor_raw_direct import extract_legacy_raw_direct_state_dict


def test_v2_forward_shape():
    model = SpectralSpatialCleanHSIPredictor(
        n_bands=16, total_steps=12, base_channels=16, time_dim=32, spectral_hidden=4
    )
    x = torch.rand(2, 16, 32, 32)
    t = torch.tensor([3, 12])
    y = model(x, t)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_raw_direct_forward_shape():
    model = RawMSIDirectPredictor(
        n_bands=16,
        n_msi_bands=4,
        total_steps=12,
        base_channels=16,
        time_dim=32,
        spectral_hidden=4,
    )
    x = torch.rand(2, 16, 32, 32)
    msi = torch.rand(2, 4, 32, 32)
    t = torch.tensor([4, 11])
    y = model(x, msi, t)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_legacy_raw_direct_filter_drops_obsolete_gate_keys():
    checkpoint = {
        "model": {
            "in_proj.weight": torch.rand(4, 3, 3, 3),
            "gate1.gate.0.weight": torch.rand(4, 8, 1, 1),
            "gate2.time_proj.1.bias": torch.rand(8),
        }
    }
    state, ignored = extract_legacy_raw_direct_state_dict(checkpoint)
    assert "in_proj.weight" in state
    assert "gate1.gate.0.weight" not in state
    assert "gate1.gate.0.weight" in ignored
