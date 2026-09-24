import torch

from models import GlobalInterGuidedInteraction, TerminalGIGISpectralRefiner


def test_gigi_shape_and_attention():
    block = GlobalInterGuidedInteraction(channels=32, heads=4)
    q = torch.randn(2, 32, 16, 16)
    c = torch.randn(2, 32, 16, 16)
    out, attn = block(q, c, return_attention=True)
    assert out.shape == q.shape
    assert attn.shape == (2, 4, 8, 8)
    probs = attn.sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)


def test_terminal_refiner_is_exact_identity_at_initialization():
    model = TerminalGIGISpectralRefiner(
        n_bands=31,
        n_msi_bands=4,
        hidden_channels=32,
        heads=4,
        variant="full",
        tangent_output=True,
    )
    x = torch.rand(2, 31, 16, 16)
    msi = torch.rand(2, 4, 16, 16)
    r_phy = torch.randn_like(x) * 0.01
    hetero = torch.rand(2, 16, 16)

    out, details = model(
        x,
        msi,
        r_phy,
        hetero,
        return_details=True,
    )
    assert torch.equal(out, x)
    assert torch.count_nonzero(details["update"]) == 0
    assert "attention" in details


def test_ablation_variants_preserve_shape():
    x = torch.rand(1, 31, 8, 8)
    msi = torch.rand(1, 4, 8, 8)
    r_phy = torch.randn_like(x)
    hetero = torch.rand(1, 8, 8)

    for variant in ("conv", "gigi", "gigi_hetero", "gigi_phy", "full"):
        model = TerminalGIGISpectralRefiner(
            n_bands=31,
            n_msi_bands=4,
            hidden_channels=32,
            heads=4,
            variant=variant,
        )
        out = model(x, msi, r_phy, hetero)
        assert out.shape == x.shape


def test_tangent_projection_removes_radial_component():
    model = TerminalGIGISpectralRefiner(
        n_bands=31,
        n_msi_bands=4,
        hidden_channels=32,
        heads=4,
        variant="full",
        tangent_output=True,
    )
    base = torch.rand(1, 31, 5, 5) + 0.1
    update = torch.randn_like(base)
    tangent = model.project_tangent(update, base)
    unit = base / torch.linalg.vector_norm(base, dim=1, keepdim=True)
    radial = (tangent * unit).sum(dim=1)
    assert radial.abs().max().item() < 1e-5
