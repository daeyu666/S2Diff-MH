"""Terminal GIGI spectral refiner for Innovation 3.

This module adapts the global inter-guided interaction idea used by CLSNet to
our diagnosed spectral-fidelity failure mode.  It is deliberately terminal:
the frozen diffusion model first reconstructs a complete HR-HSI, then the
refiner predicts one HR spectral correction.

Roles of the inputs:
    x_hat          current HR-HSI reconstruction / spectral state
    hr_msi         HR spatial-material observation
    r_phy          observable terminal physical residual back-projected to HR
    heterogeneity  ranked MSI local spectral heterogeneity

Heterogeneity is not a multiplicative correction gate.  It changes the query
used to infer a correction direction.  The physical residual enters the
key/value context and supplies observation-consistent discrepancy evidence.

Ablations:
    conv          local convolutional control, no GIGI, physical residual on
    gigi          GIGI with x_hat + HR-MSI only
    gigi_hetero   GIGI + heterogeneity-aware query
    gigi_phy      GIGI + physical residual context
    full          GIGI + heterogeneity-aware query + physical residual context

The output head is zero initialized, so every variant is exactly the frozen
baseline at initialization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANT_FLAGS = {
    "conv": (False, False, True),
    "gigi": (True, False, False),
    "gigi_hetero": (True, True, False),
    "gigi_phy": (True, False, True),
    "full": (True, True, True),
}


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        h = int(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), h, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(h), h),
            nn.SiLU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        hidden = max(int(round(channels * expansion)), channels)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class GlobalInterGuidedInteraction(nn.Module):
    """GIGI-style transposed global correlation.

    Query and context have shape BxCxHxW.  Correlation is accumulated across
    all HR spatial locations, producing a compact per-head channel interaction
    matrix rather than an HW-by-HW attention matrix.
    """

    def __init__(self, channels: int, heads: int = 4, expansion: float = 2.0):
        super().__init__()
        channels = int(channels)
        heads = int(heads)
        if channels < 1 or heads < 1 or channels % heads != 0:
            raise ValueError("channels must be positive and divisible by heads")
        self.channels = channels
        self.heads = heads
        self.dim_head = channels // heads

        self.q_norm = nn.GroupNorm(_num_groups(channels), channels)
        self.kv_norm = nn.GroupNorm(_num_groups(channels), channels)
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
        )
        self.ffn = DepthwiseFeedForward(channels, expansion=expansion)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.reshape(b, self.heads, self.dim_head, h * w)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        if query.shape != context.shape:
            raise ValueError("query and context must have identical BxCxHxW shape")
        if query.ndim != 4 or query.shape[1] != self.channels:
            raise ValueError("unexpected GIGI feature shape")

        q = self._split_heads(self.to_q(self.q_norm(query)))
        k = self._split_heads(self.to_k(self.kv_norm(context)))
        v_map = self.to_v(self.kv_norm(context))
        v = self._split_heads(v_map)

        # CLSNet-style global transposed correlation: normalize each latent
        # channel across all spatial positions, then correlate channel pairs.
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attention = torch.matmul(k, q.transpose(-2, -1))
        attention = torch.softmax(attention * self.temperature, dim=-1)

        out = torch.matmul(attention, v).reshape_as(query)
        out = self.proj(out) + self.pos_emb(v_map)
        out = query + out
        out = out + self.ffn(out)

        if return_attention:
            return out, attention
        return out


class LocalConvFusion(nn.Module):
    """Parameter-control alternative to GIGI."""

    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(_num_groups(channels), channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(_num_groups(channels), channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block1(x)
        return x + self.block2(x)


class TerminalGIGISpectralRefiner(nn.Module):
    """One-shot full-resolution spectral correction after reverse diffusion."""

    def __init__(
        self,
        n_bands: int,
        n_msi_bands: int,
        hidden_channels: int = 64,
        heads: int = 4,
        variant: str = "full",
        tangent_output: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if variant not in VARIANT_FLAGS:
            raise ValueError(
                f"unknown variant={variant!r}; choices={tuple(VARIANT_FLAGS)}"
            )
        if n_bands < 1 or n_msi_bands < 1:
            raise ValueError("n_bands and n_msi_bands must be >=1")
        if hidden_channels < 4 or hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be >=4 and divisible by heads")

        self.n_bands = int(n_bands)
        self.n_msi_bands = int(n_msi_bands)
        self.hidden_channels = int(hidden_channels)
        self.heads = int(heads)
        self.variant = str(variant)
        self.use_gigi, self.use_heterogeneity, self.use_physical = VARIANT_FLAGS[
            self.variant
        ]
        self.tangent_output = bool(tangent_output)
        self.eps = float(eps)

        h = self.hidden_channels
        self.hsi_encoder = ConvEncoder(self.n_bands, h)
        self.msi_encoder = ConvEncoder(self.n_msi_bands, h)
        self.phy_encoder = ConvEncoder(self.n_bands, h)
        self.hetero_encoder = ConvEncoder(1, h)

        # Heterogeneity enters only the query path.  A disabled source is a
        # literal zero feature, keeping the ablations structurally comparable.
        self.query_builder = nn.Sequential(
            nn.Conv2d(3 * h, h, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(h), h),
            nn.SiLU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
        )
        # Physical residual enters only the correction context (K/V side).
        self.context_builder = nn.Sequential(
            nn.Conv2d(3 * h, h, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(h), h),
            nn.SiLU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
        )

        if self.use_gigi:
            self.self_guide = GlobalInterGuidedInteraction(h, heads=heads)
            self.gigi = GlobalInterGuidedInteraction(h, heads=heads)
            self.conv_control = None
        else:
            self.self_guide = None
            self.gigi = None
            self.conv_control = LocalConvFusion(h)

        self.fuse = nn.Sequential(
            nn.Conv2d(2 * h, h, kernel_size=1, bias=False),
            nn.GroupNorm(_num_groups(h), h),
            nn.SiLU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.out_head = nn.Conv2d(h, self.n_bands, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def project_tangent(
        self,
        update: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        norm = torch.linalg.vector_norm(base, dim=1, keepdim=True).clamp_min(
            self.eps
        )
        unit = base / norm
        radial = (update * unit).sum(dim=1, keepdim=True)
        return update - radial * unit

    def forward(
        self,
        x_hat: torch.Tensor,
        hr_msi: torch.Tensor,
        r_phy: torch.Tensor,
        heterogeneity: torch.Tensor,
        *,
        return_details: bool = False,
    ):
        if x_hat.ndim != 4 or x_hat.shape[1] != self.n_bands:
            raise ValueError("x_hat must be BxCxHxW with configured HSI bands")
        if r_phy.shape != x_hat.shape:
            raise ValueError("r_phy must have the same shape as x_hat")
        if (
            hr_msi.ndim != 4
            or hr_msi.shape[0] != x_hat.shape[0]
            or hr_msi.shape[1] != self.n_msi_bands
            or hr_msi.shape[-2:] != x_hat.shape[-2:]
        ):
            raise ValueError("hr_msi shape is incompatible with x_hat")

        if heterogeneity.ndim == 3:
            heterogeneity = heterogeneity.unsqueeze(1)
        if (
            heterogeneity.ndim != 4
            or heterogeneity.shape[0] != x_hat.shape[0]
            or heterogeneity.shape[1] != 1
            or heterogeneity.shape[-2:] != x_hat.shape[-2:]
        ):
            raise ValueError("heterogeneity must be BxHxW or Bx1xHxW")

        fx = self.hsi_encoder(x_hat)
        fm = self.msi_encoder(hr_msi)

        if self.use_physical:
            fr = self.phy_encoder(r_phy)
        else:
            fr = torch.zeros_like(fx)

        if self.use_heterogeneity:
            fh = self.hetero_encoder(heterogeneity)
        else:
            fh = torch.zeros_like(fx)

        query = self.query_builder(torch.cat([fx, fm, fh], dim=1))
        context = self.context_builder(torch.cat([fx, fm, fr], dim=1))

        attention = None
        if self.use_gigi:
            guided_query = self.self_guide(query, query)
            if return_details:
                global_feature, attention = self.gigi(
                    guided_query,
                    context,
                    return_attention=True,
                )
            else:
                global_feature = self.gigi(guided_query, context)
        else:
            global_feature = self.conv_control(context)

        fused = self.fuse(torch.cat([global_feature, context], dim=1))
        raw_update = self.out_head(fused)
        update = (
            self.project_tangent(raw_update, x_hat)
            if self.tangent_output
            else raw_update
        )
        refined = x_hat + update

        if not return_details:
            return refined
        details: Dict[str, torch.Tensor] = {
            "update": update,
            "raw_update": raw_update,
            "query": query,
            "context": context,
            "heterogeneity": heterogeneity[:, 0],
            "physical_input": r_phy,
        }
        if attention is not None:
            details["attention"] = attention
        return refined, details
