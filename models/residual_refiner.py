"""Parameter-matched ordinary residual CNN control for HPGC-Refiner.

All four observed inputs are concatenated. There is no attention, query/context
path, heterogeneity-dependent gating, or inactive parameter padding. Output
projection and zero-head initialization are shared with the HPGC reference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from .innovation3_gigi_refiner import TerminalGIGISpectralRefiner, _num_groups


@dataclass(frozen=True)
class ResidualMatch:
    target_params: int
    actual_params: int
    hidden_channels: int
    spatial_widths: tuple
    pointwise_width: int

    @property
    def relative_error(self):
        return abs(self.actual_params - self.target_params) / self.target_params


@lru_cache(maxsize=128)
def parameter_match(n_bands, n_msi_bands, reference_hidden=64, heads=4):
    # Counting a reference must not advance the training initialization stream.
    with torch.random.fork_rng(devices=[]):
        reference = TerminalGIGISpectralRefiner(n_bands, n_msi_bands, reference_hidden, heads, variant="full")
        target = sum(p.numel() for p in reference.parameters() if p.requires_grad)
    in_channels = 2 * n_bands + n_msi_bands + 1
    best = None
    # Fixed topology: stem, three 3x3 residual blocks, one 1x1 residual block,
    # head. Only ordinary feature widths are selected, using parameter counts
    # alone, never validation/test performance. Bottlenecks stay within [h/2,2h].
    for h in range(4, max(256, 4 * reference_hidden) + 1):
        fixed = 9 * h * (in_channels + n_bands) + 15 * h + n_bands
        for pointwise in range(max(1, h // 2), 2 * h + 1):
            total_width = round((target - fixed - (2 * h + 1) * pointwise) / (18 * h + 1))
            if not 3 * max(1, h // 2) <= total_width <= 6 * h:
                continue
            actual = fixed + (18 * h + 1) * total_width + (2 * h + 1) * pointwise
            candidate = (abs(actual - target), abs(h - reference_hidden), h, total_width, pointwise, actual)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("No ordinary residual CNN fits the HPGC parameter budget")
    _, _, hidden, total_width, pointwise, actual = best
    widths = tuple(total_width // 3 + int(i < total_width % 3) for i in range(3))
    match = ResidualMatch(target, actual, hidden, widths, pointwise)
    if match.relative_error > .001:
        raise ValueError(f"Residual parameter mismatch exceeds 0.1%: {match}")
    return match


class PlainResidualBlock(nn.Module):
    def __init__(self, channels, bottleneck, kernel_size):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, bottleneck, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(bottleneck, channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return x + self.conv2(F.silu(self.conv1(self.norm(x))))


class ParameterMatchedResidualRefiner(nn.Module):
    def __init__(self, n_bands, n_msi_bands, hidden_channels=64, heads=4,
                 tangent_output=True, eps=1e-8):
        super().__init__()
        self.n_bands, self.n_msi_bands = int(n_bands), int(n_msi_bands)
        self.reference_hidden_channels = int(hidden_channels)
        self.variant = "residual"
        self.tangent_output, self.eps = bool(tangent_output), float(eps)
        self.match = parameter_match(self.n_bands, self.n_msi_bands, hidden_channels, heads)
        h = self.hidden_channels = self.match.hidden_channels
        self.stem = nn.Sequential(
            nn.Conv2d(2 * self.n_bands + self.n_msi_bands + 1, h, 3, padding=1),
            nn.GroupNorm(_num_groups(h), h), nn.SiLU())
        self.blocks = nn.Sequential(
            *(PlainResidualBlock(h, width, 3) for width in self.match.spatial_widths),
            PlainResidualBlock(h, self.match.pointwise_width, 1))
        self.out_head = nn.Conv2d(h, self.n_bands, 3, padding=1)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        actual = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if actual != self.match.actual_params:
            raise RuntimeError("Residual CNN parameter-count formula disagrees with the actual model")

    def project_tangent(self, update, base):
        unit = base / torch.linalg.vector_norm(base, dim=1, keepdim=True).clamp_min(self.eps)
        return update - (update * unit).sum(dim=1, keepdim=True) * unit

    def forward(self, x_hat, hr_msi, r_phy, heterogeneity, *, return_details=False):
        if x_hat.ndim != 4 or x_hat.shape[1] != self.n_bands or r_phy.shape != x_hat.shape:
            raise ValueError("Expected matching BxC_hsi xHxW base and physical residual")
        if hr_msi.shape != (x_hat.shape[0], self.n_msi_bands, *x_hat.shape[-2:]):
            raise ValueError("HR-MSI shape is incompatible with the base")
        if heterogeneity.ndim == 3:
            heterogeneity = heterogeneity[:, None]
        if heterogeneity.shape != (x_hat.shape[0], 1, *x_hat.shape[-2:]):
            raise ValueError("Heterogeneity must be Bx1xHxW or BxHxW")
        feature = self.blocks(self.stem(torch.cat((x_hat, hr_msi, r_phy, heterogeneity), dim=1)))
        raw_update = self.out_head(feature)
        update = self.project_tangent(raw_update, x_hat) if self.tangent_output else raw_update
        refined = x_hat + update
        if not return_details:
            return refined
        return refined, dict(update=update, raw_update=raw_update,
                             heterogeneity=heterogeneity[:, 0], physical_input=r_phy)

    def parameter_report(self):
        return {**asdict(self.match), "relative_error": self.match.relative_error,
                "reference_hidden_channels": self.reference_hidden_channels,
                "topology": "concat4-stem-3_spatial_residual_blocks-1_pointwise_residual_block-head",
                "all_parameters_active": True}


def build_terminal_refiner(*, variant="full", **kwargs):
    if variant == "residual":
        return ParameterMatchedResidualRefiner(**kwargs)
    return TerminalGIGISpectralRefiner(variant=variant, **kwargs)
