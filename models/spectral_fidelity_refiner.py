"""Innovation 3: MSI-heterogeneity-guided spectral-fidelity refinement.

The refiner implements the mechanism selected by the Innovation-3 diagnostics:

    HR-MSI -> local heterogeneity -> where to refine
    HSI state/base prediction -> spectral residual -> how to refine

The MSI branch never predicts an HSI spectrum.  It only produces a scalar,
monotone spatial gate.  Spectral correction is generated from HSI/state
quantities, optionally projected to the diagnosed C4:L broad-shape DCT band
and to the tangent space orthogonal to the current spectral direction.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANT_FLAGS = {
    "generic": (False, False, False),
    "hetero": (True, False, False),
    "broad": (True, True, False),
    "full": (True, True, True),
}


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus target must be > 0")
    return math.log(math.expm1(value))


def dct_ii_matrix(n: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix C with coeff = x @ C.T."""
    if n < 1:
        raise ValueError("n must be >= 1")
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    j = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    matrix = torch.cos(math.pi / n * (j + 0.5) * k)
    matrix = matrix * math.sqrt(2.0 / n)
    matrix[0] = matrix[0] / math.sqrt(2.0)
    return matrix


def local_spectral_heterogeneity(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """3x3 local variance magnitude of unit-normalized spectra.

    This is the same local statistic used by the Innovation-3 diagnostics.
    Reflect padding is used to obtain a dense map.  Diagnostic comparisons
    still use only the one-pixel interior through ranked_msi_heterogeneity().
    """
    if x.ndim != 4:
        raise ValueError("x must be BxCxHxW")
    if x.shape[-2] < 3 or x.shape[-1] < 3:
        raise ValueError("spatial size must be at least 3x3")
    xf = x.float()
    norm = torch.linalg.vector_norm(xf, dim=1, keepdim=True).clamp_min(eps)
    unit = xf / norm
    padded = F.pad(unit, (1, 1, 1, 1), mode="reflect")
    mean = F.avg_pool2d(padded, kernel_size=3, stride=1)
    mean_sq = F.avg_pool2d(padded.square(), kernel_size=3, stride=1)
    var = (mean_sq - mean.square()).clamp_min(0.0)
    return torch.sqrt(var.sum(dim=1).clamp_min(0.0))


def _rank01(values: torch.Tensor) -> torch.Tensor:
    """Rank a one-dimensional tensor to [0,1].

    This intentionally matches the diagnostic rank transform: exact ties are
    not averaged because they are rare for the continuous heterogeneity map.
    """
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    n = values.numel()
    if n == 0:
        raise ValueError("cannot rank an empty tensor")
    if n == 1:
        return torch.full_like(values, 0.5, dtype=torch.float32)
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(n, device=values.device, dtype=torch.float32)
    return ranks / float(n - 1)


def ranked_msi_heterogeneity(hr_msi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return BxHxW within-patch heterogeneity ranks available at inference.

    The interior ranking exactly follows the diagnostic support.  The one-pixel
    border is filled by replication of the nearest interior rank so the map can
    be consumed by a convolutional predictor without NaNs or a special border
    branch.
    """
    raw = local_spectral_heterogeneity(hr_msi, eps=eps)
    b, h, w = raw.shape
    ranked = torch.empty_like(raw, dtype=torch.float32)
    for index in range(b):
        interior = raw[index, 1:-1, 1:-1]
        interior_rank = _rank01(interior.reshape(-1)).reshape(h - 2, w - 2)
        dense = F.pad(
            interior_rank.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode="replicate",
        )[0, 0]
        ranked[index] = dense
    return ranked


class HSISpectralResidualGenerator(nn.Module):
    """Small HSI-only residual generator.

    Inputs are x_t, the frozen Raw-Direct base prediction, and normalized time.
    HR-MSI is intentionally absent: MSI determines only the spatial gate.
    """

    def __init__(self, n_bands: int, total_steps: int, hidden_channels: int = 64):
        super().__init__()
        self.n_bands = int(n_bands)
        self.total_steps = int(total_steps)
        hidden = int(hidden_channels)
        if self.n_bands < 1 or self.total_steps < 1 or hidden < 1:
            raise ValueError("invalid residual-generator dimensions")

        self.in_proj = nn.Conv2d(2 * self.n_bands + 1, hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1, groups=hidden
        )
        self.mix = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.norm = nn.GroupNorm(_num_groups(hidden), hidden)
        self.out = nn.Conv2d(hidden, self.n_bands, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        base_x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.shape != base_x0.shape:
            raise ValueError("x_t and base_x0 must have identical shape")
        if x_t.ndim != 4 or x_t.shape[1] != self.n_bands:
            raise ValueError("unexpected HSI shape")
        batch, _, height, width = x_t.shape
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x_t.device)
        if t.ndim == 0:
            t = t.repeat(batch)
        if t.ndim != 1 or t.shape[0] != batch:
            raise ValueError(f"t must be scalar or [B={batch}]")
        time = (t.float() / float(self.total_steps)).view(batch, 1, 1, 1)
        time = time.expand(batch, 1, height, width)

        z = torch.cat([x_t, base_x0, time], dim=1)
        z = F.silu(self.in_proj(z))
        z = F.silu(self.depthwise(z))
        z = F.silu(self.norm(self.mix(z)))
        return self.out(z)


class HeterogeneityGuidedSpectralRefiner(nn.Module):
    """A1-A4 Innovation-3 spectral refinement module."""

    def __init__(
        self,
        n_bands: int,
        total_steps: int = 12,
        hidden_channels: int = 64,
        variant: str = "full",
        gate_init_slope: float = 6.0,
        gate_init_bias: float = -3.0,
        eta_init: float = 0.10,
        eps: float = 1e-8,
    ):
        super().__init__()
        if variant not in VARIANT_FLAGS:
            raise ValueError(f"unknown variant={variant!r}; choices={tuple(VARIANT_FLAGS)}")
        self.n_bands = int(n_bands)
        self.total_steps = int(total_steps)
        self.variant = str(variant)
        self.use_heterogeneity_gate, self.use_broadshape, self.use_tangent = VARIANT_FLAGS[variant]
        self.eps = float(eps)

        low_end = self.n_bands // 3
        if low_end <= 4:
            raise ValueError(
                f"n_bands={self.n_bands} is too small for diagnosed C4:L interval"
            )
        self.low_end = int(low_end)

        dct = dct_ii_matrix(self.n_bands)
        mask = torch.zeros(self.n_bands, dtype=torch.float32)
        mask[4:self.low_end] = 1.0
        self.register_buffer("dct_matrix", dct, persistent=True)
        self.register_buffer("broadshape_mask", mask, persistent=True)

        self.generator = HSISpectralResidualGenerator(
            self.n_bands,
            self.total_steps,
            hidden_channels=hidden_channels,
        )
        self.gate_log_slope = nn.Parameter(
            torch.tensor(_inverse_softplus(float(gate_init_slope)), dtype=torch.float32)
        )
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_init_bias), dtype=torch.float32))
        self.eta_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(float(eta_init)), dtype=torch.float32)
        )
        if not self.use_heterogeneity_gate:
            self.gate_log_slope.requires_grad_(False)
            self.gate_bias.requires_grad_(False)

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self.eta_raw)

    @property
    def gate_slope(self) -> torch.Tensor:
        return F.softplus(self.gate_log_slope) + 1e-4

    def dct(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.n_bands:
            raise ValueError("x must be BxCxHxW with configured HSI bands")
        bhwc = x.permute(0, 2, 3, 1)
        coeff = torch.einsum("bhwc,kc->bhwk", bhwc, self.dct_matrix)
        return coeff.permute(0, 3, 1, 2).contiguous()

    def idct(self, coeff: torch.Tensor) -> torch.Tensor:
        if coeff.ndim != 4 or coeff.shape[1] != self.n_bands:
            raise ValueError("coeff must be BxCxHxW with configured HSI bands")
        bhwk = coeff.permute(0, 2, 3, 1)
        signal = torch.einsum("bhwk,kc->bhwc", bhwk, self.dct_matrix)
        return signal.permute(0, 3, 1, 2).contiguous()

    def project_broadshape(self, residual: torch.Tensor) -> torch.Tensor:
        coeff = self.dct(residual)
        coeff = coeff * self.broadshape_mask.view(1, -1, 1, 1)
        return self.idct(coeff)

    def project_tangent(
        self,
        residual: torch.Tensor,
        base_x0: torch.Tensor,
        *,
        preserve_broadshape: bool = False,
    ) -> torch.Tensor:
        """Remove the component that changes the current spectral magnitude.

        When preserve_broadshape=True, the subtraction direction is P_S(u),
        where S is the diagnosed C4:L DCT subspace.  Because residual is already
        in S, this keeps the correction inside S while enforcing <r,u>=0.
        """
        base_norm = torch.linalg.vector_norm(
            base_x0, dim=1, keepdim=True
        ).clamp_min(self.eps)
        unit = base_x0 / base_norm
        numerator = (residual * unit).sum(dim=1, keepdim=True)
        if not preserve_broadshape:
            return residual - numerator * unit

        unit_subspace = self.project_broadshape(unit)
        denominator = (unit_subspace * unit).sum(dim=1, keepdim=True)
        safe = denominator.abs() > self.eps
        scale = torch.where(
            safe,
            numerator / denominator.clamp_min(self.eps),
            torch.zeros_like(numerator),
        )
        return residual - scale * unit_subspace

    def gate_from_rank(self, rank: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_slope * rank + self.gate_bias)

    def forward(
        self,
        x_t: torch.Tensor,
        base_x0: torch.Tensor,
        hr_msi: torch.Tensor,
        t: torch.Tensor,
        *,
        return_details: bool = False,
    ):
        if base_x0.shape != x_t.shape:
            raise ValueError("base_x0 and x_t must have identical shape")
        if hr_msi.ndim != 4 or hr_msi.shape[0] != x_t.shape[0]:
            raise ValueError("hr_msi must be BxMxHxW with matching batch")
        if hr_msi.shape[-2:] != x_t.shape[-2:]:
            raise ValueError("HR-MSI and HSI spatial sizes must match")

        raw_residual = self.generator(x_t, base_x0, t)
        spectral_residual = (
            self.project_broadshape(raw_residual)
            if self.use_broadshape
            else raw_residual
        )
        directional_residual = (
            self.project_tangent(
                spectral_residual,
                base_x0,
                preserve_broadshape=self.use_broadshape,
            )
            if self.use_tangent
            else spectral_residual
        )

        risk = ranked_msi_heterogeneity(hr_msi, eps=self.eps)
        if self.use_heterogeneity_gate:
            gate = self.gate_from_rank(risk)
        else:
            gate = torch.ones_like(risk)

        pre_update = self.eta * gate.unsqueeze(1) * directional_residual

        if self.use_tangent:
            base_norm = torch.linalg.vector_norm(
                base_x0, dim=1, keepdim=True
            ).clamp_min(self.eps)
            unit = base_x0 / base_norm
            new_unit = F.normalize(unit + pre_update, dim=1, eps=self.eps)
            refined = base_norm * new_unit
        else:
            refined = base_x0 + pre_update

        if not return_details:
            return refined
        details: Dict[str, torch.Tensor] = {
            "risk": risk,
            "gate": gate,
            "raw_residual": raw_residual,
            "spectral_residual": spectral_residual,
            "directional_residual": directional_residual,
            "pre_update": pre_update,
            "applied_update": refined - base_x0,
            "base_x0": base_x0,
        }
        return refined, details


class HeterogeneityGuidedSpectralPredictor(nn.Module):
    """Frozen Raw-Direct backbone plus Innovation-3 refinement."""

    requires_msi = True

    def __init__(
        self,
        backbone: nn.Module,
        n_bands: int,
        total_steps: int = 12,
        hidden_channels: int = 64,
        variant: str = "full",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.refiner = HeterogeneityGuidedSpectralRefiner(
            n_bands=n_bands,
            total_steps=total_steps,
            hidden_channels=hidden_channels,
            variant=variant,
        )
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    @property
    def variant(self) -> str:
        return self.refiner.variant

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _base_predict(
        self,
        x_t: torch.Tensor,
        hr_msi: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(x_t, hr_msi, t)
        return self.backbone(x_t, hr_msi, t)

    def forward(
        self,
        x_t: torch.Tensor,
        hr_msi: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        base = self._base_predict(x_t, hr_msi, t)
        return self.refiner(x_t, base, hr_msi, t)

    def forward_with_details(
        self,
        x_t: torch.Tensor,
        hr_msi: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base = self._base_predict(x_t, hr_msi, t)
        return self.refiner(
            x_t,
            base,
            hr_msi,
            t,
            return_details=True,
        )
