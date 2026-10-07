"""Effective inter-resolution Gaussian degradation for Augsburg-Real.

This operator is intentionally separate from the benchmark ``physical`` mode.
The Augsburg-Real 10->30 m transfer is calibrated from the MDAS training pair
and is therefore an *effective inter-resolution response*, not a claim about
the native EnMAP instrument PSF.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import BaseDegradation
from .common import (
    area_average_adjoint,
    area_average_downsample,
    assert_spatial_size,
    depthwise_psf,
    resize_up,
    validate_hsi_tensor,
)


class EffectiveGaussianDegradation(BaseDegradation):
    """Fixed Gaussian blur + detector area integration/downsampling.

    ``terminal_sigma`` is expressed in HR-grid pixels.  Intermediate diffusion
    states linearly interpolate blur strength from zero to this calibrated
    terminal value.
    """

    mode = "effective_gaussian"

    def __init__(
        self,
        scale_ratio: int = 3,
        terminal_sigma: float = 1.2,
        truncate: float = 3.0,
    ):
        super().__init__(scale_ratio=scale_ratio)
        if terminal_sigma < 0.0:
            raise ValueError("terminal_sigma must be >= 0")
        if truncate <= 0.0:
            raise ValueError("truncate must be > 0")
        self.terminal_sigma = float(terminal_sigma)
        self.truncate = float(truncate)

    def sigma_at_strength(self, strength: float) -> float:
        strength = float(min(max(strength, 0.0), 1.0))
        return self.terminal_sigma * strength

    def degrade_at(self, x: torch.Tensor, *, scale: int, strength: float) -> torch.Tensor:
        validate_hsi_tensor(x)
        sigma_t = self.sigma_at_strength(strength)
        optical = depthwise_psf(x, sigma_t, truncate=self.truncate)
        return area_average_downsample(optical, int(scale))

    def adjoint_at(self, y: torch.Tensor, *, scale: int, strength: float) -> torch.Tensor:
        validate_hsi_tensor(y)
        sigma_t = self.sigma_at_strength(strength)
        backprojected = area_average_adjoint(y, int(scale))
        return depthwise_psf(backprojected, sigma_t, truncate=self.truncate)

    def lift(
        self,
        y: torch.Tensor,
        *,
        scale: int,
        strength: float,
        lift_mode: str,
        target_size: Optional[Tuple[int, int]] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        validate_hsi_tensor(y)
        scale = int(scale)
        if target_size is None:
            target_size = (y.shape[-2] * scale, y.shape[-1] * scale)

        if lift_mode in ("bilinear", "nearest"):
            out = resize_up(y, scale, mode=lift_mode, target_size=target_size)
            assert_spatial_size(out, target_size, "interpolation lift")
            return out

        if lift_mode == "adjoint":
            out = self.adjoint_at(y, scale=scale, strength=strength)
            assert_spatial_size(out, target_size, "adjoint lift")
            return out

        if lift_mode == "normalized_adjoint":
            numerator = self.adjoint_at(y, scale=scale, strength=strength)
            ones_hr = torch.ones(
                (y.shape[0], 1, target_size[0], target_size[1]),
                dtype=y.dtype,
                device=y.device,
            )
            observed_ones = self.degrade_at(
                ones_hr, scale=scale, strength=strength
            )
            denominator = self.adjoint_at(
                observed_ones, scale=scale, strength=strength
            )
            out = numerator / denominator.clamp_min(float(eps))
            assert_spatial_size(out, target_size, "normalized-adjoint lift")
            return out

        raise ValueError(
            "lift_mode must be one of: bilinear, nearest, adjoint, normalized_adjoint"
        )

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", terminal_sigma={self.terminal_sigma:.6f}, truncate={self.truncate}"
        )
