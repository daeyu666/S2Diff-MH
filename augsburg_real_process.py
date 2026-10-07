"""Augsburg-Real x3 progressive physical process."""

from __future__ import annotations

from degradations.effective_gaussian import EffectiveGaussianDegradation
from degradations.progressive import ProgressiveDegradation


def build_augsburg_real_process(
    *,
    effective_sigma: float,
    diffusion_steps: int = 12,
    truncate: float = 3.0,
) -> ProgressiveDegradation:
    operator = EffectiveGaussianDegradation(
        scale_ratio=3,
        terminal_sigma=float(effective_sigma),
        truncate=float(truncate),
    )
    return ProgressiveDegradation(
        operator=operator,
        total_steps=int(diffusion_steps),
        stages=[1, 2, 3],
        default_lift_mode="normalized_adjoint",
    )
