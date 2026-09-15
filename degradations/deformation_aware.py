"""Deformation-aware physical trajectory for CDRDI x Innovation-1 Stage-2.

The clean HR-HSI is defined in the reliable MSI coordinate system.  For every
sensor state t>=1, acquisition is synthesized forward as

    A_{t,phi}(X) = D_t(W_phi(X)).

The lift is the normalized adjoint of that *forward* acquisition operator,
not an inverse warp of the observed LR-HSI.  The warp adjoint implemented here
is the transpose of the bilinear border-sampling matrix used by grid_sample.

`t=0` is explicitly the clean latent boundary X, not a sensor acquisition
state.  Therefore A~_0 is the identity while the same fixed acquisition
geometry phi is used for every degraded state t>=1.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from cdrdi_geometry import forward_warp, sampling_coordinates
from .progressive import ProgressiveDegradation, ProgressiveState


def bilinear_border_warp_adjoint(
    values: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    theta_deg: torch.Tensor,
    local_field: torch.Tensor,
    *,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Transpose of `forward_warp` for fixed geometry.

    `forward_warp` uses `grid_sample(..., mode="bilinear",
    padding_mode="border", align_corners=True)`.  The exact linear transpose
    therefore scatters each sampled output pixel back to its four clamped
    source neighbors with the same bilinear weights.  This is an adjoint, not
    inverse resampling.
    """
    if values.ndim != 4:
        raise ValueError("values must have shape BxCxHxW")
    b, c, h_out, w_out = values.shape
    if target_size is None:
        target_size = (h_out, w_out)
    h, w = map(int, target_size)
    if (h_out, w_out) != (h, w):
        raise ValueError(
            "Stage-2 warp adjoint currently assumes the forward warp preserves "
            "the HR grid size"
        )
    if local_field.shape != (b, 2, h, w):
        raise ValueError(
            f"local_field must have shape {(b, 2, h, w)}, got {tuple(local_field.shape)}"
        )

    sx, sy = sampling_coordinates(h, w, dx, dy, theta_deg, local_field)
    # grid_sample border padding is equivalent to clamping the continuous
    # sample coordinate before bilinear interpolation.
    sx = sx.clamp(0.0, float(max(w - 1, 0)))
    sy = sy.clamp(0.0, float(max(h - 1, 0)))

    x0 = torch.floor(sx).to(torch.long)
    y0 = torch.floor(sy).to(torch.long)
    x1 = (x0 + 1).clamp(max=w - 1)
    y1 = (y0 + 1).clamp(max=h - 1)

    wx1 = sx - x0.to(sx.dtype)
    wy1 = sy - y0.to(sy.dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    flat_values = values.reshape(b, c, -1)
    out = values.new_zeros((b, c, h * w))

    def scatter(ix: torch.Tensor, iy: torch.Tensor, weight: torch.Tensor) -> None:
        index = (iy * w + ix).reshape(b, 1, -1).expand(-1, c, -1)
        contribution = flat_values * weight.reshape(b, 1, -1)
        out.scatter_add_(2, index, contribution)

    scatter(x0, y0, wx0 * wy0)
    scatter(x1, y0, wx1 * wy0)
    scatter(x0, y1, wx0 * wy1)
    scatter(x1, y1, wx1 * wy1)
    return out.view(b, c, h, w)


class DeformationAwareProgressiveDegradation:
    """Fixed-phi wrapper around the Innovation-1 progressive physical process.

    For t>=1:
        A_t       = D_t o W_phi
        A~_t      = A_t^dagger A_t
        A_t^dagger(y) = A_t^*(y) / A_t^* A_t(1)

    with A_t^* = W_phi^* o D_t^*.  The geometry is fixed for the whole reverse
    trajectory.  `t=0` is the clean reference-coordinate latent and is exactly
    identity so the deterministic reverse process still terminates at X.
    """

    def __init__(
        self,
        base_process: ProgressiveDegradation,
        *,
        rigid: torch.Tensor,
        local_field: torch.Tensor,
        eps: float = 1e-8,
    ):
        if base_process.operator.mode != "physical":
            raise ValueError("Stage-2 deformation-aware trajectory requires physical degradation")
        if not hasattr(base_process.operator, "adjoint_at"):
            raise TypeError("physical operator must expose adjoint_at")
        if rigid.ndim != 2 or rigid.shape[1] != 3:
            raise ValueError("rigid must have shape Bx3 containing dx,dy,theta_deg")
        if local_field.ndim != 4 or local_field.shape[1] != 2:
            raise ValueError("local_field must have shape Bx2xHxW")
        if rigid.shape[0] != local_field.shape[0]:
            raise ValueError("rigid/local batch mismatch")
        self.base_process = base_process
        self.operator = base_process.operator
        self.total_steps = base_process.total_steps
        self.stages = base_process.stages
        self.default_lift_mode = "normalized_adjoint"
        self.rigid = rigid.detach()
        self.local_field = local_field.detach()
        self.eps = float(eps)
        self._normalizer_cache: Dict[Tuple[int, int, int, str, str, int], torch.Tensor] = {}

    def _validate_t(self, t: int) -> int:
        return self.base_process._validate_t(t)

    def state(self, t: int) -> ProgressiveState:
        return self.base_process.state(t)

    def _geometry(
        self,
        batch: int,
        target_size: Tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = map(int, target_size)
        rigid = self.rigid.to(device=device, dtype=dtype)
        local = self.local_field.to(device=device, dtype=dtype)
        if local.shape[-2:] != (h, w):
            raise ValueError(
                f"geometry HR size {tuple(local.shape[-2:])} does not match target {(h, w)}"
            )
        if rigid.shape[0] == 1 and batch > 1:
            rigid = rigid.expand(batch, -1)
            local = local.expand(batch, -1, -1, -1)
        if rigid.shape[0] != batch:
            raise ValueError(f"geometry batch={rigid.shape[0]} does not match data batch={batch}")
        return rigid, local

    def _warp(self, x: torch.Tensor) -> torch.Tensor:
        rigid, local = self._geometry(
            x.shape[0], tuple(x.shape[-2:]), device=x.device, dtype=x.dtype
        )
        return forward_warp(
            x,
            rigid[:, 0],
            rigid[:, 1],
            rigid[:, 2],
            local,
        )

    def _warp_adjoint(self, y_hr: torch.Tensor) -> torch.Tensor:
        rigid, local = self._geometry(
            y_hr.shape[0], tuple(y_hr.shape[-2:]), device=y_hr.device, dtype=y_hr.dtype
        )
        return bilinear_border_warp_adjoint(
            y_hr,
            rigid[:, 0],
            rigid[:, 1],
            rigid[:, 2],
            local,
            target_size=tuple(y_hr.shape[-2:]),
        )

    def acquire_at(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Forward acquisition A_{t,phi}(x)."""
        t = self._validate_t(t)
        if t == 0:
            return x
        state = self.state(t)
        warped = self._warp(x)
        return self.operator.degrade_at(
            warped,
            scale=state.scale,
            strength=state.strength,
        )

    # Keep the Innovation-1 interface name so existing reverse/evaluation code
    # can operate on this wrapper without special cases.
    def degrade_at(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return self.acquire_at(x, t)

    def adjoint_at(
        self,
        y: torch.Tensor,
        t: int,
        *,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Unnormalized A_t^* = W_phi^* D_t^*."""
        t = self._validate_t(t)
        if t == 0:
            if tuple(y.shape[-2:]) != tuple(target_size):
                raise ValueError("t=0 adjoint expects the clean HR spatial size")
            return y
        state = self.state(t)
        sensor_backprojection = self.operator.adjoint_at(
            y,
            scale=state.scale,
            strength=state.strength,
        )
        if tuple(sensor_backprojection.shape[-2:]) != tuple(target_size):
            raise RuntimeError(
                "sensor adjoint produced an unexpected HR size: "
                f"{tuple(sensor_backprojection.shape[-2:])} vs {tuple(target_size)}"
            )
        return self._warp_adjoint(sensor_backprojection)

    def _normalizer(
        self,
        t: int,
        *,
        target_size: Tuple[int, int],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        h, w = map(int, target_size)
        key = (int(t), h, w, str(device), str(dtype), int(batch))
        cached = self._normalizer_cache.get(key)
        if cached is not None:
            return cached
        ones = torch.ones((batch, 1, h, w), device=device, dtype=dtype)
        observed_ones = self.acquire_at(ones, t)
        denom = self.adjoint_at(observed_ones, t, target_size=(h, w)).clamp_min(self.eps)
        self._normalizer_cache[key] = denom.detach()
        return self._normalizer_cache[key]

    def lift_at(
        self,
        y: torch.Tensor,
        t: int,
        *,
        target_size: Tuple[int, int],
        lift_mode: Optional[str] = None,
    ) -> torch.Tensor:
        t = self._validate_t(t)
        mode = lift_mode or self.default_lift_mode
        if t == 0:
            if tuple(y.shape[-2:]) != tuple(target_size):
                raise ValueError("t=0 lift expects the clean HR spatial size")
            return y
        if mode not in ("adjoint", "normalized_adjoint"):
            raise ValueError(
                "deformation-aware Stage-2 lift must be adjoint or normalized_adjoint; "
                "inverse/interpolation warps are intentionally unsupported"
            )
        numerator = self.adjoint_at(y, t, target_size=target_size)
        if mode == "adjoint":
            return numerator
        denom = self._normalizer(
            t,
            target_size=target_size,
            batch=y.shape[0],
            device=y.device,
            dtype=y.dtype,
        )
        return numerator / denom

    def state_at(
        self,
        x: torch.Tensor,
        t: int,
        *,
        lift_mode: Optional[str] = None,
    ) -> torch.Tensor:
        t = self._validate_t(t)
        if t == 0:
            return x
        target_size = tuple(x.shape[-2:])
        observation = self.acquire_at(x, t)
        return self.lift_at(
            observation,
            t,
            target_size=target_size,
            lift_mode=lift_mode,
        )

    def terminal_observation(self, x: torch.Tensor) -> torch.Tensor:
        return self.acquire_at(x, self.total_steps)

    def terminal_state(
        self,
        y_terminal: torch.Tensor,
        *,
        target_size: Tuple[int, int],
        lift_mode: Optional[str] = None,
    ) -> torch.Tensor:
        return self.lift_at(
            y_terminal,
            self.total_steps,
            target_size=target_size,
            lift_mode=lift_mode,
        )

    def reverse_update(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        t: int,
        *,
        lift_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """x_(t-1)=x_t+A~_(t-1,phi)(x0_hat)-A~_(t,phi)(x0_hat)."""
        t = self._validate_t(t)
        if t == 0:
            return x_t
        if tuple(x_t.shape) != tuple(x0_hat.shape):
            raise ValueError("x_t and x0_hat must have identical BxCxHxW shape")
        previous = self.state_at(x0_hat, t - 1, lift_mode=lift_mode)
        current = self.state_at(x0_hat, t, lift_mode=lift_mode)
        return x_t + previous - current

    def assert_terminal_closure(
        self,
        x: torch.Tensor,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        direct = self.terminal_observation(x)
        progressive = self.acquire_at(x, self.total_steps)
        if not torch.allclose(direct, progressive, atol=atol, rtol=rtol):
            max_error = (direct - progressive).abs().max().item()
            raise AssertionError(
                f"Deformation-aware terminal closure failed: max_abs_error={max_error:.6e}"
            )
