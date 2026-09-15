"""Geometry-only verifier for calibrated degradation-locked deformation inversion.

This module deliberately does not depend on any flow ground truth during optimization.
The only optimizable variables are rigid geometry and smooth control-point residuals;
PSF/MTF and SRF stay fixed outside this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SyntheticGeometry:
    dx: torch.Tensor
    dy: torch.Tensor
    theta_deg: torch.Tensor
    control: torch.Tensor
    local_field: torch.Tensor


@dataclass
class GeometrySolveResult:
    dx: float
    dy: float
    theta_deg: float
    closure_initial: float
    closure_final: float
    local_field: torch.Tensor
    sampling_x: torch.Tensor
    sampling_y: torch.Tensor
    stage_losses: Dict[str, float]


def spectral_project(x: torch.Tensor, srf_weights: torch.Tensor) -> torch.Tensor:
    """Apply fixed SRF weights R0 to BxCxHxW data."""
    if x.ndim != 4 or srf_weights.ndim != 2:
        raise ValueError("x must be BxCxHxW and srf_weights must be MxC")
    if x.shape[1] != srf_weights.shape[1]:
        raise ValueError("spectral band mismatch between x and srf_weights")
    weights = srf_weights.to(device=x.device, dtype=x.dtype)
    return torch.einsum("mc,bchw->bmhw", weights, x)


def _bspline_basis_matrix(
    n_ctrl: int,
    n_out: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_ctrl < 2 or n_out < 1:
        raise ValueError("n_ctrl must be >=2 and n_out must be >=1")
    u = torch.linspace(0.0, float(n_ctrl - 1), n_out, device=device, dtype=dtype)
    base = torch.floor(u).to(torch.long)
    t = u - base.to(dtype)
    weights = torch.stack(
        [
            (1.0 - t).pow(3) / 6.0,
            (3.0 * t.pow(3) - 6.0 * t.pow(2) + 4.0) / 6.0,
            (-3.0 * t.pow(3) + 3.0 * t.pow(2) + 3.0 * t + 1.0) / 6.0,
            t.pow(3) / 6.0,
        ],
        dim=1,
    )
    indices = torch.stack([base - 1, base, base + 1, base + 2], dim=1)
    indices = indices.clamp(0, n_ctrl - 1)
    matrix = torch.zeros((n_out, n_ctrl), device=device, dtype=dtype)
    matrix.scatter_add_(1, indices, weights)
    return matrix


def cubic_bspline_field(control: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    """Interpolate Bx2xKyxKx sparse control offsets to a dense HR-pixel field."""
    if control.ndim != 4 or control.shape[1] != 2:
        raise ValueError("control must have shape Bx2xKyxKx")
    h, w = map(int, output_size)
    ky, kx = control.shape[-2:]
    wy = _bspline_basis_matrix(ky, h, device=control.device, dtype=control.dtype)
    wx = _bspline_basis_matrix(kx, w, device=control.device, dtype=control.dtype)
    return torch.einsum("hi,bcij,wj->bchw", wy, control, wx)


def zero_mean_control(control: torch.Tensor) -> torch.Tensor:
    """Remove the global-translation gauge from local deformation controls."""
    return control - control.mean(dim=(-2, -1), keepdim=True)


def sampling_coordinates(
    height: int,
    width: int,
    dx: torch.Tensor,
    dy: torch.Tensor,
    theta_deg: torch.Tensor,
    local_field: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose local deformation then rigid motion in HR pixel coordinates.

    The returned coordinates describe the forward acquisition sampler used by
    grid_sample: output location p reads the reliable MSI-coordinate image at phi(p).
    """
    if local_field.ndim != 4 or local_field.shape[1] != 2:
        raise ValueError("local_field must have shape Bx2xHxW")
    if local_field.shape[-2:] != (height, width):
        raise ValueError("local_field spatial size mismatch")
    batch = local_field.shape[0]
    for value, name in ((dx, "dx"), (dy, "dy"), (theta_deg, "theta_deg")):
        if value.ndim != 1 or value.shape[0] != batch:
            raise ValueError(f"{name} must have shape [B]")

    yy, xx = torch.meshgrid(
        torch.arange(height, device=local_field.device, dtype=local_field.dtype),
        torch.arange(width, device=local_field.device, dtype=local_field.dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(batch, -1, -1) + local_field[:, 0]
    yy = yy.unsqueeze(0).expand(batch, -1, -1) + local_field[:, 1]

    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    angle = theta_deg * (math.pi / 180.0)
    cos_a = torch.cos(angle).view(batch, 1, 1)
    sin_a = torch.sin(angle).view(batch, 1, 1)
    x0, y0 = xx - cx, yy - cy
    sample_x = cos_a * x0 - sin_a * y0 + cx + dx.view(batch, 1, 1)
    sample_y = sin_a * x0 + cos_a * y0 + cy + dy.view(batch, 1, 1)
    return sample_x, sample_y


def forward_warp(
    image: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    theta_deg: torch.Tensor,
    local_field: torch.Tensor,
) -> torch.Tensor:
    """Forward-synthesize an HSI acquisition geometry without inverse-warping observations."""
    if image.ndim != 4:
        raise ValueError("image must have shape BxCxHxW")
    h, w = image.shape[-2:]
    sample_x, sample_y = sampling_coordinates(h, w, dx, dy, theta_deg, local_field)
    grid_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def q_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    factor = int(factor)
    if factor <= 1:
        return x
    h, w = x.shape[-2:]
    if h % factor == 0 and w % factor == 0:
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)
    return F.interpolate(
        x,
        size=(max(1, h // factor), max(1, w // factor)),
        mode="area",
    )


def charbonnier_mean(residual: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return (torch.sqrt(residual * residual + eps * eps) - eps).mean()


def multiscale_closure_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    factors: Iterable[int],
) -> torch.Tensor:
    factors = tuple(int(v) for v in factors)
    if not factors:
        raise ValueError("at least one pyramid factor is required")
    loss = prediction.new_zeros(())
    for factor in factors:
        loss = loss + charbonnier_mean(q_downsample(target, factor) - q_downsample(prediction, factor))
    return loss / float(len(factors))


def deformation_regularizer(local_field: torch.Tensor) -> torch.Tensor:
    dx = local_field[:, :, :, 1:] - local_field[:, :, :, :-1]
    dy = local_field[:, :, 1:, :] - local_field[:, :, :-1, :]
    grad = dx.abs().mean() + dy.abs().mean()
    lap_x = local_field[:, :, :, 2:] - 2.0 * local_field[:, :, :, 1:-1] + local_field[:, :, :, :-2]
    lap_y = local_field[:, :, 2:, :] - 2.0 * local_field[:, :, 1:-1, :] + local_field[:, :, :-2, :]
    bend = lap_x.abs().mean() + lap_y.abs().mean()
    return grad + 0.2 * bend


def jacobian_determinant(local_field: torch.Tensor) -> torch.Tensor:
    """Jacobian determinant of p -> p + v(p), evaluated with forward differences."""
    vx, vy = local_field[:, 0], local_field[:, 1]
    dvx_dx = vx[:, :-1, 1:] - vx[:, :-1, :-1]
    dvx_dy = vx[:, 1:, :-1] - vx[:, :-1, :-1]
    dvy_dx = vy[:, :-1, 1:] - vy[:, :-1, :-1]
    dvy_dy = vy[:, 1:, :-1] - vy[:, :-1, :-1]
    return (1.0 + dvx_dx) * (1.0 + dvy_dy) - dvx_dy * dvy_dx


def sample_synthetic_geometry(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_translation: float = 4.0,
    max_rotation_deg: float = 2.0,
    max_local_px: float = 2.0,
    control_grid: int = 5,
    min_jacobian: float = 0.5,
) -> SyntheticGeometry:
    """Create rigid + smooth B-spline deformation; GT is for synthesis/metrics only."""
    uniform = lambda lo, hi: torch.empty(1, device=device, dtype=dtype).uniform_(
        lo, hi, generator=generator
    )
    dx = uniform(-max_translation, max_translation)
    dy = uniform(-max_translation, max_translation)
    theta = uniform(-max_rotation_deg, max_rotation_deg)

    control = None
    local = None
    for _ in range(64):
        candidate = torch.randn(
            (1, 2, control_grid, control_grid),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        candidate = zero_mean_control(candidate)
        dense = cubic_bspline_field(candidate, (height, width))
        dense_norm = torch.linalg.vector_norm(dense, dim=1).amax().clamp_min(1e-8)
        strength = uniform(0.65, 1.0) * float(max_local_px)
        candidate = candidate * (strength / dense_norm)
        dense = cubic_bspline_field(candidate, (height, width))
        if float(jacobian_determinant(dense).amin().item()) >= float(min_jacobian):
            control, local = candidate, dense
            break
    if control is None or local is None:
        raise RuntimeError("failed to sample a non-folding local deformation")
    return SyntheticGeometry(dx=dx, dy=dy, theta_deg=theta, control=control, local_field=local)


class RecursiveGeometryState(nn.Module):
    """Rigid state plus coarse/fine residual control-point fields."""

    def __init__(
        self,
        *,
        coarse_grid: int = 3,
        fine_grid: int = 5,
        max_translation: float = 4.0,
        max_rotation_deg: float = 2.0,
        max_local_px: float = 2.0,
    ):
        super().__init__()
        self.rigid = nn.Parameter(torch.zeros(3))
        self.control_coarse = nn.Parameter(torch.zeros(1, 2, coarse_grid, coarse_grid))
        self.control_fine = nn.Parameter(torch.zeros(1, 2, fine_grid, fine_grid))
        self.max_translation = float(max_translation)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_local_px = float(max_local_px)

    def rigid_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.rigid[0:1], self.rigid[1:2], self.rigid[2:3]

    def local_field(
        self,
        output_size: Tuple[int, int],
        *,
        use_coarse: bool,
        use_fine: bool,
    ) -> torch.Tensor:
        h, w = output_size
        field = self.rigid.new_zeros((1, 2, h, w))
        if use_coarse:
            field = field + cubic_bspline_field(zero_mean_control(self.control_coarse), (h, w))
        if use_fine:
            field = field + cubic_bspline_field(zero_mean_control(self.control_fine), (h, w))
        return field

    @torch.no_grad()
    def clamp_(self) -> None:
        self.rigid[0:2].clamp_(-self.max_translation, self.max_translation)
        self.rigid[2].clamp_(-self.max_rotation_deg, self.max_rotation_deg)
        self.control_coarse.clamp_(-self.max_local_px, self.max_local_px)
        self.control_fine.clamp_(-self.max_local_px, self.max_local_px)


def _stage_optimize(
    state: RecursiveGeometryState,
    *,
    hr_msi: torch.Tensor,
    target_lr_msi: torch.Tensor,
    spatial_operator,
    factors: Sequence[int],
    parameters: Sequence[nn.Parameter],
    iterations: int,
    lr: float,
    lambda_def: float,
    use_coarse: bool,
    use_fine: bool,
) -> float:
    optimizer = torch.optim.Adam(parameters, lr=float(lr))
    h, w = hr_msi.shape[-2:]
    final_loss = float("nan")
    for _ in range(int(iterations)):
        local = state.local_field((h, w), use_coarse=use_coarse, use_fine=use_fine)
        dx, dy, theta = state.rigid_tensors()
        prediction = spatial_operator.degrade(forward_warp(hr_msi, dx, dy, theta, local))
        closure = multiscale_closure_loss(target_lr_msi, prediction, factors)
        reg = deformation_regularizer(local) if (use_coarse or use_fine) else closure.new_zeros(())
        loss = closure + float(lambda_def) * reg
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CDRDI geometry loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        state.clamp_()
        final_loss = float(closure.detach().item())
    return final_loss


def solve_geometry_from_closure(
    *,
    hr_msi: torch.Tensor,
    target_lr_msi: torch.Tensor,
    spatial_operator,
    max_translation: float = 4.0,
    max_rotation_deg: float = 2.0,
    max_local_px: float = 2.0,
    lambda_def: float = 1e-3,
    coarse_iterations: int = 300,
    middle_iterations: int = 400,
    fine_iterations: int = 600,
) -> GeometrySolveResult:
    """Estimate phi only from fixed-degradation closure; no flow GT argument exists."""
    if hr_msi.shape[0] != 1 or target_lr_msi.shape[0] != 1:
        raise ValueError("geometry verifier currently expects batch size 1")
    state = RecursiveGeometryState(
        max_translation=max_translation,
        max_rotation_deg=max_rotation_deg,
        max_local_px=max_local_px,
    ).to(device=hr_msi.device, dtype=hr_msi.dtype)
    with torch.no_grad():
        zero_local = torch.zeros((1, 2, *hr_msi.shape[-2:]), device=hr_msi.device, dtype=hr_msi.dtype)
        zero = torch.zeros(1, device=hr_msi.device, dtype=hr_msi.dtype)
        initial_prediction = spatial_operator.degrade(forward_warp(hr_msi, zero, zero, zero, zero_local))
        initial = float(multiscale_closure_loss(target_lr_msi, initial_prediction, (4, 2, 1)).item())

    losses: Dict[str, float] = {}
    losses["rigid"] = _stage_optimize(
        state,
        hr_msi=hr_msi,
        target_lr_msi=target_lr_msi,
        spatial_operator=spatial_operator,
        factors=(4,),
        parameters=(state.rigid,),
        iterations=coarse_iterations,
        lr=0.10,
        lambda_def=0.0,
        use_coarse=False,
        use_fine=False,
    )
    losses["local_coarse"] = _stage_optimize(
        state,
        hr_msi=hr_msi,
        target_lr_msi=target_lr_msi,
        spatial_operator=spatial_operator,
        factors=(4, 2),
        parameters=(state.control_coarse,),
        iterations=middle_iterations,
        lr=0.05,
        lambda_def=lambda_def,
        use_coarse=True,
        use_fine=False,
    )
    losses["local_fine"] = _stage_optimize(
        state,
        hr_msi=hr_msi,
        target_lr_msi=target_lr_msi,
        spatial_operator=spatial_operator,
        factors=(4, 2, 1),
        parameters=(state.control_coarse, state.control_fine),
        iterations=fine_iterations,
        lr=0.03,
        lambda_def=lambda_def,
        use_coarse=True,
        use_fine=True,
    )

    with torch.no_grad():
        local = state.local_field(tuple(hr_msi.shape[-2:]), use_coarse=True, use_fine=True)
        dx, dy, theta = state.rigid_tensors()
        prediction = spatial_operator.degrade(forward_warp(hr_msi, dx, dy, theta, local))
        final = float(multiscale_closure_loss(target_lr_msi, prediction, (4, 2, 1)).item())
        sample_x, sample_y = sampling_coordinates(
            hr_msi.shape[-2], hr_msi.shape[-1], dx, dy, theta, local
        )
    return GeometrySolveResult(
        dx=float(dx.item()),
        dy=float(dy.item()),
        theta_deg=float(theta.item()),
        closure_initial=initial,
        closure_final=final,
        local_field=local.detach(),
        sampling_x=sample_x.detach(),
        sampling_y=sample_y.detach(),
        stage_losses=losses,
    )


def geometry_metrics(
    result: GeometrySolveResult,
    gt: SyntheticGeometry,
    *,
    scale_ratio: int,
) -> Dict[str, float]:
    with torch.no_grad():
        gt_x, gt_y = sampling_coordinates(
            gt.local_field.shape[-2],
            gt.local_field.shape[-1],
            gt.dx,
            gt.dy,
            gt.theta_deg,
            gt.local_field,
        )
        epe_map = torch.sqrt((result.sampling_x - gt_x).pow(2) + (result.sampling_y - gt_y).pow(2))
        local_epe = torch.linalg.vector_norm(result.local_field - gt.local_field, dim=1)
        reduction = 1.0 - result.closure_final / max(result.closure_initial, 1e-12)
        pred_jac = jacobian_determinant(result.local_field)
        gt_jac = jacobian_determinant(gt.local_field)
        return {
            "dx_gt": float(gt.dx.item()),
            "dx_est": result.dx,
            "dx_abs_err": abs(result.dx - float(gt.dx.item())),
            "dy_gt": float(gt.dy.item()),
            "dy_est": result.dy,
            "dy_abs_err": abs(result.dy - float(gt.dy.item())),
            "theta_gt": float(gt.theta_deg.item()),
            "theta_est": result.theta_deg,
            "theta_abs_err_deg": abs(result.theta_deg - float(gt.theta_deg.item())),
            "epe_hr_mean": float(epe_map.mean().item()),
            "epe_hr_p95": float(torch.quantile(epe_map.flatten(), 0.95).item()),
            "epe_lr_mean": float(epe_map.mean().item()) / float(scale_ratio),
            "local_epe_hr_mean": float(local_epe.mean().item()),
            "closure_initial": result.closure_initial,
            "closure_final": result.closure_final,
            "closure_reduction": float(reduction),
            "pred_min_jacobian": float(pred_jac.amin().item()),
            "gt_min_jacobian": float(gt_jac.amin().item()),
        }
