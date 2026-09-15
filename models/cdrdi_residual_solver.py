"""Learned physical-residual deformation solver for CDRDI Stage-1.

The solver never receives deformation/flow ground truth.  It only sees the
fixed-degradation common-observation target Z_H and the current prediction
Z_M(phi)=P0 W_phi(Y_M).  A single shared update network predicts geometry
increments from the current physical residual, image gradients, and geometry
state.  Calling the same network once gives the one-shot baseline; unrolling
it K times gives the recursive solver with exactly the same trainable
parameters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdrdi_geometry import (
    cubic_bspline_field,
    forward_warp,
    sampling_coordinates,
    zero_mean_control,
)


def _group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


def image_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple differentiable first differences with the original shape."""
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    return gx, gy


class PhysicalResidualUpdateNet(nn.Module):
    """Predict one rigid + B-spline control increment from physical residuals."""

    def __init__(
        self,
        n_msi_bands: int,
        *,
        base_channels: int = 32,
        control_grid: int = 5,
        max_translation: float = 4.0,
        max_rotation_deg: float = 2.0,
        max_local_px: float = 4.0,
    ):
        super().__init__()
        self.n_msi_bands = int(n_msi_bands)
        self.control_grid = int(control_grid)
        self.max_translation = float(max_translation)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_local_px = float(max_local_px)

        # [target, current, residual, grad_x, grad_y] = 5*M channels,
        # plus current effective flow (2) and normalized rigid state (3).
        in_channels = 5 * self.n_msi_bands + 5
        c = int(base_channels)
        groups = _group_count(c)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            ResidualBlock(c),
            ResidualBlock(c),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
        )
        self.rigid_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c),
            nn.SiLU(inplace=True),
            nn.Linear(c, 3),
        )
        self.local_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 2, 3, padding=1),
        )

        # Start from conservative geometry updates.  The final layers are small,
        # not zero, so the unsupervised closure loss can immediately propagate.
        nn.init.normal_(self.rigid_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.rigid_head[-1].bias)
        nn.init.normal_(self.local_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.local_head[-1].bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(features)
        rigid_raw = self.rigid_head(feat)
        local_raw = self.local_head(feat)
        local_raw = F.adaptive_avg_pool2d(
            local_raw, (self.control_grid, self.control_grid)
        )

        rigid_scale = rigid_raw.new_tensor(
            [self.max_translation, self.max_translation, self.max_rotation_deg]
        )
        delta_rigid = torch.tanh(rigid_raw) * rigid_scale.view(1, 3)
        delta_control = torch.tanh(local_raw) * self.max_local_px
        return delta_rigid, delta_control


class LearnedPhysicalResidualSolver(nn.Module):
    """Shared-weight one-shot/recursive deformation solver.

    The geometry state is (dx,dy,theta,control).  The local control grid is
    converted to a dense smooth field by cubic B-spline interpolation.  The
    observation is always synthesized forward through W_phi and fixed P0;
    observed LR-HSI is never inverse-warped.
    """

    def __init__(
        self,
        n_msi_bands: int,
        *,
        base_channels: int = 32,
        control_grid: int = 5,
        max_translation: float = 4.0,
        max_rotation_deg: float = 2.0,
        max_local_px: float = 4.0,
    ):
        super().__init__()
        self.control_grid = int(control_grid)
        self.max_translation = float(max_translation)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_local_px = float(max_local_px)
        self.update_net = PhysicalResidualUpdateNet(
            n_msi_bands,
            base_channels=base_channels,
            control_grid=control_grid,
            max_translation=max_translation,
            max_rotation_deg=max_rotation_deg,
            max_local_px=max_local_px,
        )

    def _local_field(
        self, control: torch.Tensor, output_size: Tuple[int, int]
    ) -> torch.Tensor:
        return cubic_bspline_field(zero_mean_control(control), output_size)

    def _bound_control(
        self, control: torch.Tensor, output_size: Tuple[int, int]
    ) -> torch.Tensor:
        control = zero_mean_control(control)
        if self.max_local_px <= 0.0:
            return torch.zeros_like(control)
        dense = self._local_field(control, output_size)
        amplitude = torch.linalg.vector_norm(dense, dim=1).amax(dim=(-2, -1), keepdim=True)
        amplitude = amplitude.unsqueeze(1).clamp_min(1e-6)
        ratio = (self.max_local_px / amplitude).clamp(max=1.0)
        return control * ratio

    def _measurement(
        self,
        hr_msi: torch.Tensor,
        spatial_operator,
        rigid: torch.Tensor,
        control: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = hr_msi.shape[-2:]
        local = self._local_field(control, (h, w))
        sample_x, sample_y = sampling_coordinates(
            h,
            w,
            rigid[:, 0],
            rigid[:, 1],
            rigid[:, 2],
            local,
        )
        prediction = spatial_operator.degrade(
            forward_warp(
                hr_msi,
                rigid[:, 0],
                rigid[:, 1],
                rigid[:, 2],
                local,
            )
        )
        return prediction, local, sample_x, sample_y

    def _state_features(
        self,
        target: torch.Tensor,
        current: torch.Tensor,
        rigid: torch.Tensor,
        sample_x: torch.Tensor,
        sample_y: torch.Tensor,
        hr_size: Tuple[int, int],
    ) -> torch.Tensor:
        residual = target - current
        gx, gy = image_gradients(current)
        h, w = hr_size
        yy, xx = torch.meshgrid(
            torch.arange(h, device=target.device, dtype=target.dtype),
            torch.arange(w, device=target.device, dtype=target.dtype),
            indexing="ij",
        )
        flow_hr = torch.stack(
            [sample_x - xx.unsqueeze(0), sample_y - yy.unsqueeze(0)], dim=1
        )
        flow_lr = F.interpolate(
            flow_hr,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        rotation_radius = 0.5 * math.sqrt(float(h * h + w * w)) * math.sin(
            math.radians(max(self.max_rotation_deg, 0.0))
        )
        flow_scale = max(
            self.max_translation + self.max_local_px + rotation_radius,
            1.0,
        )
        flow_lr = flow_lr / float(flow_scale)

        tx_scale = max(self.max_translation, 1e-6)
        rot_scale = max(self.max_rotation_deg, 1e-6)
        rigid_norm = torch.stack(
            [
                rigid[:, 0] / tx_scale,
                rigid[:, 1] / tx_scale,
                rigid[:, 2] / rot_scale,
            ],
            dim=1,
        )
        rigid_map = rigid_norm[:, :, None, None].expand(
            -1, -1, target.shape[-2], target.shape[-1]
        )
        return torch.cat(
            [target, current, residual, gx, gy, flow_lr, rigid_map], dim=1
        )

    def _update_state(
        self,
        rigid: torch.Tensor,
        control: torch.Tensor,
        delta_rigid: torch.Tensor,
        delta_control: torch.Tensor,
        hr_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rigid = rigid + delta_rigid
        rigid = torch.stack(
            [
                rigid[:, 0].clamp(-self.max_translation, self.max_translation),
                rigid[:, 1].clamp(-self.max_translation, self.max_translation),
                rigid[:, 2].clamp(-self.max_rotation_deg, self.max_rotation_deg),
            ],
            dim=1,
        )
        control = self._bound_control(control + delta_control, hr_size)
        return rigid, control

    def forward(
        self,
        target_lr_msi: torch.Tensor,
        hr_msi: torch.Tensor,
        spatial_operator,
        *,
        steps: int = 1,
    ) -> Dict[str, object]:
        if steps < 1:
            raise ValueError("steps must be >=1")
        if target_lr_msi.ndim != 4 or hr_msi.ndim != 4:
            raise ValueError("target_lr_msi and hr_msi must be BxCxHxW")
        if target_lr_msi.shape[0] != hr_msi.shape[0]:
            raise ValueError("target/hr_msi batch mismatch")
        if target_lr_msi.shape[1] != hr_msi.shape[1]:
            raise ValueError("target/hr_msi MSI-band mismatch")

        batch = hr_msi.shape[0]
        h, w = hr_msi.shape[-2:]
        rigid = hr_msi.new_zeros((batch, 3))
        control = hr_msi.new_zeros((batch, 2, self.control_grid, self.control_grid))

        current, local, sample_x, sample_y = self._measurement(
            hr_msi, spatial_operator, rigid, control
        )
        initial_prediction = current
        predictions: List[torch.Tensor] = []
        local_fields: List[torch.Tensor] = []
        rigid_states: List[torch.Tensor] = []
        control_states: List[torch.Tensor] = []
        sampling_x_states: List[torch.Tensor] = []
        sampling_y_states: List[torch.Tensor] = []

        for _ in range(int(steps)):
            features = self._state_features(
                target_lr_msi,
                current,
                rigid,
                sample_x,
                sample_y,
                (h, w),
            )
            delta_rigid, delta_control = self.update_net(features)
            rigid, control = self._update_state(
                rigid, control, delta_rigid, delta_control, (h, w)
            )
            current, local, sample_x, sample_y = self._measurement(
                hr_msi, spatial_operator, rigid, control
            )
            predictions.append(current)
            local_fields.append(local)
            rigid_states.append(rigid)
            control_states.append(control)
            sampling_x_states.append(sample_x)
            sampling_y_states.append(sample_y)

        return {
            "initial_prediction": initial_prediction,
            "predictions": predictions,
            "local_fields": local_fields,
            "rigid_states": rigid_states,
            "control_states": control_states,
            "sampling_x": sampling_x_states,
            "sampling_y": sampling_y_states,
            "final_prediction": predictions[-1],
            "final_local_field": local_fields[-1],
            "final_rigid": rigid_states[-1],
            "final_control": control_states[-1],
        }
