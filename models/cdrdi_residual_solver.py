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
    jacobian_determinant,
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
        initial_dx_px: float = 0.0,
        initial_dy_px: float = 0.0,
    ):
        super().__init__()
        self.initial_dx_px = float(initial_dx_px)
        self.initial_dy_px = float(initial_dy_px)
        if (abs(self.initial_dx_px) > float(max_translation)
            or abs(self.initial_dy_px) > float(max_translation)):
            raise ValueError("Initial rigid translation exceeds the configured translation cap")
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

    @staticmethod
    def _observable_closure_per_sample(
        target: torch.Tensor,
        predicted: torch.Tensor,
        mask: torch.Tensor,
        window: int,
    ) -> torch.Tensor:
        """Match Wald Real-C's locally standardized masked Charbonnier metric.

        Return one score for each batch element; never mix tile decisions.
        Only used at no-grad evaluation/inference time.
        """
        if window < 1 or window % 2 == 0:
            raise ValueError("acceptance_window must be a positive odd integer")
        if target.shape != predicted.shape:
            raise ValueError("Wald target/prediction shape mismatch")
        if mask.ndim != 4 or mask.shape[0] != target.shape[0] or mask.shape[-2:] != target.shape[-2:]:
            raise ValueError("acceptance_mask must be Bx1xLRHxLRW or BxCxLRHxLRW")
        if mask.shape[1] not in (1, target.shape[1]):
            raise ValueError("acceptance_mask has an invalid spectral dimension")

        def normalize(x):
            pad = window // 2
            mu = F.avg_pool2d(x, window, stride=1, padding=pad)
            second = F.avg_pool2d(x * x, window, stride=1, padding=pad)
            var = (second - mu * mu).clamp_min(0.0)
            return (x - mu) / torch.sqrt(var + 1e-8)

        error = normalize(target) - normalize(predicted)
        charbonnier = torch.sqrt(error.square() + 1e-6) - 1e-3
        valid = (mask > 0.5).to(charbonnier.dtype).expand_as(charbonnier)
        return (charbonnier * valid).sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3)).clamp_min(1)

    def _backtracked_update(
        self, target, hr_msi, spatial_operator, rigid, control,
        current, local, sample_x, sample_y, delta_rigid, delta_control,
        acceptance_mask, acceptance_window, acceptance_min_jac, acceptance_relative_gain,
    ):
        """Greedy, per-image line search with a no-worsening physical closure guard.

        The same observed LR-MSI target and fixed PSF are used for every
        proposal.  The input MSI is never altered.  A rejected update keeps
        every component of the preceding geometry state, not just its output.
        """
        prior = self._observable_closure_per_sample(
            target, current, acceptance_mask, acceptance_window
        )
        valid = (acceptance_mask > 0.5).reshape(target.shape[0], -1).any(dim=1)
        accepted = torch.zeros_like(prior, dtype=torch.bool)
        selected_rigid, selected_control = rigid, control
        selected_current, selected_local = current, local
        selected_x, selected_y = sample_x, sample_y
        alpha_selected = torch.zeros_like(prior)
        threshold = torch.maximum(
            torch.full_like(prior, 1e-7),
            prior * float(acceptance_relative_gain),
        )
        for alpha in (1.0, 0.5, 0.25, 0.125):
            proposal_rigid, proposal_control = self._update_state(
                rigid, control, alpha * delta_rigid, alpha * delta_control, hr_msi.shape[-2:]
            )
            proposal_current, proposal_local, proposal_x, proposal_y = self._measurement(
                hr_msi, spatial_operator, proposal_rigid, proposal_control
            )
            proposal_loss = self._observable_closure_per_sample(
                target, proposal_current, acceptance_mask, acceptance_window
            )
            jac_min = jacobian_determinant(proposal_local).flatten(1).amin(dim=1)
            good = (
                (~accepted)
                & valid
                & torch.isfinite(proposal_loss)
                & torch.isfinite(jac_min)
                & (jac_min >= float(acceptance_min_jac))
                & (proposal_loss <= prior - threshold)
            )
            selected_rigid = torch.where(good[:, None], proposal_rigid, selected_rigid)
            selected_control = torch.where(good[:, None, None, None], proposal_control, selected_control)
            selected_current = torch.where(good[:, None, None, None], proposal_current, selected_current)
            selected_local = torch.where(good[:, None, None, None], proposal_local, selected_local)
            selected_x = torch.where(good[:, None, None], proposal_x, selected_x)
            selected_y = torch.where(good[:, None, None], proposal_y, selected_y)
            alpha_selected = torch.where(good, alpha_selected.new_full((), alpha), alpha_selected)
            accepted = accepted | good
            if bool(accepted.all().item()):
                break
        return (
            selected_rigid, selected_control, selected_current, selected_local,
            selected_x, selected_y, accepted, alpha_selected
        )

    def forward(
        self,
        target_lr_msi: torch.Tensor,
        hr_msi: torch.Tensor,
        spatial_operator,
        *,
        steps: int = 1,
        update_mode: str = "both",
        update_policy: str = "plain",
        acceptance_mask: torch.Tensor | None = None,
        acceptance_window: int = 5,
        acceptance_min_jac: float = 0.5,
        acceptance_relative_gain: float = 1e-4,
    ) -> Dict[str, object]:
        # Ablation acts on every recursive proposal, BEFORE the next residual
        # is computed.  All modes preserve the calibrated physical seed.
        if update_mode not in ("both", "rigid_only", "local_only", "seed_only"):
            raise ValueError(
                "update_mode must be both, rigid_only, local_only, or seed_only"
            )
        if update_policy not in ("plain", "closure_backtrack"):
            raise ValueError("update_policy must be plain or closure_backtrack")
        if update_policy == "closure_backtrack":
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("closure_backtrack is evaluation-only and requires torch.no_grad()")
            if acceptance_mask is None:
                raise ValueError("closure_backtrack requires an observed LR acceptance_mask")
            if not (0.0 <= acceptance_relative_gain < 1.0):
                raise ValueError("acceptance_relative_gain must lie in [0,1)")
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
        # Identity remains a fixed control for measuring the full improvement.
        # Non-zero physical initialization is a solver state, not a warp of
        # the input MSI or a synthetic flow supervision label.
        if self.initial_dx_px != 0.0 or self.initial_dy_px != 0.0:
            with torch.no_grad():
                unaligned_prediction = spatial_operator.degrade(hr_msi)
            rigid[:, 0] = self.initial_dx_px
            rigid[:, 1] = self.initial_dy_px
        else:
            unaligned_prediction = None
        current, local, sample_x, sample_y = self._measurement(
            hr_msi, spatial_operator, rigid, control
        )
        initial_prediction = current
        if unaligned_prediction is None:
            unaligned_prediction = initial_prediction
        predictions: List[torch.Tensor] = []
        local_fields: List[torch.Tensor] = []
        rigid_states: List[torch.Tensor] = []
        control_states: List[torch.Tensor] = []
        sampling_x_states: List[torch.Tensor] = []
        sampling_y_states: List[torch.Tensor] = []
        accepted_steps: List[torch.Tensor] = []
        accepted_alphas: List[torch.Tensor] = []

        for _ in range(int(steps)):
            features = self._state_features(
                target_lr_msi,
                current,
                rigid,
                sample_x,
                sample_y,
                (h, w),
            )
            if update_mode == "seed_only":
                # Skip learned updates entirely; the initial -0.5 px state
                # is propagated unchanged as an explicit fixed-seed control.
                delta_rigid = torch.zeros_like(rigid)
                delta_control = torch.zeros_like(control)
            else:
                delta_rigid, delta_control = self.update_net(features)
                if update_mode == "rigid_only":
                    delta_control = torch.zeros_like(delta_control)
                elif update_mode == "local_only":
                    delta_rigid = torch.zeros_like(delta_rigid)
            if update_policy == "closure_backtrack":
                (
                    rigid, control, current, local, sample_x, sample_y,
                    accepted, accepted_alpha
                ) = self._backtracked_update(
                    target_lr_msi, hr_msi, spatial_operator,
                    rigid, control, current, local, sample_x, sample_y,
                    delta_rigid, delta_control,
                    acceptance_mask, acceptance_window,
                    acceptance_min_jac, acceptance_relative_gain,
                )
            else:
                rigid, control = self._update_state(
                    rigid, control, delta_rigid, delta_control, (h, w)
                )
                current, local, sample_x, sample_y = self._measurement(
                    hr_msi, spatial_operator, rigid, control
                )
                accepted = current.new_ones((batch,), dtype=torch.bool)
                accepted_alpha = current.new_ones((batch,))
            accepted_steps.append(accepted)
            accepted_alphas.append(accepted_alpha)
            predictions.append(current)
            local_fields.append(local)
            rigid_states.append(rigid)
            control_states.append(control)
            sampling_x_states.append(sample_x)
            sampling_y_states.append(sample_y)

        return {
            "unaligned_prediction": unaligned_prediction,
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
            "accepted_steps": accepted_steps,
            "accepted_alphas": accepted_alphas,
        }
