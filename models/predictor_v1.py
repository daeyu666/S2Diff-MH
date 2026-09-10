"""Time-conditioned clean-HSI predictor V1 for Innovation 1."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    upper = min(int(max_groups), int(channels))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        denom = max(half - 1, 1)
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / float(denom)
        )
        angles = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class TimeConditionedResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        groups = _num_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels * 2))
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(time_emb).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.conv2(self.dropout(F.silu(h)))


class CleanHSIPredictor(nn.Module):
    requires_msi = False

    def __init__(
        self,
        n_bands: int,
        total_steps: int = 12,
        base_channels: int = 64,
        time_dim: int = 256,
        dropout: float = 0.0,
        residual_prediction: bool = True,
    ):
        super().__init__()
        self.n_bands = int(n_bands)
        self.total_steps = int(total_steps)
        self.residual_prediction = bool(residual_prediction)
        c1, c2, c3 = int(base_channels), int(base_channels) * 2, int(base_channels) * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.in_proj = nn.Conv2d(self.n_bands, c1, 3, padding=1)
        self.enc1 = nn.ModuleList([TimeConditionedResBlock(c1, time_dim, dropout) for _ in range(2)])
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = nn.ModuleList([TimeConditionedResBlock(c2, time_dim, dropout) for _ in range(2)])
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid = nn.ModuleList([TimeConditionedResBlock(c3, time_dim, dropout) for _ in range(2)])
        self.up2_proj = nn.Conv2d(c3, c2, 3, padding=1)
        self.up2_fuse = nn.Conv2d(c2 * 2, c2, 1)
        self.dec2 = nn.ModuleList([TimeConditionedResBlock(c2, time_dim, dropout) for _ in range(2)])
        self.up1_proj = nn.Conv2d(c2, c1, 3, padding=1)
        self.up1_fuse = nn.Conv2d(c1 * 2, c1, 1)
        self.dec1 = nn.ModuleList([TimeConditionedResBlock(c1, time_dim, dropout) for _ in range(2)])
        self.out_norm = nn.GroupNorm(_num_groups(c1), c1)
        self.out_conv = nn.Conv2d(c1, self.n_bands, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _time_embedding(self, t, batch_size: int):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.out_conv.weight.device)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        return self.time_embed(t.float() / float(self.total_steps) * 1000.0)

    @staticmethod
    def _apply_blocks(x, blocks, time_emb):
        for block in blocks:
            x = block(x, time_emb)
        return x

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self._time_embedding(t, x_t.shape[0])
        input_state = x_t
        x = self.in_proj(x_t)
        skip1 = self._apply_blocks(x, self.enc1, time_emb)
        x = self.down1(skip1)
        skip2 = self._apply_blocks(x, self.enc2, time_emb)
        x = self.down2(skip2)
        x = self._apply_blocks(x, self.mid, time_emb)
        x = F.interpolate(x, size=skip2.shape[-2:], mode="nearest")
        x = self.up2_fuse(torch.cat([self.up2_proj(x), skip2], dim=1))
        x = self._apply_blocks(x, self.dec2, time_emb)
        x = F.interpolate(x, size=skip1.shape[-2:], mode="nearest")
        x = self.up1_fuse(torch.cat([self.up1_proj(x), skip1], dim=1))
        x = self._apply_blocks(x, self.dec1, time_emb)
        residual = self.out_conv(F.silu(self.out_norm(x)))
        return input_state + residual if self.residual_prediction else residual
