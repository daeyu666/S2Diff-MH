"""Clean Raw-MSI direct fusion backbone used after Innovation 1.

This file intentionally contains no MSI high-pass branch, no transfer gate,
no time-varying MSI schedule, and no registration/alignment code.  It keeps the
useful parameter names from the legacy V3 Raw-Direct ablation so trained
weights can be reused with strict=False while obsolete gate keys are ignored.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor_v2 import (
    EMRInspiredTimeBlock,
    LocalSpectralStem,
    SinusoidalTimeEmbedding,
    _num_groups,
)


class RawMSIDirectPredictor(nn.Module):
    """F_theta(x_t, HR-MSI, t) -> clean HR-HSI with full Raw MSI injection."""

    requires_msi = True

    def __init__(
        self,
        n_bands: int,
        n_msi_bands: int,
        total_steps: int = 12,
        base_channels: int = 64,
        time_dim: int = 256,
        dropout: float = 0.0,
        residual_prediction: bool = True,
        spectral_hidden: int = 8,
    ):
        super().__init__()
        if n_bands < 1 or n_msi_bands < 1:
            raise ValueError("n_bands and n_msi_bands must be >= 1")
        self.n_bands = int(n_bands)
        self.n_msi_bands = int(n_msi_bands)
        self.total_steps = int(total_steps)
        self.residual_prediction = bool(residual_prediction)
        c1, c2, c3 = int(base_channels), int(base_channels) * 2, int(base_channels) * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.spectral_stem = LocalSpectralStem(hidden_channels=spectral_hidden)
        self.in_proj = nn.Conv2d(self.n_bands, c1, kernel_size=3, padding=1)

        self.msi_in = nn.Sequential(
            nn.Conv2d(self.n_msi_bands, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
        )
        self.msi_down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
        )
        self.msi_down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
        )

        self.enc1 = nn.ModuleList([
            EMRInspiredTimeBlock(c1, time_dim, dropout),
            EMRInspiredTimeBlock(c1, time_dim, dropout),
        ])
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.ModuleList([
            EMRInspiredTimeBlock(c2, time_dim, dropout),
            EMRInspiredTimeBlock(c2, time_dim, dropout),
        ])
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.mid = nn.ModuleList([
            EMRInspiredTimeBlock(c3, time_dim, dropout),
            EMRInspiredTimeBlock(c3, time_dim, dropout),
        ])
        self.up2_proj = nn.Conv2d(c3, c2, kernel_size=3, padding=1)
        self.up2_fuse = nn.Conv2d(c2 + c2, c2, kernel_size=1)
        self.dec2 = nn.ModuleList([
            EMRInspiredTimeBlock(c2, time_dim, dropout),
            EMRInspiredTimeBlock(c2, time_dim, dropout),
        ])
        self.up1_proj = nn.Conv2d(c2, c1, kernel_size=3, padding=1)
        self.up1_fuse = nn.Conv2d(c1 + c1, c1, kernel_size=1)
        self.dec1 = nn.ModuleList([
            EMRInspiredTimeBlock(c1, time_dim, dropout),
            EMRInspiredTimeBlock(c1, time_dim, dropout),
        ])
        self.out_norm = nn.GroupNorm(_num_groups(c1), c1)
        self.out_conv = nn.Conv2d(c1, self.n_bands, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _time_embedding(self, t: torch.Tensor, batch_size: int, device) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(f"t must be scalar or shape [B={batch_size}], got {tuple(t.shape)}")
        return self.time_embed(t.float() / float(self.total_steps) * 1000.0)

    @staticmethod
    def _apply_blocks(x, blocks, time_emb):
        for block in blocks:
            x = block(x, time_emb)
        return x

    def forward(self, x_t: torch.Tensor, hr_msi: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4 or hr_msi.ndim != 4:
            raise ValueError("x_t and hr_msi must both be BxCxHxW")
        if x_t.shape[1] != self.n_bands:
            raise ValueError(f"expected {self.n_bands} HSI bands, got {x_t.shape[1]}")
        if hr_msi.shape[1] != self.n_msi_bands:
            raise ValueError(f"expected {self.n_msi_bands} MSI bands, got {hr_msi.shape[1]}")
        if hr_msi.shape[-2:] != x_t.shape[-2:]:
            raise ValueError("Raw-Direct requires registered HR-MSI and HSI grids of the same spatial size")

        time_emb = self._time_embedding(t, x_t.shape[0], x_t.device)
        input_state = x_t
        m1 = self.msi_in(hr_msi)
        m2 = self.msi_down1(m1)
        m3 = self.msi_down2(m2)

        x = self.in_proj(self.spectral_stem(x_t)) + m1
        skip1 = self._apply_blocks(x, self.enc1, time_emb)
        x = self.down1(skip1) + m2
        skip2 = self._apply_blocks(x, self.enc2, time_emb)
        x = self.down2(skip2) + m3
        x = self._apply_blocks(x, self.mid, time_emb)
        x = F.interpolate(x, size=skip2.shape[-2:], mode="nearest")
        x = self.up2_fuse(torch.cat([self.up2_proj(x), skip2], dim=1))
        x = self._apply_blocks(x, self.dec2, time_emb)
        x = F.interpolate(x, size=skip1.shape[-2:], mode="nearest")
        x = self.up1_fuse(torch.cat([self.up1_proj(x), skip1], dim=1))
        x = self._apply_blocks(x, self.dec1, time_emb)
        residual = self.out_conv(F.silu(self.out_norm(x)))
        return input_state + residual if self.residual_prediction else residual


def extract_legacy_raw_direct_state_dict(checkpoint: Mapping) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Convert an old S2Diff V3/raw_direct checkpoint for this clean model.

    Obsolete high-frequency/gate parameters are intentionally dropped.  All
    backbone, MSI encoder and decoder keys keep their original names.
    """
    state = checkpoint
    if isinstance(checkpoint, Mapping):
        for candidate in ("model", "model_state_dict", "state_dict"):
            if candidate in checkpoint and isinstance(checkpoint[candidate], Mapping):
                state = checkpoint[candidate]
                break
    cleaned: Dict[str, torch.Tensor] = {}
    ignored: Dict[str, str] = {}
    obsolete_prefixes = ("gate1.", "gate2.", "gate3.", "msi_highpass.")
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith(obsolete_prefixes):
            ignored[key] = "legacy HF/gate parameter"
            continue
        cleaned[key] = value
    return cleaned, ignored


def load_legacy_raw_direct_checkpoint(model: RawMSIDirectPredictor, path: str, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    state, ignored = extract_legacy_raw_direct_state_dict(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    return {
        "ignored_legacy_keys": sorted(ignored),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
