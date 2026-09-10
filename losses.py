import torch
import torch.nn as nn


class SAMLoss(nn.Module):
    """Numerically stable spectral-angle loss for BxCxHxW HSI tensors."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        pred_norm = torch.linalg.vector_norm(pred, ord=2, dim=1, keepdim=True).clamp_min(self.eps)
        target_norm = torch.linalg.vector_norm(target, ord=2, dim=1, keepdim=True).clamp_min(self.eps)
        pred_unit = pred / pred_norm
        target_unit = target / target_norm
        chord = torch.linalg.vector_norm(pred_unit - target_unit, ord=2, dim=1)
        anti_chord = torch.linalg.vector_norm(pred_unit + target_unit, ord=2, dim=1).clamp_min(self.eps)
        return (2.0 * torch.atan2(chord, anti_chord)).mean()
