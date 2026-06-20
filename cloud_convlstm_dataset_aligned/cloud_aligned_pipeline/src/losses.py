from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSmoothL1Loss(nn.Module):
    """Per-channel SmoothL1 loss.

    pred/target: [B, C, H, W]
    """

    def __init__(self, channel_weights: Sequence[float]):
        super().__init__()
        w = torch.tensor(channel_weights, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")

        loss = F.smooth_l1_loss(pred, target, reduction="none")
        return (loss * self.weights).mean()


@torch.no_grad()
def per_channel_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean(dim=(0, 2, 3))


@torch.no_grad()
def per_channel_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    return torch.sqrt(mse.clamp_min(1e-12))
