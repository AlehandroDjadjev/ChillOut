from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class AbstractRewardModel(nn.Module):
    """Base reward interface.

    The interface follows the user's requested first form:
        reward_fn(original_mask, generated_mask, feature_vector, target_temperature, property_maps)

    Return:
        reward: [B,1], higher is better
        info: dict of tensors for logging
    """

    def forward(
        self,
        original_mask: torch.Tensor,
        generated_mask: torch.Tensor,
        feature_vector: torch.Tensor,
        target_temperature: torch.Tensor,
        property_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class DummyMaxReward(AbstractRewardModel):
    """Temporary reward. It returns max score for pipeline smoke tests.

    Because this is constant, PPO will not learn meaningful behavior. Use it to verify
    data loading, action rasterization, checkpointing, and evaluation. Once the
    ConvLSTM/world model exists, replace this class with WorldModelReward below.
    """

    def __init__(self, max_score: float = 1.0, optional_budget_penalty: float = 0.0) -> None:
        super().__init__()
        self.max_score = float(max_score)
        self.optional_budget_penalty = float(optional_budget_penalty)

    def forward(self, original_mask, generated_mask, feature_vector, target_temperature, property_maps):
        b = original_mask.shape[0]
        reward = torch.full((b, 1), self.max_score, device=original_mask.device, dtype=original_mask.dtype)
        if self.optional_budget_penalty > 0:
            changed = (generated_mask - original_mask).abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
            prop_budget = property_maps.abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
            reward = reward - self.optional_budget_penalty * (changed + prop_budget)
        return reward, {"dummy_reward": reward.detach()}


class CallableReward(AbstractRewardModel):
    """Wrap any Python callable with the required reward signature."""

    def __init__(self, fn: Callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, original_mask, generated_mask, feature_vector, target_temperature, property_maps):
        out = self.fn(original_mask, generated_mask, feature_vector, target_temperature, property_maps)
        if isinstance(out, tuple):
            return out
        return out, {}


class WorldModelReward(AbstractRewardModel):
    """Adapter for the future ConvLSTM/world model.

    The future world model can accept a single combined tensor or separate tensors.
    Here we build a combined tensor:
      concat(original_mask, generated_mask, property_maps, feature_planes, target_plane)

    Expected world_model output options:
      - dict with key 'predicted_final_temp' or 'mean_delta'
      - tensor [B,1] interpreted as predicted final temperature
    """

    def __init__(
        self,
        world_model: nn.Module,
        feature_mean: Optional[torch.Tensor] = None,
        feature_std: Optional[torch.Tensor] = None,
        lambda_budget: float = 0.02,
        lambda_uncertainty: float = 0.05,
    ) -> None:
        super().__init__()
        self.world_model = world_model.eval()
        for p in self.world_model.parameters():
            p.requires_grad_(False)
        self.lambda_budget = float(lambda_budget)
        self.lambda_uncertainty = float(lambda_uncertainty)

    def forward(self, original_mask, generated_mask, feature_vector, target_temperature, property_maps):
        b, _, h, w = original_mask.shape
        feature_planes = feature_vector[:, :, None, None].expand(-1, -1, h, w)
        # Raw target temp is passed as a plane. You may normalize it before this adapter if needed.
        target_plane = target_temperature[:, :, None, None].expand(-1, -1, h, w)
        combined = torch.cat([original_mask, generated_mask, property_maps, feature_planes, target_plane], dim=1)

        with torch.no_grad():
            pred = self.world_model(combined)

        uncertainty = torch.zeros((b, 1), device=original_mask.device, dtype=original_mask.dtype)
        if isinstance(pred, dict):
            if "predicted_final_temp" in pred:
                final_temp = pred["predicted_final_temp"]
            elif "mean_delta" in pred:
                # If model predicts delta, subtract it from current scalar temperature proxy.
                # Replace this with your exact semantics once the ConvLSTM is trained.
                final_temp = target_temperature + pred["mean_delta"]
            else:
                raise KeyError("world_model dict must contain predicted_final_temp or mean_delta")
            if "uncertainty" in pred:
                uncertainty = pred["uncertainty"].view(b, 1)
        else:
            final_temp = pred.view(b, 1)

        temp_error = (final_temp - target_temperature).abs()
        budget = (generated_mask - original_mask).abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
        prop_budget = property_maps.abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
        reward = -temp_error - self.lambda_budget * (budget + prop_budget) - self.lambda_uncertainty * uncertainty
        return reward, {
            "predicted_final_temp": final_temp.detach(),
            "temp_error": temp_error.detach(),
            "budget": budget.detach(),
            "uncertainty": uncertainty.detach(),
        }
