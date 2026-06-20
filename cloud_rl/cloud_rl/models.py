from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


@dataclass
class PolicyOutput:
    op_logits: torch.Tensor
    param_mu: torch.Tensor
    param_std: torch.Tensor
    value: torch.Tensor


class CloudActorCritic(nn.Module):
    """Hybrid actor-critic policy.

    Observation uses both:
      - obs_map: mask + scalar feature planes + target plane, [B,C,H,W]
      - features and target again through an MLP, [B,14] and [B,1]

    Action uses up to K tokens. Each token has:
      - categorical op: noop/remove/modify/create
      - continuous params: x, y, radius, mass, humidity, thickness, height, type
    """

    def __init__(
        self,
        obs_channels: int = 16,
        feature_dim: int = 14,
        max_actions: int = 3,
        param_dim: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.max_actions = max_actions
        self.param_dim = param_dim

        self.image_encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.scalar_encoder = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(128 + 128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.op_head = nn.Linear(hidden_dim, max_actions * 4)
        self.param_mu_head = nn.Linear(hidden_dim, max_actions * param_dim)
        self.param_log_std = nn.Parameter(torch.full((max_actions, param_dim), -0.7))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_map: torch.Tensor, features: torch.Tensor, target_temp_norm: torch.Tensor) -> PolicyOutput:
        img = self.image_encoder(obs_map)
        scalars = self.scalar_encoder(torch.cat([features, target_temp_norm], dim=-1))
        h = self.trunk(torch.cat([img, scalars], dim=-1))
        b = obs_map.shape[0]
        op_logits = self.op_head(h).view(b, self.max_actions, 4)
        param_mu = self.param_mu_head(h).view(b, self.max_actions, self.param_dim).clamp(-5, 5)
        param_std = self.param_log_std.exp()[None, :, :].expand_as(param_mu)
        value = self.value_head(h)
        return PolicyOutput(op_logits, param_mu, param_std, value)

    def sample(self, obs_map: torch.Tensor, features: torch.Tensor, target_temp_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self(obs_map, features, target_temp_norm)
        op_dist = Categorical(logits=out.op_logits)
        op = op_dist.sample()
        op_log_prob = op_dist.log_prob(op).sum(dim=1, keepdim=False)
        op_entropy = op_dist.entropy().sum(dim=1, keepdim=False)

        param_dist = Normal(out.param_mu, out.param_std)
        raw = param_dist.rsample()
        params = torch.tanh(raw)
        # tanh correction for log prob.
        param_log_prob = param_dist.log_prob(raw) - torch.log(1.0 - params.pow(2) + 1e-6)
        param_log_prob = param_log_prob.sum(dim=(1, 2), keepdim=False)
        param_entropy = param_dist.entropy().sum(dim=(1, 2), keepdim=False)

        return {
            "op": op,
            "params": params,
            "log_prob": (op_log_prob + param_log_prob).unsqueeze(-1),
            "entropy": (op_entropy + param_entropy).unsqueeze(-1),
            "value": out.value,
        }

    def evaluate_actions(
        self,
        obs_map: torch.Tensor,
        features: torch.Tensor,
        target_temp_norm: torch.Tensor,
        op: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self(obs_map, features, target_temp_norm)
        op_dist = Categorical(logits=out.op_logits)
        op_log_prob = op_dist.log_prob(op).sum(dim=1, keepdim=False)
        op_entropy = op_dist.entropy().sum(dim=1, keepdim=False)

        params = params.clamp(-0.999, 0.999)
        raw = torch.atanh(params)
        param_dist = Normal(out.param_mu, out.param_std)
        param_log_prob = param_dist.log_prob(raw) - torch.log(1.0 - params.pow(2) + 1e-6)
        param_log_prob = param_log_prob.sum(dim=(1, 2), keepdim=False)
        param_entropy = param_dist.entropy().sum(dim=(1, 2), keepdim=False)

        return (op_log_prob + param_log_prob).unsqueeze(-1), (op_entropy + param_entropy).unsqueeze(-1), out.value
