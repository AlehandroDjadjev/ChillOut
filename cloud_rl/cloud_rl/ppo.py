from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class RolloutBatch:
    obs_map: torch.Tensor
    features: torch.Tensor
    target_temp_norm: torch.Tensor
    original_mask: torch.Tensor
    raw_features: torch.Tensor
    target_temp: torch.Tensor
    op: torch.Tensor
    params: torch.Tensor
    old_log_prob: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def normalize_advantages(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def ppo_update(
    model,
    batch: RolloutBatch,
    optimizer,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    bc_coef: float = 0.0,
    grad_clip: float = 0.5,
) -> Dict[str, float]:
    log_prob, entropy, value = model.evaluate_actions(
        batch.obs_map, batch.features, batch.target_temp_norm, batch.op, batch.params
    )
    ratio = torch.exp(log_prob - batch.old_log_prob)
    adv = normalize_advantages(batch.advantages)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()
    bc_loss = -log_prob.mean()
    value_loss = F.mse_loss(value, batch.returns)
    entropy_bonus = entropy.mean()
    loss = policy_loss + value_coef * value_loss + bc_coef * bc_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    approx_kl = (batch.old_log_prob - log_prob).mean().detach()
    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "bc_loss": float(bc_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_bonus.detach().cpu()),
        "approx_kl": float(approx_kl.cpu()),
        "mean_return": float(batch.returns.mean().detach().cpu()),
    }
