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
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    out = (x - x.mean()) / (x.std(unbiased=False) + 1e-8)
    return out.clamp(-8.0, 8.0)


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
    log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=20.0, neginf=-20.0)
    old_log_prob = torch.nan_to_num(batch.old_log_prob, nan=0.0, posinf=20.0, neginf=-20.0)
    returns = torch.nan_to_num(batch.returns.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)
    value = torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

    log_ratio = (log_prob - old_log_prob).clamp(-10.0, 10.0)
    ratio = torch.exp(log_ratio)
    adv = normalize_advantages(batch.advantages)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()
    bc_loss = -log_prob.clamp(-100.0, 100.0).mean()
    value_loss = F.mse_loss(value, returns)
    entropy_bonus = entropy.mean()
    loss = policy_loss + value_coef * value_loss + bc_coef * bc_loss - entropy_coef * entropy_bonus

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        approx_kl = (old_log_prob - log_prob).mean().detach()
        return {
            "loss": float("nan"),
            "policy_loss": float(policy_loss.detach().cpu()) if torch.isfinite(policy_loss) else float("nan"),
            "bc_loss": float(bc_loss.detach().cpu()) if torch.isfinite(bc_loss) else float("nan"),
            "value_loss": float(value_loss.detach().cpu()) if torch.isfinite(value_loss) else float("nan"),
            "entropy": float(entropy_bonus.detach().cpu()) if torch.isfinite(entropy_bonus) else float("nan"),
            "approx_kl": float(approx_kl.cpu()) if torch.isfinite(approx_kl) else float("nan"),
            "mean_return": float(returns.mean().detach().cpu()),
            "skipped_update": 1.0,
        }

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    approx_kl = (old_log_prob - log_prob).mean().detach()
    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "bc_loss": float(bc_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_bonus.detach().cpu()),
        "approx_kl": float(approx_kl.cpu()),
        "mean_return": float(returns.mean().detach().cpu()),
        "skipped_update": 0.0,
    }
