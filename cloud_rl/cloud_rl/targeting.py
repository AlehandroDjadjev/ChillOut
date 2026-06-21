from __future__ import annotations

from typing import Dict

import torch


def target_augmentation_config(cfg: Dict) -> Dict:
    return dict(cfg.get("target_augmentation") or {})


def augment_target_goal(
    batch: Dict,
    cfg: Dict,
    target_std: float,
    device: torch.device,
) -> Dict:
    """Apply intervention-style scalar goals in-place during RL training.

    The policy uses one scalar goal channel. For temperature rewards this stays
    Celsius-normalized. For radiation rewards it becomes target cloud-loss W/m2
    normalized by `radiation_loss_scale_wm2`.
    """
    aug = target_augmentation_config(cfg)
    if not aug.get("enabled", False):
        return batch

    target_kind = str(cfg.get("reward_target_kind") or aug.get("target_kind") or "temperature")
    if target_kind in {"radiation", "radiation_loss", "radiation_v8"} and "radiation_clear_wm2" in batch:
        scale = max(float(aug.get("radiation_loss_scale_wm2", 300.0)), 1e-6)
        min_target = float(aug.get("min_target_wm2", 0.0))
        jitter = float(aug.get("radiation_jitter_wm2", 320.0))
        random_absolute_prob = float(aug.get("random_absolute_prob", 0.65))

        current = batch.get("current_radiation_loss_wm2", torch.zeros_like(batch["target_temp"])).to(device)
        clear = batch["radiation_clear_wm2"].to(device).clamp_min(min_target)
        valid = batch.get("radiation_valid", torch.ones_like(clear)).to(device)

        jittered = current + (torch.rand_like(current) * 2.0 - 1.0) * jitter
        absolute = min_target + torch.rand_like(current) * (clear - min_target).clamp_min(0.0)
        use_abs = (torch.rand_like(current) < random_absolute_prob).float()
        target = use_abs * absolute + (1.0 - use_abs) * jittered
        target = torch.minimum(target.clamp_min(min_target), clear)
        target = torch.where(valid > 0.5, target, current.clamp_min(min_target))

        batch["target_temp"] = target
        batch["target_temp_norm"] = target / scale
        batch["target_radiation_loss_wm2"] = target
        batch["obs_map"][:, -1:, :, :] = batch["target_temp_norm"][:, :, None, None]
        return batch

    min_abs = float(aug.get("min_abs_offset_c", 1.0))
    max_abs = float(aug.get("max_abs_offset_c", 4.0))
    if max_abs <= 0:
        return batch
    min_abs = max(0.0, min(min_abs, max_abs))

    b = batch["target_temp"].shape[0]
    magnitude = torch.empty((b, 1), device=device).uniform_(min_abs, max_abs)
    sign = torch.where(
        torch.rand((b, 1), device=device) < 0.5,
        torch.full((b, 1), -1.0, device=device),
        torch.full((b, 1), 1.0, device=device),
    )
    offset = magnitude * sign

    target_std = max(float(target_std), 1e-6)
    batch["target_temp"] = batch["target_temp"] + offset
    batch["target_temp_norm"] = batch["target_temp_norm"] + offset / target_std
    batch["obs_map"][:, -1:, :, :] = batch["target_temp_norm"][:, :, None, None]
    return batch


def augment_target_temperature(
    batch: Dict,
    cfg: Dict,
    target_std: float,
    device: torch.device,
) -> Dict:
    return augment_target_goal(batch, cfg, target_std=target_std, device=device)
