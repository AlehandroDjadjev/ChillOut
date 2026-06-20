from __future__ import annotations

from typing import Dict

import torch


def target_augmentation_config(cfg: Dict) -> Dict:
    return dict(cfg.get("target_augmentation") or {})


def augment_target_temperature(
    batch: Dict,
    cfg: Dict,
    target_std: float,
    device: torch.device,
) -> Dict:
    """Apply intervention-style target offsets in-place during RL training."""
    aug = target_augmentation_config(cfg)
    if not aug.get("enabled", False):
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
