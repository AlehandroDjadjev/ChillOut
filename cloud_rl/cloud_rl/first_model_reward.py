from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rewards import AbstractRewardModel


ANGLE_FEATURE_NAMES = {"wind_direction_10m_dominant"}


def resolve_first_model_checkpoint(source: str | Path, prefer_best: bool = True) -> Path:
    path = Path(source)

    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"First-model checkpoint path not found: {path}")

    candidates = ["best.pt", "last.pt"] if prefer_best else ["last.pt", "best.pt"]
    for name in candidates:
        candidate = path / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find best.pt or last.pt under {path}")


def _transform_raw_features(raw_features: torch.Tensor, raw_feature_names: Sequence[str]) -> torch.Tensor:
    columns: List[torch.Tensor] = []
    for index, name in enumerate(raw_feature_names):
        column = raw_features[:, index]
        if name in ANGLE_FEATURE_NAMES:
            radians = column * math.pi / 180.0
            columns.append(torch.sin(radians).unsqueeze(1))
            columns.append(torch.cos(radians).unsqueeze(1))
        else:
            columns.append(column.unsqueeze(1))
    return torch.cat(columns, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CloudImageEncoder(nn.Module):
    def __init__(self, image_embedding_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, dropout=0.02),
            ConvBlock(32, 64, dropout=0.03),
            ConvBlock(64, 128, dropout=0.05),
            ConvBlock(128, 192, dropout=0.05),
            ConvBlock(192, 256, dropout=0.05),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, image_embedding_dim),
            nn.LayerNorm(image_embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(self.cnn(mask))


class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DownStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int, dropout: float):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.append(ResBlock(out_ch, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepCloudCNN(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            DownStage(1, 48, blocks=2, dropout=0.02),
            DownStage(48, 96, blocks=2, dropout=0.03),
            DownStage(96, 192, blocks=3, dropout=0.04),
            DownStage(192, 384, blocks=3, dropout=0.05),
            DownStage(384, 512, blocks=2, dropout=0.05),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularEncoder(nn.Module):
    def __init__(self, num_features: int, tab_embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 192),
            nn.LayerNorm(192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(192, tab_embedding_dim),
            nn.LayerNorm(tab_embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CloudTempDeepModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.image_encoder = DeepCloudCNN(embedding_dim=512)
        self.tabular_encoder = TabularEncoder(num_features=num_features, tab_embedding_dim=256)
        self.head = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        image_emb = self.image_encoder(mask)
        tab_emb = self.tabular_encoder(features)
        return self.head(torch.cat([image_emb, tab_emb], dim=1))


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def _is_deep_checkpoint(state: Dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("image_encoder.net.") for key in state)


def _assert_deep_checkpoint(state: Dict[str, torch.Tensor], resolved_path: Path) -> None:
    if _is_deep_checkpoint(state):
        return
    raise RuntimeError(
        "The reward wrapper now accepts only deep first-model checkpoints. "
        f"Checkpoint {resolved_path} does not look like a deep-model state dict."
    )


class CloudTempCheckpointReward(AbstractRewardModel):
    """Reward model backed by a saved CloudTempModel checkpoint.

    The checkpoint is the "first model" trained on the cloud masks + weather
    features. The reward is higher when the checkpoint predicts a temperature
    closer to the target temperature.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        reward_scale_c: float = 5.0,
        optional_budget_penalty: float = 0.0,
        prefer_best: bool = True,
    ) -> None:
        super().__init__()
        resolved = resolve_first_model_checkpoint(checkpoint_path, prefer_best=prefer_best)
        ckpt = torch.load(resolved, map_location="cpu")

        self.checkpoint_path = str(resolved)
        self.reward_scale_c = float(reward_scale_c)
        self.optional_budget_penalty = float(optional_budget_penalty)
        self.raw_feature_names = list(ckpt["raw_feature_names"])
        self.model_feature_names = list(ckpt["model_feature_names"])
        self.image_height = int(ckpt["image_height"])
        self.image_width = int(ckpt["image_width"])

        normalizer = ckpt["normalizer"]
        feature_mean = torch.tensor(normalizer["mean"], dtype=torch.float32)
        feature_std = torch.tensor(normalizer["std"], dtype=torch.float32)
        feature_std = torch.where(feature_std.abs() < 1e-6, torch.ones_like(feature_std), feature_std)
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)

        state = _strip_module_prefix(ckpt["model_state"])
        _assert_deep_checkpoint(state, resolved)
        self.model = CloudTempDeepModel(num_features=len(self.model_feature_names))
        self.model.load_state_dict(state)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _prepare_features(self, feature_vector: torch.Tensor) -> torch.Tensor:
        raw = feature_vector.float()
        processed = _transform_raw_features(raw, self.raw_feature_names)
        return (processed - self.feature_mean) / self.feature_std

    def _prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] == (self.image_height, self.image_width):
            return mask.float()
        return F.interpolate(
            mask.float(),
            size=(self.image_height, self.image_width),
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        original_mask: torch.Tensor,
        generated_mask: torch.Tensor,
        feature_vector: torch.Tensor,
        target_temperature: torch.Tensor,
        property_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask = self._prepare_mask(generated_mask)
        features = self._prepare_features(feature_vector)

        with torch.no_grad():
            predicted_temperature = self.model(mask, features)

        temp_error = (predicted_temperature - target_temperature.float()).abs()
        reward = torch.exp(-temp_error / max(1e-6, self.reward_scale_c))

        if self.optional_budget_penalty > 0:
            change_cost = (generated_mask - original_mask).abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
            prop_cost = property_maps.abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
            reward = reward - self.optional_budget_penalty * (change_cost + prop_cost)

        return reward, {
            "predicted_temperature_c": predicted_temperature.detach(),
            "temp_error_c": temp_error.detach(),
        }
