from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import FEATURE_KEYS
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


def _align_by_names(
    raw_features: torch.Tensor,
    source_names: Sequence[str],
    target_names: Sequence[str],
) -> torch.Tensor:
    index_by_name = {name: idx for idx, name in enumerate(source_names)}
    columns: List[torch.Tensor] = []
    for name in target_names:
        index = index_by_name.get(name)
        if index is None:
            columns.append(torch.zeros((raw_features.shape[0], 1), dtype=raw_features.dtype, device=raw_features.device))
        else:
            columns.append(raw_features[:, index : index + 1])
    return torch.cat(columns, dim=1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.clamp_min(0.0)
    den = weights.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-6)
    num = (values * weights).flatten(1).sum(dim=1, keepdim=True)
    return num / den


def _weighted_std(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    mean = _weighted_mean(values, weights).view(-1, 1, 1, 1)
    var = _weighted_mean((values - mean).pow(2), weights)
    return var.clamp_min(0.0).sqrt()


def _binary_edge_density(mask: torch.Tensor) -> torch.Tensor:
    binary = (mask > 0.5).float()
    dx = (binary[:, :, :, 1:] - binary[:, :, :, :-1]).abs().mean(dim=(1, 2, 3), keepdim=True)
    dy = (binary[:, :, 1:, :] - binary[:, :, :-1, :]).abs().mean(dim=(1, 2, 3), keepdim=True)
    return 0.5 * (dx + dy)


def _masked_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    flat_values = values.flatten(1)
    flat_weights = weights.flatten(1)
    for row_values, row_weights in zip(flat_values, flat_weights):
        selected = row_values[row_weights > 0.05]
        if selected.numel() == 0:
            rows.append(row_values.new_zeros(1))
        else:
            rows.append(torch.quantile(selected.float(), q).view(1).to(dtype=row_values.dtype))
    return torch.stack(rows, dim=0)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
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


class DeepTabularMLP(nn.Module):
    def __init__(self, num_features: int, embedding_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),

            nn.Linear(256, 384),
            nn.LayerNorm(384),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),

            nn.Linear(384, 384),
            nn.LayerNorm(384),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),

            nn.Linear(384, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CloudTempDeepModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.image_encoder = DeepCloudCNN(embedding_dim=512)
        self.tabular_encoder = DeepTabularMLP(num_features=num_features, embedding_dim=256)
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


class CloudImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            DownStage(1, 32, blocks=1, dropout=0.02),
            DownStage(32, 64, blocks=2, dropout=0.03),
            DownStage(64, 128, blocks=2, dropout=0.04),
            DownStage(128, 256, blocks=2, dropout=0.05),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim), nn.LayerNorm(out_dim), nn.SiLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CloudWorldInteractionModel(nn.Module):
    def __init__(
        self,
        num_cloud_features: int,
        num_world_features: int,
        cloud_dim: int = 256,
        world_dim: int = 192,
        seq_dim: int = 192,
        dropout: float = 0.10,
        use_gru: bool = True,
    ):
        super().__init__()
        self.use_gru = use_gru
        self.cloud_image = CloudImageEncoder(embedding_dim=cloud_dim)
        self.image_frame = MLP(cloud_dim, [cloud_dim], seq_dim, dropout=dropout)
        self.cloud_scalar = MLP(num_cloud_features, [128, 128], cloud_dim // 2, dropout=dropout)
        self.cloud_frame = MLP(cloud_dim + cloud_dim // 2, [cloud_dim], seq_dim, dropout=dropout)
        self.world_frame = MLP(num_world_features, [192, 192], seq_dim, dropout=dropout)

        if use_gru:
            self.image_gru = nn.GRU(seq_dim, seq_dim, batch_first=True)
            self.cloud_gru = nn.GRU(seq_dim, seq_dim, batch_first=True)
            self.world_gru = nn.GRU(seq_dim, seq_dim, batch_first=True)
        else:
            self.image_gru = None
            self.cloud_gru = None
            self.world_gru = None

        self.image_head = nn.Sequential(nn.Linear(seq_dim, 96), nn.SiLU(inplace=True), nn.Linear(96, 1))
        self.cloud_head = nn.Sequential(nn.Linear(seq_dim, 96), nn.SiLU(inplace=True), nn.Linear(96, 1))
        self.world_head = nn.Sequential(nn.Linear(seq_dim, 96), nn.SiLU(inplace=True), nn.Linear(96, 1))
        self.final_bias = nn.Parameter(torch.zeros(1))

    def encode_image_sequence(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = mask.shape
        mask_2d = mask.reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        img_emb = self.cloud_image(mask_2d).view(b, t, -1)
        img_frame = self.image_frame(img_emb.reshape(b * t, -1)).view(b, t, -1)
        if self.use_gru:
            _, h_last = self.image_gru(img_frame)
            return h_last[-1], img_emb
        return img_frame.mean(dim=1), img_emb

    def encode_cloud(self, mask: torch.Tensor, cloud_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _, _, _ = mask.shape
        image_seq, img_emb = self.encode_image_sequence(mask)
        scalar_emb = self.cloud_scalar(cloud_features.reshape(b * t, cloud_features.size(-1))).view(b, t, -1)
        frame = self.cloud_frame(torch.cat([img_emb, scalar_emb], dim=-1).reshape(b * t, -1)).view(b, t, -1)
        if self.use_gru:
            _, h_last = self.cloud_gru(frame)
            return h_last[-1], image_seq
        return frame.mean(dim=1), image_seq

    def encode_world(self, world_features: torch.Tensor) -> torch.Tensor:
        b, t, f = world_features.shape
        frame = self.world_frame(world_features.reshape(b * t, f)).view(b, t, -1)
        if self.use_gru:
            _, h_last = self.world_gru(frame)
            return h_last[-1]
        return frame.mean(dim=1)

    def forward(self, mask: torch.Tensor, cloud_features: torch.Tensor, world_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        cloud, image_only = self.encode_cloud(mask, cloud_features)
        world = self.encode_world(world_features)
        image_pred = self.image_head(image_only)
        cloud_pred = self.cloud_head(cloud)
        world_pred = self.world_head(world)
        return {
            "final": image_pred + cloud_pred + world_pred + self.final_bias,
            "image": image_pred,
            "cloud": cloud_pred,
            "world": world_pred,
        }


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


<<<<<<< HEAD
=======
def _strip_compile_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state):
        return state
    return {key.replace("_orig_mod.", "", 1): value for key, value in state.items()}


def _load_v6_model_class():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "NewModel" / "train_cloud_temp_cloudforced_radiation_v6.py"
    if not module_path.exists():
        raise FileNotFoundError(
            "Could not load CloudForcedRadiationSplitConvLSTM_v6 reward model because "
            f"{module_path} does not exist."
        )
    spec = importlib.util.spec_from_file_location("cloudforced_radiation_v6_reward", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import v6 reward model from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ResidualTrendSplitConvLSTM


>>>>>>> 7121b02f0e3503fc5d02214fb2f226d64c554238
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
        improvement_gain: float = 100.0,
        absolute_error_weight: float = 0.0,
        optional_budget_penalty: float = 0.0,
        prefer_best: bool = True,
    ) -> None:
        super().__init__()
        resolved = resolve_first_model_checkpoint(checkpoint_path, prefer_best=prefer_best)
        ckpt = torch.load(resolved, map_location="cpu")

        self.checkpoint_path = str(resolved)
        self.reward_scale_c = float(reward_scale_c)
        self.improvement_gain = float(improvement_gain)
        self.absolute_error_weight = float(absolute_error_weight)
        self.optional_budget_penalty = float(optional_budget_penalty)
        self.raw_feature_names = list(ckpt["raw_feature_names"])
        self.model_feature_names = list(ckpt.get("model_feature_names", self.raw_feature_names))
        self.cloud_feature_names = list(ckpt.get("cloud_feature_names", []))
        self.world_feature_names = list(ckpt.get("world_feature_names", []))
        self.architecture = str(ckpt.get("architecture", "CloudTempDeepModel"))
        self.model_kind = (
            "cloudforced_radiation_v6"
            if "CloudForcedRadiationSplitConvLSTM_v6" in self.architecture or ckpt.get("target_is_delta")
            else ("interaction" if self.cloud_feature_names and self.world_feature_names else "deep")
        )
        self.image_height = int(ckpt["image_height"])
        self.image_width = int(ckpt["image_width"])
        self.lookback = int(ckpt.get("lookback", 1))

        normalizer = ckpt["normalizer"]
        feature_mean = torch.tensor(normalizer["mean"], dtype=torch.float32)
        feature_std = torch.tensor(normalizer["std"], dtype=torch.float32)
        feature_std = torch.where(feature_std.abs() < 1e-6, torch.ones_like(feature_std), feature_std)
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        target_norm = ckpt.get("target_normalizer") or {"mean": 0.0, "std": 1.0}
        self.target_mean_c = float(target_norm.get("mean", 0.0))
        self.target_std_c = float(target_norm.get("std", 1.0)) or 1.0
        delta_norm = ckpt.get("delta_normalizer") or target_norm
        self.delta_mean_c = float(delta_norm.get("mean", self.target_mean_c))
        self.delta_std_c = float(delta_norm.get("std", self.target_std_c)) or 1.0
        self.context_scale = float((ckpt.get("args") or {}).get("context_scale", 0.15))

        state = _strip_compile_prefix(_strip_module_prefix(ckpt["model_state"]))
        if self.model_kind == "cloudforced_radiation_v6":
            model_cls = _load_v6_model_class()
            self.model = model_cls(**dict(ckpt.get("model_kwargs") or {}))
        elif self.model_kind == "interaction":
            kwargs = dict(ckpt.get("model_kwargs") or {})
            kwargs.setdefault("num_cloud_features", len(self.cloud_feature_names))
            kwargs.setdefault("num_world_features", len(self.world_feature_names))
            self.model = CloudWorldInteractionModel(**kwargs)
        else:
            self.model = CloudTempDeepModel(num_features=len(self.model_feature_names))
        self.model.load_state_dict(state)
        if torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _resolve_feature_names(self, raw_feature_count: int) -> Sequence[str]:
        if raw_feature_count == len(self.raw_feature_names):
            return self.raw_feature_names
        if raw_feature_count == len(FEATURE_KEYS):
            return FEATURE_KEYS
        if raw_feature_count == len(self.model_feature_names):
            return self.model_feature_names
        raise ValueError(
            "Feature vector width does not match the checkpoint metadata or the canonical dataset schema: "
            f"got {raw_feature_count}, raw_feature_names={len(self.raw_feature_names)}, "
            f"feature_keys={len(FEATURE_KEYS)}, model_feature_names={len(self.model_feature_names)}."
        )

    def _align_raw_features(self, feature_vector: torch.Tensor) -> torch.Tensor:
        raw = feature_vector.float()
        source_names = self._resolve_feature_names(raw.shape[1])
        if self.model_kind == "deep" and raw.shape[1] == len(self.model_feature_names):
            return raw
        return _align_by_names(raw, source_names, self.raw_feature_names)

    def _normalize_processed_features(self, processed: torch.Tensor) -> torch.Tensor:
        expected_dim = self.feature_mean.shape[0]
        if processed.shape[1] < expected_dim:
            processed = F.pad(processed, (0, expected_dim - processed.shape[1]))
        elif processed.shape[1] > expected_dim:
            processed = processed[:, :expected_dim]
        return (processed - self.feature_mean) / self.feature_std

    def _prepare_features(self, feature_vector: torch.Tensor) -> torch.Tensor:
        raw = feature_vector.float()
        source_names = self._resolve_feature_names(raw.shape[1])
        if self.model_kind == "deep" and raw.shape[1] == len(self.model_feature_names):
            processed = raw
        else:
            aligned = _align_by_names(raw, source_names, self.raw_feature_names)
            processed = aligned if self.model_kind == "interaction" else _transform_raw_features(aligned, self.raw_feature_names)
        return self._normalize_processed_features(processed)

    def _prepare_aligned_features(self, aligned_features: torch.Tensor) -> torch.Tensor:
        processed = aligned_features if self.model_kind == "interaction" else _transform_raw_features(aligned_features, self.raw_feature_names)
        return self._normalize_processed_features(processed)

    def _apply_action_to_cloud_features(
        self,
        aligned_raw_features: torch.Tensor,
        original_mask: torch.Tensor,
        generated_mask: torch.Tensor,
        property_maps: torch.Tensor,
    ) -> torch.Tensor:
        if self.model_kind != "interaction":
            return aligned_raw_features

        index = {name: i for i, name in enumerate(self.raw_feature_names)}
        if not any(name.startswith("cloud_s2_") for name in index):
            return aligned_raw_features

        updated = aligned_raw_features.clone()
        original = original_mask.float().clamp(0.0, 1.0)
        generated = generated_mask.float().clamp(0.0, 1.0)
        props = property_maps.float()
        if props.shape[-2:] != generated.shape[-2:]:
            props = F.interpolate(props, size=generated.shape[-2:], mode="bilinear", align_corners=False)
        props = props.clamp(0.0, 1.0)

        original_fraction_img = original.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        generated_fraction_img = generated.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        fraction_delta = generated_fraction_img - original_fraction_img

        original_prob_mean_img = _weighted_mean(original, original)
        generated_prob_mean_img = _weighted_mean(generated, generated)
        prob_mean_delta = generated_prob_mean_img - original_prob_mean_img
        prob_std_delta = _weighted_std(generated, generated) - _weighted_std(original, original)
        prob_p90_delta = _masked_quantile(generated, generated, 0.90) - _masked_quantile(original, original, 0.90)
        edge_delta = _binary_edge_density(generated).view(-1, 1) - _binary_edge_density(original).view(-1, 1)
        texture_delta = _weighted_std(generated, generated) - _weighted_std(original, original)

        action_strength = props.amax(dim=1, keepdim=True)
        action_area = action_strength.mean(dim=(1, 2, 3), keepdim=False)[:, None].clamp(0.0, 1.0)
        action_present = (action_area > 1e-6).float()
        action_prob = _weighted_mean(props[:, 0:1], action_strength)
        action_aot = 2.5 * _weighted_mean(props[:, 1:2], action_strength)
        action_layer = _weighted_mean(props[:, 2:3], action_strength)
        action_texture = _weighted_mean(props[:, 3:4], action_strength)
        action_cirrus = _weighted_mean(props[:, 4:5], action_strength)

        def set_feature(name: str, value: torch.Tensor, lo: float | None = None, hi: float | None = None) -> None:
            pos = index.get(name)
            if pos is None:
                return
            out = value
            if lo is not None or hi is not None:
                out = out.clamp(
                    min=-float("inf") if lo is None else lo,
                    max=float("inf") if hi is None else hi,
                )
            updated[:, pos : pos + 1] = out

        def old(name: str) -> torch.Tensor:
            pos = index[name]
            return aligned_raw_features[:, pos : pos + 1]

        if "cloud_s2_fraction" in index:
            set_feature("cloud_s2_fraction", old("cloud_s2_fraction") + fraction_delta, 0.0, 1.0)
        if "cloud_s2_prob_mean" in index:
            set_feature(
                "cloud_s2_prob_mean",
                old("cloud_s2_prob_mean") + prob_mean_delta + action_present * action_area * (action_prob - old("cloud_s2_prob_mean")),
                0.0,
                1.0,
            )
        if "cloud_s2_prob_std" in index:
            set_feature("cloud_s2_prob_std", old("cloud_s2_prob_std") + prob_std_delta + 0.25 * action_area * action_texture, 0.0, 1.0)
        if "cloud_s2_prob_p90" in index:
            set_feature("cloud_s2_prob_p90", old("cloud_s2_prob_p90") + prob_p90_delta + action_area * (action_prob - old("cloud_s2_prob_p90")), 0.0, 1.0)
        if "cloud_s2_aot_mean" in index:
            set_feature("cloud_s2_aot_mean", old("cloud_s2_aot_mean") + action_present * action_area * (action_aot - old("cloud_s2_aot_mean")), 0.0, None)
        if "cloud_s2_cirrus_fraction" in index:
            set_feature("cloud_s2_cirrus_fraction", old("cloud_s2_cirrus_fraction") + action_area * action_cirrus + 0.25 * fraction_delta, 0.0, 1.0)
        if "cloud_s2_high_fraction" in index:
            high_proxy = (action_layer > 0.66).float() * action_prob
            set_feature("cloud_s2_high_fraction", old("cloud_s2_high_fraction") + action_area * high_proxy + 0.25 * fraction_delta, 0.0, 1.0)
        if "cloud_s2_medium_fraction" in index:
            medium_proxy = ((action_layer >= 0.33) & (action_layer <= 0.66)).float() * action_prob
            set_feature("cloud_s2_medium_fraction", old("cloud_s2_medium_fraction") + action_area * medium_proxy + 0.25 * fraction_delta, 0.0, 1.0)
        if "cloud_s2_edge_density" in index:
            set_feature("cloud_s2_edge_density", old("cloud_s2_edge_density") + edge_delta, 0.0, None)
        if "cloud_s2_texture_std" in index:
            set_feature("cloud_s2_texture_std", old("cloud_s2_texture_std") + texture_delta + action_area * action_texture, 0.0, None)

        return updated

    def _prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] == (self.image_height, self.image_width):
            return mask.float()
        return F.interpolate(
            mask.float(),
            size=(self.image_height, self.image_width),
            mode="bilinear",
            align_corners=False,
        )

<<<<<<< HEAD
    def _predict_temperature(self, mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if self.model_kind == "deep":
            return self.model(mask, features)

=======
    def _split_cloud_world(self, normalized_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
>>>>>>> 7121b02f0e3503fc5d02214fb2f226d64c554238
        raw_index = {name: i for i, name in enumerate(self.raw_feature_names)}
        cloud_idx = [raw_index[name] for name in self.cloud_feature_names]
        world_idx = [raw_index[name] for name in self.world_feature_names]
        return normalized_features[..., cloud_idx], normalized_features[..., world_idx]

    def _normalize_raw_sequence(self, raw_sequence: torch.Tensor) -> torch.Tensor:
        b, t, f = raw_sequence.shape
        flat = raw_sequence.reshape(b * t, f)
        source_names = self._resolve_feature_names(f)
        aligned = _align_by_names(flat.float(), source_names, self.raw_feature_names)
        norm = self._prepare_aligned_features(aligned)
        return norm.view(b, t, -1)

    def _predict_temperature(
        self,
        mask: torch.Tensor,
        features: torch.Tensor,
        mask_sequence: Optional[torch.Tensor] = None,
        feature_sequence: Optional[torch.Tensor] = None,
        trend_features: Optional[torch.Tensor] = None,
        current_temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.model_kind == "deep":
            return self.model(mask, features)

        if self.model_kind == "cloudforced_radiation_v6":
            steps = max(1, self.lookback)
            if mask_sequence is None:
                mask_sequence = mask[:, None, :, :, :].expand(-1, steps, -1, -1, -1).contiguous()
            if feature_sequence is None:
                feature_sequence = features[:, None, :].expand(-1, steps, -1).contiguous()
            if trend_features is None:
                trend_features = torch.zeros(
                    (mask_sequence.shape[0], mask_sequence.shape[1], 4),
                    dtype=features.dtype,
                    device=features.device,
                )
            if current_temperature is None:
                current_temperature = torch.zeros((mask_sequence.shape[0], 1), dtype=features.dtype, device=features.device)

            cloud, world = self._split_cloud_world(feature_sequence)
            out = self.model(mask_sequence, cloud, world, trend_features, context_scale=self.context_scale)
            delta_c = out["final_delta"] * self.delta_std_c + self.delta_mean_c
            return current_temperature.float() + delta_c

        cloud, world = self._split_cloud_world(features)
        steps = max(1, self.lookback)
        seq_mask = mask[:, None, :, :, :].expand(-1, steps, -1, -1, -1).contiguous()
        seq_cloud = cloud[:, None, :].expand(-1, steps, -1).contiguous()
        seq_world = world[:, None, :].expand(-1, steps, -1).contiguous()
        pred_norm = self.model(seq_mask, seq_cloud, seq_world)["final"]
        return pred_norm * self.target_std_c + self.target_mean_c

    def forward(
        self,
        original_mask: torch.Tensor,
        generated_mask: torch.Tensor,
        feature_vector: torch.Tensor,
        target_temperature: torch.Tensor,
        property_maps: torch.Tensor,
        original_mask_sequence: Optional[torch.Tensor] = None,
        raw_feature_sequence: Optional[torch.Tensor] = None,
        trend_features: Optional[torch.Tensor] = None,
        current_temperature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        original = self._prepare_mask(original_mask)
        generated = self._prepare_mask(generated_mask)
        if original.ndim == 4 and original.is_cuda:
            original = original.contiguous(memory_format=torch.channels_last)
        if generated.ndim == 4 and generated.is_cuda:
            generated = generated.contiguous(memory_format=torch.channels_last)
        original_seq = generated_seq = None
        original_feature_seq = generated_feature_seq = None
        trend = None
        current = None

        if self.model_kind == "cloudforced_radiation_v6":
            if original_mask_sequence is not None:
                original_seq = original_mask_sequence.float()
                if original_seq.shape[-2:] != (self.image_height, self.image_width):
                    b, t, c, _, _ = original_seq.shape
                    original_seq = F.interpolate(
                        original_seq.reshape(b * t, c, *original_mask_sequence.shape[-2:]),
                        size=(self.image_height, self.image_width),
                        mode="bilinear",
                        align_corners=False,
                    ).view(b, t, c, self.image_height, self.image_width)
            else:
                original_seq = original[:, None, :, :, :].expand(-1, max(1, self.lookback), -1, -1, -1).contiguous()
            generated_seq = original_seq.clone()
            generated_seq[:, -1] = generated

            if raw_feature_sequence is not None:
                original_feature_seq = self._normalize_raw_sequence(raw_feature_sequence.float())
            else:
                original_feature_seq = self._prepare_aligned_features(self._align_raw_features(feature_vector))[:, None, :].expand(
                    -1, original_seq.shape[1], -1
                ).contiguous()
            aligned_raw = self._align_raw_features(feature_vector)
            generated_raw = self._apply_action_to_cloud_features(aligned_raw, original, generated, property_maps)
            generated_last = self._prepare_aligned_features(generated_raw)
            generated_feature_seq = original_feature_seq.clone()
            generated_feature_seq[:, -1] = generated_last
            original_features = original_feature_seq[:, -1]
            generated_features = generated_last
            trend = trend_features.float() if trend_features is not None else None
            current = current_temperature.float() if current_temperature is not None else None
        elif self.model_kind == "interaction":
            aligned_raw = self._align_raw_features(feature_vector)
            generated_raw = self._apply_action_to_cloud_features(aligned_raw, original, generated, property_maps)
            original_features = self._prepare_aligned_features(aligned_raw)
            generated_features = self._prepare_aligned_features(generated_raw)
        else:
            generated_raw = None
            original_features = self._prepare_features(feature_vector)
            generated_features = original_features

        with torch.inference_mode():
            if generated.is_cuda:
                stacked_masks = torch.cat([original, generated], dim=0)
                stacked_features = torch.cat([original_features, generated_features], dim=0)
                if self.model_kind == "cloudforced_radiation_v6":
                    stacked_mask_seq = torch.cat([original_seq, generated_seq], dim=0)
                    stacked_feature_seq = torch.cat([original_feature_seq, generated_feature_seq], dim=0)
                    stacked_trend = torch.cat([trend, trend], dim=0) if trend is not None else None
                    stacked_current = torch.cat([current, current], dim=0) if current is not None else None
                else:
                    stacked_mask_seq = stacked_feature_seq = stacked_trend = stacked_current = None
                with torch.autocast(device_type="cuda", enabled=True):
                    stacked_pred = self._predict_temperature(
                        stacked_masks,
                        stacked_features,
                        mask_sequence=stacked_mask_seq,
                        feature_sequence=stacked_feature_seq,
                        trend_features=stacked_trend,
                        current_temperature=stacked_current,
                    )
            else:
                stacked_pred = self._predict_temperature(
                    torch.cat([original, generated], dim=0),
                    torch.cat([original_features, generated_features], dim=0),
                    mask_sequence=torch.cat([original_seq, generated_seq], dim=0) if self.model_kind == "cloudforced_radiation_v6" else None,
                    feature_sequence=torch.cat([original_feature_seq, generated_feature_seq], dim=0) if self.model_kind == "cloudforced_radiation_v6" else None,
                    trend_features=torch.cat([trend, trend], dim=0) if self.model_kind == "cloudforced_radiation_v6" and trend is not None else None,
                    current_temperature=torch.cat([current, current], dim=0) if self.model_kind == "cloudforced_radiation_v6" and current is not None else None,
                )

        original_predicted_temperature, predicted_temperature = stacked_pred.chunk(2, dim=0)

        original_temp_error = (original_predicted_temperature - target_temperature.float()).abs()
        temp_error = (predicted_temperature - target_temperature.float()).abs()
        temp_improvement = original_temp_error - temp_error
        reward = self.improvement_gain * temp_improvement
        if self.absolute_error_weight:
            reward = reward - self.absolute_error_weight * temp_error

        if self.optional_budget_penalty > 0:
            change_cost = (generated_mask - original_mask).abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
            prop_cost = property_maps.abs().mean(dim=(1, 2, 3), keepdim=False)[:, None]
            reward = reward - self.optional_budget_penalty * (change_cost + prop_cost)

        return reward, {
            "original_predicted_temperature_c": original_predicted_temperature.detach(),
            "original_temp_error_c": original_temp_error.detach(),
            "predicted_temperature_c": predicted_temperature.detach(),
            "temp_error_c": temp_error.detach(),
            "temp_improvement_c": temp_improvement.detach(),
            "generated_cloud_fraction": generated_raw[:, self.raw_feature_names.index("cloud_s2_fraction") : self.raw_feature_names.index("cloud_s2_fraction") + 1].detach()
            if generated_raw is not None and "cloud_s2_fraction" in self.raw_feature_names else torch.zeros_like(temp_error).detach(),
        }
