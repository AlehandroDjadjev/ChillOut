#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


ANGLE_FEATURE_NAMES = {"wind_direction_10m_dominant"}


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

    def forward(self, x):
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

    def forward(self, mask):
        return self.proj(self.cnn(mask))


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

    def forward(self, features):
        return self.net(features)


class CloudTempModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.image_encoder = CloudImageEncoder(256)
        self.tabular_encoder = TabularEncoder(num_features, 128)
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256),
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

    def forward(self, mask, features):
        img_emb = self.image_encoder(mask)
        tab_emb = self.tabular_encoder(features)
        return self.head(torch.cat([img_emb, tab_emb], dim=1))


def transform_raw_feature_row(raw_values: List[float], raw_feature_names: List[str]) -> List[float]:
    out = []
    for value, name in zip(raw_values, raw_feature_names):
        if name in ANGLE_FEATURE_NAMES:
            radians = math.radians(value)
            out.append(math.sin(radians))
            out.append(math.cos(radians))
        else:
            out.append(value)
    return out


def extract_raw_features(record: Dict[str, Any], raw_feature_names: List[str]) -> List[float]:
    if "inputs" in record:
        return [float(record["inputs"][name]) for name in raw_feature_names]
    return [float(x) for x in record["feature_vector"]]


def load_mask(data_root: Path, mask_path: str, height: int, width: int) -> torch.Tensor:
    img = Image.open(data_root / mask_path).convert("L")
    img = img.resize((width, height), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-json", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    sample_path = Path(args.sample_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    raw_feature_names = ckpt["raw_feature_names"]
    model_feature_names = ckpt["model_feature_names"]
    image_height = int(ckpt["image_height"])
    image_width = int(ckpt["image_width"])
    mean = np.array(ckpt["normalizer"]["mean"], dtype=np.float32)
    std = np.array(ckpt["normalizer"]["std"], dtype=np.float32)
    std[std == 0] = 1.0

    record = json.loads(sample_path.read_text(encoding="utf-8"))

    raw = extract_raw_features(record, raw_feature_names)
    processed = np.array(transform_raw_feature_row(raw, raw_feature_names), dtype=np.float32)
    features = (processed - mean) / std
    features_t = torch.from_numpy(features).unsqueeze(0).to(device)

    mask_t = load_mask(data_root, record["mask_path"], image_height, image_width).to(device)

    model = CloudTempModel(num_features=len(model_feature_names)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pred = model(mask_t, features_t).item()

    print(json.dumps({
        "sample_id": record.get("sample_id"),
        "predicted_temperature_c": pred,
        "target_temperature_c": record.get("target_temperature_c"),
    }, indent=2))


if __name__ == "__main__":
    main()