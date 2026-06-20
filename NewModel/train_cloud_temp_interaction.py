#!/usr/bin/env python3
"""
train_cloud_temp_interaction.py

Two-stream cloud/world temperature model.

Compared with the old CNN+tabular MLP:
  - cloud image + cloud scalar features are encoded in a cloud branch
  - non-cloud world features are encoded in a separate world branch
  - a cloud->world bridge learns how cloud state modulates world context
  - separate cloud/world/interaction heads are exposed for diagnostics
  - optional sequence/GRU head lets the CNN/MLP model use recent history

Output is still a single temperature number in Celsius, trained internally in normalized target space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def can_use_torch_compile() -> tuple[bool, str]:
    """Return whether torch.compile/inductor is likely usable.

    On many Windows CUDA installs, Inductor fails at first forward because Triton is
    missing or unavailable. In that case eager mode is better and avoids the long
    no-progress compile stall.
    """
    if not hasattr(torch, "compile"):
        return False, "torch.compile is not available in this PyTorch build"
    try:
        import triton  # type: ignore  # noqa: F401
    except Exception as exc:
        return False, f"Triton is not available: {exc}"
    return True, "ok"


def resolve_resume_checkpoint(source: str, out_dir: Path, fresh: bool) -> Optional[Path]:
    if fresh or source in {"", "none"}:
        return None
    if source == "auto":
        candidate = out_dir / "last.pt"
        return candidate if candidate.exists() else None

    path = Path(source)
    if path.is_dir():
        candidate = path / "last.pt"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No last.pt found under resume directory: {path}")
    if path.exists():
        return path
    raise FileNotFoundError(f"Resume checkpoint not found: {path}")


def maybe_build_dataset(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root or args.dataset_out).resolve()
    if not args.build_dataset:
        if not data_root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {data_root}. "
                "Run with --build-dataset --start-date YYYY-MM-DD --end-date YYYY-MM-DD first."
            )
        return data_root

    if not args.start_date or not args.end_date:
        raise ValueError("--build-dataset requires --start-date and --end-date")

    builder = Path(__file__).resolve().parent / "build_temperature_dataset_cloudforce.py"
    cmd = [
        sys.executable,
        str(builder),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--out",
        str(data_root),
        "--patch-km",
        str(args.patch_km),
        "--resolution-m",
        str(args.resolution_m),
        "--max-cloud",
        str(args.max_cloud),
        "--max-scenes-per-location",
        str(args.max_scenes_per_location),
        "--s2-workers",
        str(args.s2_workers),
        "--openmeteo-workers",
        str(args.openmeteo_workers),
        "--openmeteo-retries",
        str(args.openmeteo_retries),
        "--openmeteo-sleep-s",
        str(args.openmeteo_sleep_s),
        "--target-offset-days",
        str(args.target_offset_days),
        "--seed",
        str(args.seed),
    ]
    if args.locations_json:
        cmd.extend(["--locations-json", args.locations_json])
    if args.openmeteo_continue_on_fail:
        cmd.append("--openmeteo-continue-on-fail")
    if args.skip_torch_dataset:
        cmd.append("--skip-torch")

    print("Building dataset with:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return data_root


def load_records(data_root: Path, split: str) -> List[Dict[str, Any]]:
    p = data_root / "splits" / f"{split}.jsonl"
    if p.exists():
        return read_jsonl(p)

    rows = read_jsonl(data_root / "dataset.jsonl")
    rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("location", r.get("city", "")))))
    n = len(rows)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    if split == "train":
        return rows[:train_end]
    if split == "val":
        return rows[train_end:val_end]
    if split == "test":
        return rows[val_end:]
    raise ValueError(split)


def load_metadata(data_root: Path) -> Dict[str, Any]:
    p = data_root / "metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing metadata.json in {data_root}")
    return json.loads(p.read_text(encoding="utf-8"))


def parse_dt(record: Dict[str, Any]) -> np.datetime64:
    raw = record.get("anchor") or record.get("date")
    if raw is None:
        return np.datetime64("NaT")
    s = str(raw).replace("Z", "")
    if "+" in s:
        s = s.split("+", 1)[0]
    return np.datetime64(s)


def get_target(record: Dict[str, Any]) -> float:
    if "target_temperature_c" in record:
        return float(record["target_temperature_c"])
    if "target" in record:
        return float(record["target"])
    raise KeyError("Missing target_temperature_c")


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std[~np.isfinite(self.std)] = 1.0
        self.std[self.std == 0.0] = 1.0

    @classmethod
    def fit(cls, values: np.ndarray) -> "Normalizer":
        return cls(values.mean(axis=0), values.std(axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values.astype(np.float32) - self.mean) / self.std

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "Normalizer":
        return cls(np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32))


class TargetNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if float(std) != 0.0 and math.isfinite(float(std)) else 1.0

    @classmethod
    def fit(cls, targets: Iterable[float]) -> "TargetNormalizer":
        arr = np.asarray(list(targets), dtype=np.float32)
        return cls(float(arr.mean()), float(arr.std()))

    def transform_tensor(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def inverse_tensor(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, float]) -> "TargetNormalizer":
        return cls(state["mean"], state["std"])


def feature_vector(record: Dict[str, Any], names: List[str]) -> np.ndarray:
    inputs = record.get("inputs")
    if inputs:
        return np.asarray([float(inputs[name]) for name in names], dtype=np.float32)
    raw = record.get("feature_vector")
    if raw is None:
        raise KeyError("Record missing inputs/feature_vector")
    return np.asarray(raw, dtype=np.float32)


@dataclass
class Window:
    records: List[Dict[str, Any]]


def build_windows(records: List[Dict[str, Any]], lookback: int, max_gap_days: float) -> List[Window]:
    by_loc: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        key = str(r.get("location", r.get("city", "")))
        by_loc.setdefault(key, []).append(r)

    windows: List[Window] = []
    max_gap = np.timedelta64(int(round(max_gap_days * 24)), "h")

    for _, rows in by_loc.items():
        rows.sort(key=parse_dt)
        if lookback <= 1:
            windows.extend(Window([r]) for r in rows)
            continue

        for end_idx in range(lookback - 1, len(rows)):
            chunk = rows[end_idx - lookback + 1 : end_idx + 1]
            ok = True
            for a, b in zip(chunk[:-1], chunk[1:]):
                if parse_dt(b) - parse_dt(a) > max_gap:
                    ok = False
                    break
            if ok:
                windows.append(Window(chunk))

    return windows


class CloudTempSequenceDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        records: List[Dict[str, Any]],
        raw_names: List[str],
        cloud_names: List[str],
        world_names: List[str],
        x_norm: Normalizer,
        y_norm: TargetNormalizer,
        image_height: int,
        image_width: int,
        lookback: int,
        max_gap_days: float,
        augment: bool,
        cache_images: bool = False,
    ):
        self.data_root = data_root
        self.records = records
        self.raw_names = raw_names
        self.cloud_names = cloud_names
        self.world_names = world_names
        self.raw_to_idx = {name: i for i, name in enumerate(raw_names)}
        self.cloud_idx = [self.raw_to_idx[name] for name in cloud_names]
        self.world_idx = [self.raw_to_idx[name] for name in world_names]
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.image_height = image_height
        self.image_width = image_width
        self.lookback = lookback
        self.augment = augment
        self.cache_images = cache_images
        self.image_cache: Dict[str, np.ndarray] = {}
        self.windows = build_windows(records, lookback=lookback, max_gap_days=max_gap_days)

        if self.cache_images:
            unique_paths = sorted({str(r["mask_path"]) for w in self.windows for r in w.records})
            for rel_path in tqdm(unique_paths, desc="caching masks", leave=False):
                self.image_cache[rel_path] = self._load_mask_array(rel_path)

    def __len__(self) -> int:
        return len(self.windows)

    def _load_mask_array(self, rel_path: str) -> np.ndarray:
        img = Image.open(self.data_root / rel_path).convert("L")
        if img.size != (self.image_width, self.image_height):
            img = img.resize((self.image_width, self.image_height), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def _load_mask_array(self, rel_path: str) -> np.ndarray:
        img = Image.open(self.data_root / rel_path).convert("L")
        if img.size != (self.image_width, self.image_height):
            img = img.resize((self.image_width, self.image_height), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def load_mask(self, rel_path: str) -> torch.Tensor:
        if self.cache_images and rel_path in self.image_cache:
            arr = self.image_cache[rel_path].copy()
        else:
            arr = self._load_mask_array(rel_path)

        if self.augment:
            if random.random() < 0.5:
                arr = np.ascontiguousarray(arr[:, ::-1])
            if random.random() < 0.25:
                arr = np.clip(arr + np.random.normal(0.0, 0.01, size=arr.shape).astype(np.float32), 0, 1)
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]
        masks = []
        features = []

        for r in win.records:
            masks.append(self.load_mask(str(r["mask_path"])))
            raw = feature_vector(r, self.raw_names)
            features.append(self.x_norm.transform(raw))

        x = np.stack(features, axis=0).astype(np.float32)
        x_cloud = x[:, self.cloud_idx]
        x_world = x[:, self.world_idx]

        target_raw = np.array([get_target(win.records[-1])], dtype=np.float32)
        target_norm = np.array([(target_raw[0] - self.y_norm.mean) / self.y_norm.std], dtype=np.float32)

        last_record = win.records[-1]
        current_temp = last_record.get("current_temperature_c")
        if current_temp is None:
            current_temp = last_record.get("inputs", {}).get("world_temperature_2m", float("nan"))
        current_temp_raw = np.array([float(current_temp)], dtype=np.float32)

        return {
            "mask": torch.stack(masks, dim=0),              # [T,1,H,W]
            "cloud_features": torch.from_numpy(x_cloud),    # [T,Cc]
            "world_features": torch.from_numpy(x_world),    # [T,Cw]
            "target": torch.from_numpy(target_norm),        # [1]
            "target_raw": torch.from_numpy(target_raw),     # [1]
            "current_temp_raw": torch.from_numpy(current_temp_raw),  # [1]
            "sample_id": win.records[-1].get("sample_id", str(idx)),
            "location": win.records[-1].get("location", win.records[-1].get("city", "")),
            "anchor": win.records[-1].get("anchor", win.records[-1].get("date", "")),
        }


def collect_train_feature_matrix(records: List[Dict[str, Any]], raw_names: List[str]) -> np.ndarray:
    return np.asarray([feature_vector(r, raw_names) for r in records], dtype=np.float32)


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
    """
    V3 simpler architecture.

    Removed the previous interaction head because diagnostics showed it was mostly noise.
    The model now has three interpretable components:

      image  : raw Sentinel-2 cloud mask/CNN path
      cloud  : image + Sentinel cloud scalar path
      world  : non-temperature world state path

    Final prediction is the sum of image + cloud + world + bias.
    """

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

        # Image-only path is directly supervised.
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
        # mask [B,T,1,H,W]
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

        final = image_pred + cloud_pred + world_pred + self.final_bias

        return {
            "final": final,
            "image": image_pred,
            "cloud": cloud_pred,
            "world": world_pred,
        }


@torch.no_grad()
def metrics_from_pred(pred_raw: torch.Tensor, target_raw: torch.Tensor) -> Dict[str, float]:
    err = pred_raw - target_raw
    mae = float(err.abs().mean().item())
    rmse = float(torch.sqrt((err ** 2).mean()).item())
    bias = float(err.mean().item())

    p = pred_raw.flatten()
    y = target_raw.flatten()
    if p.numel() > 1 and float(p.std().item()) > 1e-8 and float(y.std().item()) > 1e-8:
        corr = float(torch.corrcoef(torch.stack([p, y]))[0, 1].item())
    else:
        corr = float("nan")
    return {"mae_c": mae, "rmse_c": rmse, "bias_c": bias, "corr": corr}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, y_norm: TargetNormalizer, amp_dtype: torch.dtype) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []
    losses = []
    loss_fn = nn.SmoothL1Loss(beta=0.75, reduction="none")

    for batch in loader:
        mask = batch["mask"].to(device, non_blocking=True)
        cloud = batch["cloud_features"].to(device, non_blocking=True)
        world = batch["world_features"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        target_raw = batch["target_raw"].to(device, non_blocking=True)

        with autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
            out = model(mask, cloud, world)
            loss = loss_fn(out["final"], target).mean()

        pred_raw = y_norm.inverse_tensor(out["final"].float())
        preds.append(pred_raw.detach().cpu())
        targets.append(target_raw.detach().cpu())
        losses.append(float(loss.item()))

    pred_all = torch.cat(preds, dim=0)
    targ_all = torch.cat(targets, dim=0)
    m = metrics_from_pred(pred_all, targ_all)
    m["loss_norm"] = float(np.mean(losses)) if losses else float("nan")
    return m


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace, metadata: Dict[str, Any], x_norm: Normalizer, y_norm: TargetNormalizer, metrics: Dict[str, Any]) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "raw_feature_names": metadata["raw_feature_names"],
            "model_feature_names": metadata.get("model_feature_names", metadata["raw_feature_names"]),
            "cloud_feature_names": metadata["cloud_feature_names"],
            "world_feature_names": metadata["world_feature_names"],
            "normalizer": x_norm.state_dict(),
            "target_normalizer": y_norm.state_dict(),
            "image_height": args.image_height,
            "image_width": args.image_width,
            "lookback": args.lookback,
            "model_kwargs": {
                "num_cloud_features": len(metadata["cloud_feature_names"]),
                "num_world_features": len(metadata["world_feature_names"]),
                "cloud_dim": args.cloud_dim,
                "world_dim": args.world_dim,
                "seq_dim": args.seq_dim,
                "dropout": args.dropout,
                "use_gru": not args.no_gru,
            },
            "metrics": metrics,
            "architecture": "CloudWorldInteractionModelV3_NoInteraction",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset-out", default="dataset_cloudforce_interaction")
    parser.add_argument("--build-dataset", action="store_true", help="Run build_temperature_dataset_cloudforce.py before training.")
    parser.add_argument("--start-date", default=None, help="Dataset extraction start date, used with --build-dataset.")
    parser.add_argument("--end-date", default=None, help="Dataset extraction end date, used with --build-dataset.")
    parser.add_argument("--locations-json", default=None)
    parser.add_argument("--patch-km", type=float, default=10.0)
    parser.add_argument("--resolution-m", type=float, default=250.0)
    parser.add_argument("--max-cloud", type=float, default=100.0)
    parser.add_argument("--max-scenes-per-location", type=int, default=200)
    parser.add_argument("--s2-workers", type=int, default=2)
    parser.add_argument("--openmeteo-workers", type=int, default=1)
    parser.add_argument("--openmeteo-retries", type=int, default=8)
    parser.add_argument("--openmeteo-sleep-s", type=float, default=8.0)
    parser.add_argument("--openmeteo-continue-on-fail", action="store_true")
    parser.add_argument("--target-offset-days", type=float, default=5.0)
    parser.add_argument("--skip-torch-dataset", action="store_true")
    parser.add_argument("--out-dir", default="runs/cloud_temp_interaction")
    parser.add_argument("--resume", default="auto", help="auto, none, a checkpoint path, or a run directory.")
    parser.add_argument("--fresh", action="store_true", help="Start from scratch even if out-dir/last.pt exists.")
    parser.add_argument("--image-height", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--max-gap-days", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows; increase on Linux.")
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cloud-dim", type=int, default=256)
    parser.add_argument("--world-dim", type=int, default=192)
    parser.add_argument("--seq-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--no-gru", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--world-drop-prob", type=float, default=0.20, help="Randomly zero world features for some batches to force cloud branch learning.")
    parser.add_argument("--world-feature-drop-prob", type=float, default=0.08)
    parser.add_argument("--image-aux-weight", type=float, default=2.50, help="Strong image-only auxiliary loss to force the Sentinel-2 image branch to learn.")
    parser.add_argument("--cloud-aux-weight", type=float, default=1.00)
    parser.add_argument("--world-aux-weight", type=float, default=0.05)
    parser.add_argument("--cloud-scalar-drop-prob", type=float, default=0.25, help="Randomly zero Sentinel-derived scalar cloud features so the image path matters.")
    parser.add_argument("--cache-images", action="store_true", help="Cache resized mask images in RAM for faster epochs.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile only if Triton/Inductor is available. Otherwise it safely skips.")
    parser.add_argument("--force-compile", action="store_true", help="Force torch.compile even if the safety check says Triton is unavailable.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last conv weights for speed on modern GPUs.")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    data_root = maybe_build_dataset(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = resolve_resume_checkpoint(args.resume, out_dir, args.fresh)
    resume_state = torch.load(resume_path, map_location="cpu") if resume_path is not None else None

    metadata = load_metadata(data_root)
    raw_names = list(metadata["raw_feature_names"])
    cloud_names = list(metadata["cloud_feature_names"])
    world_names = list(metadata["world_feature_names"])

    train_records = load_records(data_root, "train")
    val_records = load_records(data_root, "val")
    test_records = load_records(data_root, "test")

    if resume_state is not None:
        x_norm = Normalizer.from_state_dict(resume_state["normalizer"])
        y_norm = TargetNormalizer.from_state_dict(resume_state["target_normalizer"])
        print(f"Resuming first model from {resume_path}")
    else:
        x_norm = Normalizer.fit(collect_train_feature_matrix(train_records, raw_names))
        y_norm = TargetNormalizer.fit(get_target(r) for r in train_records)
        print("Starting first model training from scratch.")

    train_ds = CloudTempSequenceDataset(data_root, train_records, raw_names, cloud_names, world_names, x_norm, y_norm, args.image_height, args.image_width, args.lookback, args.max_gap_days, args.augment, cache_images=args.cache_images)
    val_ds = CloudTempSequenceDataset(data_root, val_records, raw_names, cloud_names, world_names, x_norm, y_norm, args.image_height, args.image_width, args.lookback, args.max_gap_days, False, cache_images=args.cache_images)
    test_ds = CloudTempSequenceDataset(data_root, test_records, raw_names, cloud_names, world_names, x_norm, y_norm, args.image_height, args.image_width, args.lookback, args.max_gap_days, False, cache_images=args.cache_images)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = CloudWorldInteractionModel(
        num_cloud_features=len(cloud_names),
        num_world_features=len(world_names),
        cloud_dim=args.cloud_dim,
        world_dim=args.world_dim,
        seq_dim=args.seq_dim,
        dropout=args.dropout,
        use_gru=not args.no_gru,
    ).to(device)

    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if args.compile:
        ok_compile, compile_reason = can_use_torch_compile()
        if ok_compile or args.force_compile:
            print(f"Using torch.compile: {compile_reason}")
            # reduce-overhead avoids the heaviest max-autotune startup stall.
            model = torch.compile(model, mode="reduce-overhead")
        else:
            print(f"[WARN] Skipping torch.compile: {compile_reason}")
            print("[WARN] Continue in eager mode. Remove --compile to hide this warning.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    best_mae = float("inf")
    history = []
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        if "optimizer_state" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        history_path = out_dir / "history.json"
        if history_path.exists():
            try:
                history = list(json.loads(history_path.read_text(encoding="utf-8")).get("history", []))
            except Exception:
                history = []
        best_path = out_dir / "best.pt"
        if best_path.exists():
            try:
                best_state = torch.load(best_path, map_location="cpu")
                best_mae = float(best_state.get("metrics", {}).get("val", {}).get("mae_c", float("inf")))
            except Exception:
                best_mae = float("inf")
        if start_epoch > args.epochs:
            print(f"Checkpoint is already at epoch {start_epoch - 1}; configured --epochs is {args.epochs}.")
            return

    loss_fn = nn.SmoothL1Loss(beta=0.75)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(args.amp_dtype == "fp16" and device.type == "cuda"))

    print(json.dumps({
        "device": str(device),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_test_records": len(test_records),
        "num_train_windows": len(train_ds),
        "num_val_windows": len(val_ds),
        "num_test_windows": len(test_ds),
        "cloud_features": cloud_names,
        "world_features": world_names,
        "target_mean_c": y_norm.mean,
        "target_std_c": y_norm.std,
        "lookback": args.lookback,
        "image_aux_weight": args.image_aux_weight,
        "cloud_scalar_drop_prob": args.cloud_scalar_drop_prob,
        "cache_images": args.cache_images,
    }, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0
        total_loss = 0.0
        total_mae = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            mask = batch["mask"].to(device, non_blocking=True)
            cloud = batch["cloud_features"].to(device, non_blocking=True)
            world = batch["world_features"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            # Force robustness: sometimes world state is absent, so cloud branch must carry signal.
            if args.world_drop_prob > 0 and random.random() < args.world_drop_prob:
                world = torch.zeros_like(world)
            if args.world_feature_drop_prob > 0:
                drop = (torch.rand(world.shape[-1], device=device) < args.world_feature_drop_prob).float()
                world = world * (1.0 - drop.view(1, 1, -1))

            if args.cloud_scalar_drop_prob > 0:
                drop = (torch.rand(cloud.shape[-1], device=device) < args.cloud_scalar_drop_prob).float()
                cloud = cloud * (1.0 - drop.view(1, 1, -1))

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
                out = model(mask, cloud, world)
                loss_final = loss_fn(out["final"], target)
                loss_image = loss_fn(out["image"], target)
                loss_cloud = loss_fn(out["cloud"], target)
                loss_world = loss_fn(out["world"], target)
                loss = (
                    loss_final
                    + args.image_aux_weight * loss_image
                    + args.cloud_aux_weight * loss_cloud
                    + args.world_aux_weight * loss_world
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            with torch.no_grad():
                pred_raw = y_norm.inverse_tensor(out["final"].float())
                target_raw = y_norm.inverse_tensor(target.float())
                mae = (pred_raw - target_raw).abs().mean()

            bs = target.size(0)
            total += bs
            total_loss += float(loss.item()) * bs
            total_mae += float(mae.item()) * bs
            pbar.set_postfix(loss=total_loss / max(1, total), mae_c=total_mae / max(1, total))

        val = evaluate(model, val_loader, device, y_norm, amp_dtype)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total),
            "train_mae_c": total_mae / max(1, total),
            "val": val,
        }
        history.append(row)
        write_json(out_dir / "history.json", {"history": history})

        print(f"epoch={epoch:03d} train_mae={row['train_mae_c']:.3f}C val_mae={val['mae_c']:.3f}C val_rmse={val['rmse_c']:.3f}C corr={val['corr']:.3f}")

        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, args, metadata, x_norm, y_norm, row)

        if val["mae_c"] < best_mae:
            best_mae = val["mae_c"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, args, metadata, x_norm, y_norm, row)
            print(f"saved best.pt val_mae={best_mae:.3f}C")

    if len(test_ds) > 0 and (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test = evaluate(model, test_loader, device, y_norm, amp_dtype)
        write_json(out_dir / "test_metrics.json", test)
        print("test:", json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
