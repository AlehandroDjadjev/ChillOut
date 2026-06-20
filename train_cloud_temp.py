#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_RAW_FEATURE_NAMES = [
    "s5p_cloud_fraction",
    "s5p_cloud_optical_thickness",
    "s5p_cloud_top_height_m",
    "s5p_cloud_top_pressure_pa",
    "s5p_aerosol_index",
    "s3_humidity_pct",
    "s3_sea_level_pressure_hpa",
    "s3_water_vapour_kg_m2",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "precipitation_sum",
    "cloud_cover_mean",
    "surface_pressure_mean",
]

ANGLE_FEATURE_NAMES = {"wind_direction_10m_dominant"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_feature_names(data_root: Path) -> List[str]:
    metadata_path = data_root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "raw_feature_names" in metadata:
            return list(metadata["raw_feature_names"])

    dataset_path = data_root / "dataset.jsonl"
    if dataset_path.exists():
        first = read_jsonl(dataset_path)[0]
        if "inputs" in first:
            return list(first["inputs"].keys())
        if "feature_vector" in first:
            return DEFAULT_RAW_FEATURE_NAMES[: len(first["feature_vector"])]

    return DEFAULT_RAW_FEATURE_NAMES


def expand_feature_names(raw_feature_names: List[str]) -> List[str]:
    out = []
    for name in raw_feature_names:
        if name in ANGLE_FEATURE_NAMES:
            out.append(f"{name}_sin")
            out.append(f"{name}_cos")
        else:
            out.append(name)
    return out


def extract_raw_features(record: Dict[str, Any], raw_feature_names: List[str]) -> List[float]:
    if "inputs" in record:
        inputs = record["inputs"]
        values = []
        for name in raw_feature_names:
            if name not in inputs:
                raise KeyError(f"Missing input feature {name!r} in sample {record.get('sample_id')}")
            values.append(float(inputs[name]))
        return values

    if "feature_vector" in record:
        vec = record["feature_vector"]
        if len(vec) != len(raw_feature_names):
            raise ValueError(
                f"feature_vector length {len(vec)} does not match raw_feature_names length {len(raw_feature_names)}"
            )
        return [float(x) for x in vec]

    raise KeyError("Record must contain either 'inputs' or 'feature_vector'.")


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


def target_from_record(record: Dict[str, Any]) -> float:
    if "target_temperature_c" in record:
        return float(record["target_temperature_c"])
    if "target" in record:
        return float(record["target"])
    raise KeyError(f"Missing target temperature in sample {record.get('sample_id')}")


def load_records_for_split(data_root: Path, split: str) -> List[Dict[str, Any]]:
    split_path = data_root / "splits" / f"{split}.jsonl"
    if split_path.exists():
        return read_jsonl(split_path)

    dataset_path = data_root / "dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find {split_path} or {dataset_path}")

    all_records = read_jsonl(dataset_path)
    all_records.sort(key=lambda r: (str(r.get("date", "")), str(r.get("city", ""))))

    n = len(all_records)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    if split == "train":
        return all_records[:train_end]
    if split == "val":
        return all_records[train_end:val_end]
    if split == "test":
        return all_records[val_end:]

    raise ValueError(f"Unknown split {split}")


class FeatureNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std[self.std == 0.0] = 1.0

    @classmethod
    def fit(cls, rows: np.ndarray) -> "FeatureNormalizer":
        mean = rows.mean(axis=0)
        std = rows.std(axis=0)
        std[np.isnan(std)] = 1.0
        std[std == 0.0] = 1.0
        return cls(mean, std)

    def transform(self, row: np.ndarray) -> np.ndarray:
        return (row.astype(np.float32) - self.mean) / self.std

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "FeatureNormalizer":
        return cls(np.array(state["mean"], dtype=np.float32), np.array(state["std"], dtype=np.float32))


class CloudTemperatureDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        records: List[Dict[str, Any]],
        raw_feature_names: List[str],
        normalizer: FeatureNormalizer,
        image_height: int = 300,
        image_width: int = 480,
        augment: bool = False,
    ):
        self.data_root = data_root
        self.records = records
        self.raw_feature_names = raw_feature_names
        self.normalizer = normalizer
        self.image_height = image_height
        self.image_width = image_width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _load_mask(self, mask_path: str) -> torch.Tensor:
        path = self.data_root / mask_path
        if not path.exists():
            raise FileNotFoundError(f"Mask not found: {path}")

        img = Image.open(path).convert("L")
        img = img.resize((self.image_width, self.image_height), Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0

        # Gentle augmentation only for training.
        # Horizontal flip is usually acceptable for learning cloud texture,
        # but disable it if your geography orientation must be preserved.
        if self.augment and random.random() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1])

        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        raw_values = extract_raw_features(record, self.raw_feature_names)
        processed_values = transform_raw_feature_row(raw_values, self.raw_feature_names)
        processed_values = np.array(processed_values, dtype=np.float32)
        features = self.normalizer.transform(processed_values)

        mask = self._load_mask(record["mask_path"])
        target = np.array([target_from_record(record)], dtype=np.float32)

        return {
            "mask": mask,
            "features": torch.from_numpy(features),
            "target": torch.from_numpy(target),
            "sample_id": record.get("sample_id", str(idx)),
            "city": record.get("city", ""),
            "date": record.get("date", ""),
        }


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
    """
    Input:  [B, 1, 300, 480]
    Output: [B, 256]
    """

    def __init__(self, image_embedding_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, dropout=0.02),     # 300x480 -> 150x240
            ConvBlock(32, 64, dropout=0.03),   # 150x240 -> 75x120
            ConvBlock(64, 128, dropout=0.05),  # 75x120 -> 37x60
            ConvBlock(128, 192, dropout=0.05), # 37x60 -> 18x30
            ConvBlock(192, 256, dropout=0.05), # 18x30 -> 9x15
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, image_embedding_dim),
            nn.LayerNorm(image_embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        x = self.cnn(mask)
        return self.proj(x)


class TabularEncoder(nn.Module):
    """
    Input:  [B, num_features]
    Output: [B, 128]
    """

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


class CloudTempModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        image_embedding_dim: int = 256,
        tab_embedding_dim: int = 128,
    ):
        super().__init__()

        self.image_encoder = CloudImageEncoder(image_embedding_dim=image_embedding_dim)
        self.tabular_encoder = TabularEncoder(
            num_features=num_features,
            tab_embedding_dim=tab_embedding_dim,
        )

        fusion_dim = image_embedding_dim + tab_embedding_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
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
        img_emb = self.image_encoder(mask)
        tab_emb = self.tabular_encoder(features)
        fused = torch.cat([img_emb, tab_emb], dim=1)
        return self.head(fused)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse_num = 0.0
    total_count = 0

    for batch in loader:
        mask = batch["mask"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            pred = model(mask, features)
            loss = loss_fn(pred, target)

        error = pred - target
        batch_size = target.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_mae += float(error.abs().sum().item())
        total_rmse_num += float((error ** 2).sum().item())
        total_count += batch_size

    return {
        "loss": total_loss / max(1, total_count),
        "mae_c": total_mae / max(1, total_count),
        "rmse_c": math.sqrt(total_rmse_num / max(1, total_count)),
    }


def build_normalizer(
    records: List[Dict[str, Any]],
    raw_feature_names: List[str],
) -> FeatureNormalizer:
    rows = []
    for record in records:
        raw = extract_raw_features(record, raw_feature_names)
        processed = transform_raw_feature_row(raw, raw_feature_names)
        rows.append(processed)

    arr = np.array(rows, dtype=np.float32)
    return FeatureNormalizer.fit(arr)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    raw_feature_names: List[str],
    model_feature_names: List[str],
    normalizer: FeatureNormalizer,
    image_height: int,
    image_width: int,
    metrics: Dict[str, Any],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "raw_feature_names": raw_feature_names,
        "model_feature_names": model_feature_names,
        "normalizer": normalizer.state_dict(),
        "image_height": image_height,
        "image_width": image_width,
        "metrics": metrics,
    }
    torch.save(payload, path)


def resolve_resume_checkpoint(out_dir: Path, resume: str) -> Optional[Path]:
    if resume in {"", "none", None}:
        return None

    if resume == "auto":
        for candidate in [out_dir / "last.pt", out_dir / "best.pt"]:
            if candidate.exists():
                return candidate
        return None

    path = Path(resume)
    if path.is_dir():
        for candidate in [path / "last.pt", path / "best.pt"]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No last.pt or best.pt found under {path}")

    if path.exists():
        return path

    raise FileNotFoundError(f"Resume checkpoint not found: {path}")


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_feature_names = load_feature_names(data_root)
    model_feature_names = expand_feature_names(raw_feature_names)

    train_records = load_records_for_split(data_root, "train")
    val_records = load_records_for_split(data_root, "val")
    test_records = load_records_for_split(data_root, "test")

    if len(train_records) == 0:
        raise RuntimeError("No training records found.")
    if len(val_records) == 0:
        print("WARNING: No validation records found. Using train split as validation.")
        val_records = train_records

    normalizer = build_normalizer(train_records, raw_feature_names)

    train_ds = CloudTemperatureDataset(
        data_root=data_root,
        records=train_records,
        raw_feature_names=raw_feature_names,
        normalizer=normalizer,
        image_height=args.image_height,
        image_width=args.image_width,
        augment=args.augment,
    )

    val_ds = CloudTemperatureDataset(
        data_root=data_root,
        records=val_records,
        raw_feature_names=raw_feature_names,
        normalizer=normalizer,
        image_height=args.image_height,
        image_width=args.image_width,
        augment=False,
    )

    test_ds = CloudTemperatureDataset(
        data_root=data_root,
        records=test_records,
        raw_feature_names=raw_feature_names,
        normalizer=normalizer,
        image_height=args.image_height,
        image_width=args.image_width,
        augment=False,
    ) if len(test_records) > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    model = CloudTempModel(num_features=len(model_feature_names)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Huber loss is usually more stable than MSE if some weather samples are noisy.
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    run_config = {
        "data_root": str(data_root),
        "image_height": args.image_height,
        "image_width": args.image_width,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "raw_feature_names": raw_feature_names,
        "model_feature_names": model_feature_names,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds) if test_ds is not None else 0,
    }
    write_json(out_dir / "run_config.json", run_config)

    print(json.dumps(run_config, indent=2))

    resume_ckpt = resolve_resume_checkpoint(out_dir, args.resume)
    history = []
    start_epoch = 1
    best_val_mae = float("inf")
    bad_epochs = 0

    if (out_dir / "history.json").exists():
        try:
            history = json.loads((out_dir / "history.json").read_text(encoding="utf-8")).get("history", [])
        except Exception:
            history = []

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_path = out_dir / "best.pt"
        if best_path.exists():
            try:
                best_ckpt = torch.load(best_path, map_location="cpu")
                best_val_mae = float(best_ckpt.get("metrics", {}).get("val", {}).get("mae_c", best_val_mae))
            except Exception:
                pass
        else:
            best_val_mae = float(ckpt.get("metrics", {}).get("val", {}).get("mae_c", best_val_mae))
        print(f"Resumed first-model training from {resume_ckpt} at epoch {start_epoch}.")
    else:
        print("Starting first-model training from scratch.")

    if start_epoch > args.epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}, which is beyond the configured total {args.epochs}.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in pbar:
            mask = batch["mask"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(mask, features)
                loss = loss_fn(pred, target)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            batch_size = target.shape[0]
            err = pred.detach() - target

            train_loss_sum += float(loss.item()) * batch_size
            train_mae_sum += float(err.abs().sum().item())
            train_count += batch_size

            pbar.set_postfix({
                "loss": train_loss_sum / max(1, train_count),
                "mae": train_mae_sum / max(1, train_count),
            })

        train_metrics = {
            "loss": train_loss_sum / max(1, train_count),
            "mae_c": train_mae_sum / max(1, train_count),
        }

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            use_amp=args.amp and device.type == "cuda",
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        write_json(out_dir / "history.json", {"history": history})

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_mae={train_metrics['mae_c']:.3f}C "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae_c']:.3f}C "
            f"val_rmse={val_metrics['rmse_c']:.3f}C"
        )

        save_checkpoint(
            out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            raw_feature_names=raw_feature_names,
            model_feature_names=model_feature_names,
            normalizer=normalizer,
            image_height=args.image_height,
            image_width=args.image_width,
            metrics=row,
        )

        if val_metrics["mae_c"] < best_val_mae:
            best_val_mae = val_metrics["mae_c"]
            bad_epochs = 0
            save_checkpoint(
                out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                raw_feature_names=raw_feature_names,
                model_feature_names=model_feature_names,
                normalizer=normalizer,
                image_height=args.image_height,
                image_width=args.image_width,
                metrics=row,
            )
            print(f"  saved best.pt with val_mae={best_val_mae:.3f}C")
        else:
            bad_epochs += 1

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(f"Early stopping after {bad_epochs} bad epochs.")
            break

    if test_loader is not None:
        best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            loss_fn=loss_fn,
            use_amp=args.amp and device.type == "cuda",
        )
        write_json(out_dir / "test_metrics.json", test_metrics)
        print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True, help="Dataset output folder from your extraction script.")
    parser.add_argument("--out-dir", default="runs/cloud_temp", help="Output folder for checkpoints.")

    parser.add_argument("--image-height", type=int, default=300)
    parser.add_argument("--image-width", type=int, default=480)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=1.0)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="auto", help="Resume from last.pt, best.pt, a checkpoint path, or none.")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
