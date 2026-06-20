#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


ANGLE_FEATURE_NAMES = {"wind_direction_10m_dominant"}


def is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_dist() -> tuple[int, int, int, torch.device]:
    if is_dist():
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return rank, local_rank, world_size, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, device


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank0(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_records(data_root: Path, split: str) -> List[Dict[str, Any]]:
    split_path = data_root / "splits" / f"{split}.jsonl"
    if split_path.exists():
        return read_jsonl(split_path)

    all_rows = read_jsonl(data_root / "dataset.jsonl")
    all_rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("city", ""))))

    n = len(all_rows)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    if split == "train":
        return all_rows[:train_end]
    if split == "val":
        return all_rows[train_end:val_end]
    if split == "test":
        return all_rows[val_end:]

    raise ValueError(split)


def load_raw_feature_names(data_root: Path) -> List[str]:
    meta_path = data_root / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "raw_feature_names" in meta:
            return list(meta["raw_feature_names"])

    first = read_jsonl(data_root / "dataset.jsonl")[0]
    if "inputs" in first:
        return list(first["inputs"].keys())

    raise RuntimeError("Could not infer raw feature names. Use metadata.json or inputs dict.")


def expand_feature_names(raw_names: List[str]) -> List[str]:
    out = []
    for name in raw_names:
        if name in ANGLE_FEATURE_NAMES:
            out.append(name + "_sin")
            out.append(name + "_cos")
        else:
            out.append(name)
    return out


def extract_raw_features(record: Dict[str, Any], raw_names: List[str]) -> List[float]:
    if "inputs" in record:
        return [float(record["inputs"][name]) for name in raw_names]

    vec = record.get("feature_vector")
    if vec is None:
        raise KeyError("Sample has neither inputs nor feature_vector.")
    if len(vec) != len(raw_names):
        raise ValueError("feature_vector length does not match raw_feature_names.")
    return [float(x) for x in vec]


def transform_features(raw_values: List[float], raw_names: List[str]) -> List[float]:
    out = []
    for value, name in zip(raw_values, raw_names):
        if name in ANGLE_FEATURE_NAMES:
            radians = math.radians(value)
            out.append(math.sin(radians))
            out.append(math.cos(radians))
        else:
            out.append(value)
    return out


def get_target(record: Dict[str, Any]) -> float:
    if "target_temperature_c" in record:
        return float(record["target_temperature_c"])
    if "target" in record:
        return float(record["target"])
    raise KeyError("Missing target_temperature_c.")


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std[np.isnan(self.std)] = 1.0
        self.std[self.std == 0.0] = 1.0

    @classmethod
    def fit(cls, records: List[Dict[str, Any]], raw_names: List[str]) -> "Normalizer":
        rows = []
        for r in records:
            raw = extract_raw_features(r, raw_names)
            rows.append(transform_features(raw, raw_names))
        arr = np.asarray(rows, dtype=np.float32)
        return cls(arr.mean(axis=0), arr.std(axis=0))

    def transform(self, values: List[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return (arr - self.mean) / self.std

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class CloudDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        records: List[Dict[str, Any]],
        raw_names: List[str],
        normalizer: Normalizer,
        image_height: int,
        image_width: int,
        augment: bool,
    ):
        self.data_root = data_root
        self.records = records
        self.raw_names = raw_names
        self.normalizer = normalizer
        self.image_height = image_height
        self.image_width = image_width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def load_mask(self, rel_path: str) -> torch.Tensor:
        path = self.data_root / rel_path
        img = Image.open(path).convert("L")

        if img.size != (self.image_width, self.image_height):
            img = img.resize((self.image_width, self.image_height), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0

        if self.augment:
            if random.random() < 0.5:
                arr = np.ascontiguousarray(arr[:, ::-1])
            if random.random() < 0.25:
                noise = np.random.normal(0.0, 0.015, size=arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0.0, 1.0)

        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        raw = extract_raw_features(r, self.raw_names)
        processed = transform_features(raw, self.raw_names)
        x = self.normalizer.transform(processed)

        return {
            "mask": self.load_mask(r["mask_path"]),
            "features": torch.from_numpy(x),
            "target": torch.tensor([get_target(r)], dtype=torch.float32),
        }


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
    """
    Input:  [B, 1, 300, 480]
    Output: [B, 512]

    Spatial path:
      300x480
      150x240
      75x120
      38x60
      19x30
      10x15
    """

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
        fused = torch.cat([image_emb, tab_emb], dim=1)
        return self.head(fused)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> Dict[str, float]:
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_loss = 0.0
    n = 0
    loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="sum")

    for batch in loader:
        mask = batch["mask"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
            pred = model(mask, features)
            loss = loss_fn(pred, target)

        err = pred - target
        total_loss += float(loss.item())
        total_abs += float(err.abs().sum().item())
        total_sq += float((err ** 2).sum().item())
        n += target.shape[0]

    return {
        "loss": total_loss / max(1, n),
        "mae_c": total_abs / max(1, n),
        "rmse_c": math.sqrt(total_sq / max(1, n)),
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    raw_names: List[str],
    model_names: List[str],
    normalizer: Normalizer,
    image_height: int,
    image_width: int,
    metrics: Dict[str, Any],
) -> None:
    if isinstance(model, DDP):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    torch.save(
        {
            "model_state": state,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "raw_feature_names": raw_names,
            "model_feature_names": model_names,
            "normalizer": normalizer.state_dict(),
            "image_height": image_height,
            "image_width": image_width,
            "metrics": metrics,
            "architecture": "CloudTempDeepModel",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-height", type=int, default=300)
    parser.add_argument("--image-width", type=int, default=480)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--max-hours", type=float, default=7.8)
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_dist()
    seed_everything(args.seed, rank)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if rank0(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    raw_names = load_raw_feature_names(data_root)
    model_names = expand_feature_names(raw_names)

    train_records = load_records(data_root, "train")
    val_records = load_records(data_root, "val")
    test_records = load_records(data_root, "test")

    normalizer = Normalizer.fit(train_records, raw_names)

    train_ds = CloudDataset(data_root, train_records, raw_names, normalizer, args.image_height, args.image_width, args.augment)
    val_ds = CloudDataset(data_root, val_records, raw_names, normalizer, args.image_height, args.image_width, False)
    test_ds = CloudDataset(data_root, test_records, raw_names, normalizer, args.image_height, args.image_width, False)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = CloudTempDeepModel(num_features=len(model_names)).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = args.amp_dtype == "fp16" and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_scaler)

    if rank0(rank):
        print(json.dumps({
            "device": str(device),
            "world_size": world_size,
            "num_train": len(train_ds),
            "num_val": len(val_ds),
            "num_test": len(test_ds),
            "raw_features": raw_names,
            "model_features": model_names,
            "per_gpu_batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * world_size,
            "image_size": [args.image_height, args.image_width],
        }, indent=2))

    best_mae = float("inf")
    history = []
    start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    import time
    wall_start = time.time()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        total_abs = 0.0
        total_n = 0

        iterator = train_loader
        if rank0(rank):
            iterator = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)

        for batch in iterator:
            mask = batch["mask"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
                pred = model(mask, features)
                loss = loss_fn(pred, target)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            bs = target.shape[0]
            total_loss += float(loss.detach().item()) * bs
            total_abs += float((pred.detach() - target).abs().sum().item())
            total_n += bs

        val_metrics = evaluate(model, val_loader, device, amp_dtype)

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_n),
            "train_mae_c": total_abs / max(1, total_n),
            "val": val_metrics,
        }

        if rank0(rank):
            history.append(row)
            write_json(out_dir / "history.json", {"history": history})

            print(
                f"epoch={epoch:03d} "
                f"train_mae={row['train_mae_c']:.3f}C "
                f"val_mae={val_metrics['mae_c']:.3f}C "
                f"val_rmse={val_metrics['rmse_c']:.3f}C"
            )

            save_checkpoint(
                out_dir / "last.pt",
                model,
                optimizer,
                epoch,
                raw_names,
                model_names,
                normalizer,
                args.image_height,
                args.image_width,
                row,
            )

            if val_metrics["mae_c"] < best_mae:
                best_mae = val_metrics["mae_c"]
                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    raw_names,
                    model_names,
                    normalizer,
                    args.image_height,
                    args.image_width,
                    row,
                )
                print(f"saved best.pt val_mae={best_mae:.3f}C")

        elapsed_hours = (time.time() - wall_start) / 3600.0
        if elapsed_hours >= args.max_hours:
            if rank0(rank):
                print(f"Reached max-hours={args.max_hours}. Stopping cleanly.")
            break

    if rank0(rank) and len(test_ds) > 0:
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state"])
        test_metrics = evaluate(model, test_loader, device, amp_dtype)
        write_json(out_dir / "test_metrics.json", test_metrics)
        print("test:", json.dumps(test_metrics, indent=2))

    cleanup_dist()


if __name__ == "__main__":
    main()