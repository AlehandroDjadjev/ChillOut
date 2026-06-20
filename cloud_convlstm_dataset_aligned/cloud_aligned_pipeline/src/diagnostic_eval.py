from __future__ import annotations

import argparse
import csv
import json
import math
import heapq
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from channel_spec import TARGET_CHANNELS
from dataset import IndexedWeatherWindowDataset, WeatherWindowDataset, WindowConfig
from losses import WeightedSmoothL1Loss
from model import CloudForwardConvLSTM
from utils import get_device, load_yaml


class DatasetWithIndex(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return idx, x, y


class MetricAccumulator:
    def __init__(self, channels: int):
        self.channels = channels
        self.n_pixels = 0
        self.n_samples = 0

        self.sum_abs = np.zeros(channels, dtype=np.float64)
        self.sum_sq = np.zeros(channels, dtype=np.float64)
        self.sum_err = np.zeros(channels, dtype=np.float64)

        self.sum_pred = np.zeros(channels, dtype=np.float64)
        self.sum_targ = np.zeros(channels, dtype=np.float64)
        self.sum_pred2 = np.zeros(channels, dtype=np.float64)
        self.sum_targ2 = np.zeros(channels, dtype=np.float64)
        self.sum_cross = np.zeros(channels, dtype=np.float64)

        self.pred_spatial_std_sum = np.zeros(channels, dtype=np.float64)
        self.targ_spatial_std_sum = np.zeros(channels, dtype=np.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # pred/target: [B,C,H,W]
        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()

        b, c, h, w = pred.shape
        assert c == self.channels

        err = pred - target

        self.sum_abs += err.abs().sum(dim=(0, 2, 3)).numpy()
        self.sum_sq += (err ** 2).sum(dim=(0, 2, 3)).numpy()
        self.sum_err += err.sum(dim=(0, 2, 3)).numpy()

        self.sum_pred += pred.sum(dim=(0, 2, 3)).numpy()
        self.sum_targ += target.sum(dim=(0, 2, 3)).numpy()
        self.sum_pred2 += (pred ** 2).sum(dim=(0, 2, 3)).numpy()
        self.sum_targ2 += (target ** 2).sum(dim=(0, 2, 3)).numpy()
        self.sum_cross += (pred * target).sum(dim=(0, 2, 3)).numpy()

        self.pred_spatial_std_sum += pred.std(dim=(2, 3), unbiased=False).sum(dim=0).numpy()
        self.targ_spatial_std_sum += target.std(dim=(2, 3), unbiased=False).sum(dim=0).numpy()

        self.n_pixels += b * h * w
        self.n_samples += b

    def finalize(self) -> List[Dict[str, float]]:
        eps = 1e-12
        n = max(1, self.n_pixels)
        ns = max(1, self.n_samples)

        mae = self.sum_abs / n
        rmse = np.sqrt(np.maximum(self.sum_sq / n, 0.0))
        bias = self.sum_err / n

        cov_num = n * self.sum_cross - self.sum_pred * self.sum_targ
        cov_den = np.sqrt(
            np.maximum(n * self.sum_pred2 - self.sum_pred ** 2, 0.0)
            * np.maximum(n * self.sum_targ2 - self.sum_targ ** 2, 0.0)
        )
        corr = np.where(cov_den > eps, cov_num / np.maximum(cov_den, eps), np.nan)

        sst = self.sum_targ2 - (self.sum_targ ** 2) / n
        r2 = np.where(sst > eps, 1.0 - self.sum_sq / np.maximum(sst, eps), np.nan)

        pred_spatial_std = self.pred_spatial_std_sum / ns
        targ_spatial_std = self.targ_spatial_std_sum / ns

        rows = []
        for i in range(self.channels):
            rows.append(
                {
                    "mae": float(mae[i]),
                    "rmse": float(rmse[i]),
                    "bias": float(bias[i]),
                    "corr": float(corr[i]) if np.isfinite(corr[i]) else float("nan"),
                    "r2": float(r2[i]) if np.isfinite(r2[i]) else float("nan"),
                    "pred_spatial_std": float(pred_spatial_std[i]),
                    "target_spatial_std": float(targ_spatial_std[i]),
                }
            )
        return rows


class SensitivityAccumulator:
    def __init__(self, channels: int):
        self.channels = channels
        self.n_pixels = 0
        self.sum_abs_delta = np.zeros(channels, dtype=np.float64)
        self.sum_signed_delta = np.zeros(channels, dtype=np.float64)

    def update(self, base: torch.Tensor, changed: torch.Tensor) -> None:
        # base/changed: [B,C,H,W]
        delta = (changed.detach().float().cpu() - base.detach().float().cpu())
        b, c, h, w = delta.shape
        self.sum_abs_delta += delta.abs().sum(dim=(0, 2, 3)).numpy()
        self.sum_signed_delta += delta.sum(dim=(0, 2, 3)).numpy()
        self.n_pixels += b * h * w

    def finalize(self) -> List[Dict[str, float]]:
        n = max(1, self.n_pixels)
        return [
            {
                "mean_abs_prediction_delta": float(self.sum_abs_delta[i] / n),
                "mean_signed_prediction_delta": float(self.sum_signed_delta[i] / n),
            }
            for i in range(self.channels)
        ]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def denorm_y(x: torch.Tensor, y_mean: Optional[torch.Tensor], y_std: Optional[torch.Tensor]) -> torch.Tensor:
    if y_mean is None or y_std is None:
        return x
    return x * y_std.to(x.device) + y_mean.to(x.device)


def load_y_stats(data_dir: Path, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    stats_path = data_dir / "normalization_stats.npz"
    if not stats_path.exists():
        return None, None
    stats = np.load(stats_path)
    if "y_mean" not in stats or "y_std" not in stats:
        return None, None

    y_mean = torch.from_numpy(stats["y_mean"]).float().to(device)
    y_std = torch.from_numpy(stats["y_std"]).float().to(device)

    # Expected [1,C,1,1]. Keep robust if saved as [C].
    if y_mean.ndim == 1:
        y_mean = y_mean.view(1, -1, 1, 1)
    if y_std.ndim == 1:
        y_std = y_std.view(1, -1, 1, 1)
    return y_mean, y_std


def make_dataset(cfg: Dict[str, Any], split: str):
    win_cfg = WindowConfig(
        input_len=int(cfg["window"]["input_len"]),
        horizon=int(cfg["window"]["horizon"]),
        target_mode=cfg["window"]["target_mode"],
    )

    if "x_series" in cfg["data"]:
        index_key = {"train": "train_index", "val": "val_index", "test": "test_index"}[split]
        return IndexedWeatherWindowDataset(
            cfg["data"]["x_series"],
            cfg["data"]["y_series"],
            cfg["data"][index_key],
            target_mode=cfg["window"]["target_mode"],
        )

    x_key = f"{split}_x"
    y_key = f"{split}_y"
    if x_key not in cfg["data"]:
        # Fallback to val for older configs.
        x_key, y_key = "val_x", "val_y"
    return WeatherWindowDataset(cfg["data"][x_key], cfg["data"][y_key], win_cfg)


def make_model(cfg: Dict[str, Any], device: torch.device) -> CloudForwardConvLSTM:
    model = CloudForwardConvLSTM(
        input_channels=int(cfg["model"]["input_channels"]),
        output_channels=int(cfg["model"]["output_channels"]),
        encoder_channels=int(cfg["model"]["encoder_channels"]),
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint: str | Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    return ckpt if isinstance(ckpt, dict) else {}


def apply_variant(x: torch.Tensor, variant: str, cloud_indices: List[int], input_channels: int) -> torch.Tensor:
    x = x.clone()
    cloud_indices = [int(i) for i in cloud_indices if 0 <= int(i) < input_channels]
    non_cloud = [i for i in range(input_channels) if i not in set(cloud_indices)]

    if variant == "normal":
        return x

    if variant == "no_cloud":
        if cloud_indices:
            x[:, :, cloud_indices, :, :] = 0.0
        return x

    if variant == "cloud_shuffle":
        if x.size(0) > 1 and cloud_indices:
            perm = torch.randperm(x.size(0), device=x.device)
            shuffled = x[perm].clone()
            x[:, :, cloud_indices, :, :] = shuffled[:, :, cloud_indices, :, :]
        return x

    if variant == "no_noncloud":
        if non_cloud:
            x[:, :, non_cloud, :, :] = 0.0
        return x

    if variant == "all_zero_input":
        return torch.zeros_like(x)

    raise ValueError(f"Unknown variant: {variant}")


def get_persistence_prediction(ds: Dataset, sample_indices: Iterable[int], y_shape: torch.Size, device: torch.device) -> torch.Tensor:
    # For IndexedWeatherWindowDataset, use y_series at the last input timestep as persistence.
    if hasattr(ds, "rows") and hasattr(ds, "y_data"):
        preds = []
        for idx in sample_indices:
            row = ds.rows[int(idx)]
            x_end = int(row["x_end"])
            last_input_state = x_end - 1
            arr = np.asarray(ds.y_data[last_input_state], dtype=np.float32)
            preds.append(torch.from_numpy(arr))
        return torch.stack(preds, dim=0).to(device)

    # Fallback: no direct state index available. Use zeros/train mean baseline shape.
    return torch.zeros(y_shape, device=device)


def sample_metadata(ds: Dataset, idx: int) -> Dict[str, Any]:
    if hasattr(ds, "rows"):
        row = ds.rows[int(idx)]
        return {
            "window_index": int(idx),
            "location": row.get("location", ""),
            "target_timestamp": row.get("target_timestamp", ""),
            "x_start": row.get("x_start", ""),
            "x_end": row.get("x_end", ""),
            "y_start": row.get("y_start", ""),
            "y_end": row.get("y_end", ""),
            "split": row.get("split", ""),
        }
    return {"window_index": int(idx)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    ds: Dataset,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    cfg: Dict[str, Any],
    target_names: List[str],
    y_mean: Optional[torch.Tensor],
    y_std: Optional[torch.Tensor],
    out_dir: Path,
    split: str,
    worst_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model.eval()

    input_channels = int(cfg["model"]["input_channels"])
    output_channels = int(cfg["model"]["output_channels"])
    cloud_indices = cfg.get("ablation", {}).get("cloud_channel_indices", list(range(min(12, input_channels))))

    variants = ["normal", "no_cloud", "cloud_shuffle", "no_noncloud", "all_zero_input"]
    raw_acc = {v: MetricAccumulator(output_channels) for v in variants}
    norm_acc = {v: MetricAccumulator(output_channels) for v in variants}

    baseline_raw_acc = {
        "train_mean_zero_norm": MetricAccumulator(output_channels),
        "raw_zero_anomaly": MetricAccumulator(output_channels),
        "persistence_last_target": MetricAccumulator(output_channels),
    }
    baseline_norm_acc = {
        "train_mean_zero_norm": MetricAccumulator(output_channels),
        "raw_zero_anomaly": MetricAccumulator(output_channels),
        "persistence_last_target": MetricAccumulator(output_channels),
    }

    sensitivity_acc = {
        "normal_vs_no_cloud": SensitivityAccumulator(output_channels),
        "normal_vs_no_noncloud": SensitivityAccumulator(output_channels),
        "normal_vs_cloud_shuffle": SensitivityAccumulator(output_channels),
    }

    loss_totals: Dict[str, float] = {v: 0.0 for v in variants}
    loss_counts: Dict[str, int] = {v: 0 for v in variants}

    worst_heap: List[Tuple[float, Dict[str, Any]]] = []

    for batch_indices, x, y in tqdm(loader, desc=f"diagnostic {split}", leave=False):
        batch_indices_list = [int(i) for i in batch_indices.tolist()]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        preds_norm: Dict[str, torch.Tensor] = {}
        preds_raw: Dict[str, torch.Tensor] = {}

        for variant in variants:
            xv = apply_variant(x, variant, cloud_indices, input_channels)
            pred_norm = model(xv)
            pred_raw = denorm_y(pred_norm, y_mean, y_std)
            y_raw = denorm_y(y, y_mean, y_std)

            preds_norm[variant] = pred_norm
            preds_raw[variant] = pred_raw

            norm_acc[variant].update(pred_norm, y)
            raw_acc[variant].update(pred_raw, y_raw)

            loss = loss_fn(pred_norm, y)
            loss_totals[variant] += float(loss.item()) * x.size(0)
            loss_counts[variant] += x.size(0)

        # Baselines.
        y_raw = denorm_y(y, y_mean, y_std)

        zero_norm = torch.zeros_like(y)
        zero_norm_raw = denorm_y(zero_norm, y_mean, y_std)

        norm_acc_name = "train_mean_zero_norm"
        baseline_norm_acc[norm_acc_name].update(zero_norm, y)
        baseline_raw_acc[norm_acc_name].update(zero_norm_raw, y_raw)

        if y_mean is not None and y_std is not None:
            raw_zero_norm = (torch.zeros_like(y_raw) - y_mean.to(device)) / y_std.to(device)
        else:
            raw_zero_norm = torch.zeros_like(y)
        raw_zero_raw = denorm_y(raw_zero_norm, y_mean, y_std)

        baseline_norm_acc["raw_zero_anomaly"].update(raw_zero_norm, y)
        baseline_raw_acc["raw_zero_anomaly"].update(raw_zero_raw, y_raw)

        pers_norm = get_persistence_prediction(ds, batch_indices_list, y.shape, device)
        pers_raw = denorm_y(pers_norm, y_mean, y_std)
        baseline_norm_acc["persistence_last_target"].update(pers_norm, y)
        baseline_raw_acc["persistence_last_target"].update(pers_raw, y_raw)

        # Prediction sensitivity: how much outputs move when we damage inputs.
        sensitivity_acc["normal_vs_no_cloud"].update(preds_raw["normal"], preds_raw["no_cloud"])
        sensitivity_acc["normal_vs_no_noncloud"].update(preds_raw["normal"], preds_raw["no_noncloud"])
        sensitivity_acc["normal_vs_cloud_shuffle"].update(preds_raw["normal"], preds_raw["cloud_shuffle"])

        # Worst normal-model examples, raw scale.
        err = (preds_raw["normal"] - y_raw).abs()
        per_sample = err.mean(dim=(1, 2, 3)).detach().cpu().numpy()
        per_channel = err.mean(dim=(2, 3)).detach().cpu().numpy()

        for local_i, score in enumerate(per_sample):
            idx = batch_indices_list[local_i]
            meta = sample_metadata(ds, idx)
            row = {
                **meta,
                "mean_mae_raw": float(score),
            }
            for c, name in enumerate(target_names):
                row[f"mae_raw_{name}"] = float(per_channel[local_i, c])

            item = (float(score), row)
            if len(worst_heap) < worst_k:
                heapq.heappush(worst_heap, item)
            else:
                if item[0] > worst_heap[0][0]:
                    heapq.heapreplace(worst_heap, item)

    metrics_rows: List[Dict[str, Any]] = []

    def add_metric_rows(kind: str, name: str, raw_rows: List[Dict[str, float]], norm_rows: List[Dict[str, float]], loss: Optional[float] = None):
        for c, channel in enumerate(target_names):
            r = {
                "kind": kind,
                "name": name,
                "channel": channel,
                "loss_norm": "" if loss is None else float(loss),
                "mae_raw": raw_rows[c]["mae"],
                "rmse_raw": raw_rows[c]["rmse"],
                "bias_raw": raw_rows[c]["bias"],
                "corr_raw": raw_rows[c]["corr"],
                "r2_raw": raw_rows[c]["r2"],
                "pred_spatial_std_raw": raw_rows[c]["pred_spatial_std"],
                "target_spatial_std_raw": raw_rows[c]["target_spatial_std"],
                "mae_norm": norm_rows[c]["mae"],
                "rmse_norm": norm_rows[c]["rmse"],
                "bias_norm": norm_rows[c]["bias"],
                "corr_norm": norm_rows[c]["corr"],
                "r2_norm": norm_rows[c]["r2"],
            }
            metrics_rows.append(r)

    for variant in variants:
        loss_avg = loss_totals[variant] / max(1, loss_counts[variant])
        add_metric_rows("model_variant", variant, raw_acc[variant].finalize(), norm_acc[variant].finalize(), loss=loss_avg)

    for baseline in baseline_raw_acc:
        add_metric_rows("baseline", baseline, baseline_raw_acc[baseline].finalize(), baseline_norm_acc[baseline].finalize(), loss=None)

    sensitivity_rows: List[Dict[str, Any]] = []
    for name, acc in sensitivity_acc.items():
        rows = acc.finalize()
        for c, channel in enumerate(target_names):
            sensitivity_rows.append({"comparison": name, "channel": channel, **rows[c]})

    worst_rows = [row for _, row in sorted(worst_heap, key=lambda x: x[0], reverse=True)]

    summary = {
        "split": split,
        "num_windows": len(ds),
        "target_names": target_names,
        "cloud_channel_indices": cloud_indices,
        "model_losses": {
            v: loss_totals[v] / max(1, loss_counts[v])
            for v in variants
        },
        "files": {
            "metrics_csv": str(out_dir / "metrics_by_channel.csv"),
            "sensitivity_csv": str(out_dir / "prediction_sensitivity.csv"),
            "worst_csv": str(out_dir / "worst_windows.csv"),
        },
    }

    return metrics_rows, sensitivity_rows, {"worst_rows": worst_rows, "summary": summary}


@torch.no_grad()
def channel_importance(
    model: torch.nn.Module,
    ds: Dataset,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    cfg: Dict[str, Any],
    target_names: List[str],
    y_mean: Optional[torch.Tensor],
    y_std: Optional[torch.Tensor],
    max_batches: int,
) -> List[Dict[str, Any]]:
    """Zero one input channel at a time and report loss/MAE delta.

    Positive delta means removing that input channel made the model worse.
    Negative delta means the channel may be noisy/harmful for this checkpoint/dataset.
    """
    model.eval()

    input_names = cfg.get("channels", {}).get("input", [f"input_{i}" for i in range(int(cfg["model"]["input_channels"]))])
    input_channels = int(cfg["model"]["input_channels"])
    output_channels = int(cfg["model"]["output_channels"])

    # First compute normal aggregate over the capped batches.
    normal_acc = MetricAccumulator(output_channels)
    normal_loss_total = 0.0
    seen = 0
    cached_batches = []

    for b_idx, (batch_indices, x, y) in enumerate(tqdm(loader, desc="channel importance baseline", leave=False)):
        if max_batches > 0 and b_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        normal_loss_total += float(loss_fn(pred, y).item()) * x.size(0)
        seen += x.size(0)
        normal_acc.update(denorm_y(pred, y_mean, y_std), denorm_y(y, y_mean, y_std))
        cached_batches.append((x.detach().cpu(), y.detach().cpu()))

    normal_loss = normal_loss_total / max(1, seen)
    normal_mae = np.array([r["mae"] for r in normal_acc.finalize()], dtype=np.float64)

    rows: List[Dict[str, Any]] = []
    for ch in tqdm(range(input_channels), desc="zero-one-channel", leave=False):
        acc = MetricAccumulator(output_channels)
        loss_total = 0.0
        count = 0

        for x_cpu, y_cpu in cached_batches:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            x[:, :, ch, :, :] = 0.0
            pred = model(x)
            loss_total += float(loss_fn(pred, y).item()) * x.size(0)
            count += x.size(0)
            acc.update(denorm_y(pred, y_mean, y_std), denorm_y(y, y_mean, y_std))

        loss = loss_total / max(1, count)
        mae = np.array([r["mae"] for r in acc.finalize()], dtype=np.float64)

        row = {
            "input_channel_index": ch,
            "input_channel_name": input_names[ch] if ch < len(input_names) else f"input_{ch}",
            "normal_loss": normal_loss,
            "zeroed_loss": loss,
            "loss_delta_zeroed_minus_normal": loss - normal_loss,
            "mean_mae_delta_zeroed_minus_normal": float(np.nanmean(mae - normal_mae)),
        }
        for c, target_name in enumerate(target_names):
            row[f"mae_delta_{target_name}"] = float(mae[c] - normal_mae[c])
        rows.append(row)

    rows.sort(key=lambda r: r["loss_delta_zeroed_minus_normal"], reverse=True)
    return rows


def save_worst_plots(
    model: torch.nn.Module,
    ds: Dataset,
    worst_rows: List[Dict[str, Any]],
    device: torch.device,
    target_names: List[str],
    y_mean: Optional[torch.Tensor],
    y_std: Optional[torch.Tensor],
    out_dir: Path,
    max_plots: int,
) -> None:
    if max_plots <= 0:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not installed; skipping diagnostic plots.")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for rank, row in enumerate(worst_rows[:max_plots], start=1):
        idx = int(row["window_index"])
        x, y = ds[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)

        pred_raw = denorm_y(pred, y_mean, y_std)[0].detach().cpu().numpy()
        y_raw = denorm_y(y, y_mean, y_std)[0].detach().cpu().numpy()
        err = pred_raw - y_raw

        loc = str(row.get("location", "unknown"))
        ts = str(row.get("target_timestamp", "unknown")).replace(":", "").replace("+", "_")

        for c, name in enumerate(target_names):
            fig = plt.figure(figsize=(12, 4))

            ax1 = fig.add_subplot(1, 3, 1)
            im1 = ax1.imshow(y_raw[c])
            ax1.set_title("target")
            ax1.axis("off")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            ax2 = fig.add_subplot(1, 3, 2)
            im2 = ax2.imshow(pred_raw[c])
            ax2.set_title("prediction")
            ax2.axis("off")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            ax3 = fig.add_subplot(1, 3, 3)
            im3 = ax3.imshow(err[c])
            ax3.set_title("pred - target")
            ax3.axis("off")
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            fig.suptitle(f"rank={rank} idx={idx} {loc} {ts} {name}")
            fig.tight_layout()

            safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
            fig.savefig(plot_dir / f"worst_{rank:02d}_idx_{idx}_{safe_name}.png", dpi=130)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep diagnostic evaluation for cloud ConvLSTM checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default=None, help="Default: diagnostics/<split>")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows unless you know multiprocessing works.")
    parser.add_argument("--worst-k", type=int, default=25)
    parser.add_argument("--plot-worst", type=int, default=8)
    parser.add_argument("--channel-importance", action="store_true", help="Run slower one-input-channel-at-a-time zeroing diagnostic.")
    parser.add_argument("--channel-importance-batches", type=int, default=20, help="Cap batches for channel importance. 0 means all batches.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    target_names = cfg.get("channels", {}).get("target", TARGET_CHANNELS)
    ckpt_path = Path(args.checkpoint or cfg["training"]["checkpoint_path"])

    out_dir = Path(args.out_dir or f"diagnostics/{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg["training"]["device"])
    print(f"device={device}")

    ds = make_dataset(cfg, args.split)
    wrapped = DatasetWithIndex(ds)

    batch_size = int(args.batch_size or cfg["training"]["batch_size"])
    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = make_model(cfg, device)
    ckpt = load_checkpoint(model, ckpt_path, device)
    print(f"checkpoint={ckpt_path}")
    if "epoch" in ckpt:
        print(f"checkpoint_epoch={ckpt['epoch']}")
    if "val_loss" in ckpt:
        print(f"checkpoint_val_loss={ckpt['val_loss']}")

    data_dir = Path(cfg["data"]["x_series"]).parent if "x_series" in cfg["data"] else Path(".")
    y_mean, y_std = load_y_stats(data_dir, device)
    if y_mean is None:
        print("[WARN] normalization_stats.npz not found; raw metrics will equal normalized metrics.")

    loss_fn = WeightedSmoothL1Loss(cfg["loss"]["channel_weights"]).to(device)

    metrics_rows, sensitivity_rows, extra = evaluate(
        model=model,
        ds=ds,
        loader=loader,
        loss_fn=loss_fn,
        device=device,
        cfg=cfg,
        target_names=target_names,
        y_mean=y_mean,
        y_std=y_std,
        out_dir=out_dir,
        split=args.split,
        worst_k=int(args.worst_k),
    )

    worst_rows = extra["worst_rows"]
    summary = extra["summary"]

    write_csv(out_dir / "metrics_by_channel.csv", metrics_rows)
    write_csv(out_dir / "prediction_sensitivity.csv", sensitivity_rows)
    write_csv(out_dir / "worst_windows.csv", worst_rows)

    if args.channel_importance:
        importance_rows = channel_importance(
            model=model,
            ds=ds,
            loader=loader,
            loss_fn=loss_fn,
            device=device,
            cfg=cfg,
            target_names=target_names,
            y_mean=y_mean,
            y_std=y_std,
            max_batches=int(args.channel_importance_batches),
        )
        write_csv(out_dir / "input_channel_importance.csv", importance_rows)
        summary["files"]["input_channel_importance_csv"] = str(out_dir / "input_channel_importance.csv")

    save_worst_plots(
        model=model,
        ds=ds,
        worst_rows=worst_rows,
        device=device,
        target_names=target_names,
        y_mean=y_mean,
        y_std=y_std,
        out_dir=out_dir,
        max_plots=int(args.plot_worst),
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print(f"wrote: {out_dir / 'summary.json'}")
    print(f"wrote: {out_dir / 'metrics_by_channel.csv'}")
    print(f"wrote: {out_dir / 'prediction_sensitivity.csv'}")
    print(f"wrote: {out_dir / 'worst_windows.csv'}")
    if args.channel_importance:
        print(f"wrote: {out_dir / 'input_channel_importance.csv'}")
    if int(args.plot_worst) > 0:
        print(f"plots: {out_dir / 'plots'}")

    # Compact console readout.
    print("\nModel variant losses:")
    for name, loss in summary["model_losses"].items():
        print(f"  {name:<16} {loss:.6f}")

    print("\nRead these first:")
    print("  1) metrics_by_channel.csv: model vs baselines, MAE/RMSE/bias/corr/R2 per target")
    print("  2) prediction_sensitivity.csv: how much predictions move when clouds/non-clouds are damaged")
    print("  3) worst_windows.csv + plots/: where the model fails hardest")
    if args.channel_importance:
        print("  4) input_channel_importance.csv: which input channels actually help")


if __name__ == "__main__":
    main()
