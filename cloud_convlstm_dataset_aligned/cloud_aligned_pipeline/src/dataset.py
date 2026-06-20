from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import csv
import numpy as np
import torch
from torch.utils.data import Dataset

TargetMode = Literal["mean", "sequence"]


@dataclass(frozen=True)
class WindowConfig:
    input_len: int
    horizon: int
    target_mode: TargetMode = "mean"


class WeatherWindowDataset(Dataset):
    """Backward-compatible contiguous time-series dataset.

    x_data: [N_time, C_in, H, W]
    y_data: [N_time, C_out, H, W]

    This class is fine for a single continuous location/time series.
    For multiple locations, prefer IndexedWeatherWindowDataset so windows never
    cross location boundaries.
    """

    def __init__(
        self,
        x_path: str | Path,
        y_path: str | Path,
        cfg: WindowConfig,
        mmap_mode: str | None = "r",
        require_finite: bool = True,
    ) -> None:
        self.x_data = np.load(x_path, mmap_mode=mmap_mode)
        self.y_data = np.load(y_path, mmap_mode=mmap_mode)
        self.cfg = cfg
        self.require_finite = require_finite

        if self.x_data.ndim != 4:
            raise ValueError(f"x_data must be [N_time,C,H,W], got {self.x_data.shape}")
        if self.y_data.ndim != 4:
            raise ValueError(f"y_data must be [N_time,C,H,W], got {self.y_data.shape}")
        if self.x_data.shape[0] != self.y_data.shape[0]:
            raise ValueError("x_data and y_data must have equal N_time")

        self.valid_start = cfg.input_len
        self.valid_end = self.x_data.shape[0] - cfg.horizon
        if self.valid_end <= self.valid_start:
            raise ValueError("Not enough timesteps for input_len + horizon")

    def __len__(self) -> int:
        return self.valid_end - self.valid_start

    def __getitem__(self, idx: int):
        t = self.valid_start + idx
        x = np.asarray(self.x_data[t - self.cfg.input_len : t], dtype=np.float32)
        future = np.asarray(self.y_data[t : t + self.cfg.horizon], dtype=np.float32)
        y = _make_target(future, self.cfg.target_mode)

        if self.require_finite:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(x), torch.from_numpy(y)


class IndexedWeatherWindowDataset(Dataset):
    """Location-safe dataset for downloader-built tensors.

    x_series: [N_state, C_in, H, W]
    y_series: [N_state, C_out, H, W]

    window_index_csv columns:
      x_start, x_end, y_start, y_end, location, split

    x_start/x_end are Python slice bounds. y_start/y_end are future target
    slice bounds. This avoids accidental windows crossing from one location
    into another when multiple places are concatenated in one tensor.
    """

    def __init__(
        self,
        x_series_path: str | Path,
        y_series_path: str | Path,
        window_index_csv: str | Path,
        target_mode: TargetMode = "mean",
        mmap_mode: str | None = "r",
        require_finite: bool = True,
    ) -> None:
        self.x_data = np.load(x_series_path, mmap_mode=mmap_mode)
        self.y_data = np.load(y_series_path, mmap_mode=mmap_mode)
        self.target_mode = target_mode
        self.require_finite = require_finite

        if self.x_data.ndim != 4:
            raise ValueError(f"x_series must be [N_state,C,H,W], got {self.x_data.shape}")
        if self.y_data.ndim != 4:
            raise ValueError(f"y_series must be [N_state,C,H,W], got {self.y_data.shape}")
        if self.x_data.shape[0] != self.y_data.shape[0]:
            raise ValueError("x_series and y_series must have equal N_state")

        self.rows: list[dict[str, str]] = []
        with open(window_index_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No rows found in window index: {window_index_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x_start = int(row["x_start"])
        x_end = int(row["x_end"])
        y_start = int(row["y_start"])
        y_end = int(row["y_end"])

        x = np.asarray(self.x_data[x_start:x_end], dtype=np.float32)
        future = np.asarray(self.y_data[y_start:y_end], dtype=np.float32)
        y = _make_target(future, self.target_mode)

        if self.require_finite:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(x), torch.from_numpy(y)


def _make_target(future: np.ndarray, target_mode: TargetMode) -> np.ndarray:
    if target_mode == "mean":
        return future.mean(axis=0)
    if target_mode == "sequence":
        return future
    raise ValueError(f"Unknown target_mode={target_mode}")
