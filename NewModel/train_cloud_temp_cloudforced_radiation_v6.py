#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# IO / normalization
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_metadata(root: Path) -> Dict[str, Any]:
    p = root / "metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_records(root: Path, split: str) -> List[Dict[str, Any]]:
    p = root / "splits" / f"{split}.jsonl"
    if p.exists():
        return read_jsonl(p)
    rows = read_jsonl(root / "dataset.jsonl")
    rows.sort(key=lambda r: (str(r.get("date", "")), str(r.get("location", r.get("city", "")))))
    n = len(rows); a = int(0.70*n); b = int(0.85*n)
    if split == "train": return rows[:a]
    if split == "val": return rows[a:b]
    if split == "test": return rows[b:]
    raise ValueError(split)


def parse_dt(r: Dict[str, Any]) -> np.datetime64:
    s = str(r.get("anchor") or r.get("date") or "").replace("Z", "")
    if "+" in s: s = s.split("+", 1)[0]
    return np.datetime64(s)


def get_target_temp(r: Dict[str, Any]) -> float:
    if "target_temperature_c" in r: return float(r["target_temperature_c"])
    if "target" in r: return float(r["target"])
    raise KeyError("missing target_temperature_c")


def get_current_temp(r: Dict[str, Any]) -> float:
    if r.get("current_temperature_c") is not None:
        return float(r["current_temperature_c"])
    inputs = r.get("inputs") or {}
    if "world_temperature_2m" in inputs:
        return float(inputs["world_temperature_2m"])
    raise KeyError("missing current_temperature_c; residual model needs current temp anchor")


def days_between(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    da = parse_dt(a)
    db = parse_dt(b)
    try:
        return float((da - db) / np.timedelta64(1, "D"))
    except Exception:
        return 0.0


def get_target_offset_days(record: Dict[str, Any]) -> float:
    raw = record.get("target_offset_days")
    if raw is not None:
        try:
            return float(raw)
        except Exception:
            pass
    if record.get("target_timestamp") and (record.get("anchor") or record.get("date")):
        try:
            target_dt = np.datetime64(str(record["target_timestamp"]).replace("Z", ""))
            anchor_dt = np.datetime64(str(record.get("anchor") or record.get("date")).replace("Z", ""))
            return float((target_dt - anchor_dt) / np.timedelta64(1, "D"))
        except Exception:
            pass
    return 5.0


def clipped(value: float, limit: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(-limit, min(limit, value)))


def trend_bundle(records: List[Dict[str, Any]], clip_c: float = 18.0) -> Dict[str, Any]:
    """Derive temperature-change information from the existing lookback window.

    This requires no dataset rebuild: every record already has current_temperature_c.

    Returns fixed-scale trend features for the trend LSTM and two simple baselines:
      previous_snapshot_trend: extrapolate the last observed temp change
      window_linear_trend: fit a linear slope over the lookback window
    """
    last = records[-1]
    temps = np.asarray([get_current_temp(r) for r in records], dtype=np.float32)
    days = np.asarray([days_between(r, last) for r in records], dtype=np.float32)  # <= 0, last = 0
    target_offset = get_target_offset_days(last)

    # step changes measured inside the lookback window
    step_temp = np.zeros_like(temps, dtype=np.float32)
    step_days = np.zeros_like(days, dtype=np.float32)
    if len(temps) > 1:
        step_temp[1:] = temps[1:] - temps[:-1]
        step_days[1:] = np.maximum(0.0, days[1:] - days[:-1])

    # fixed scaling, no fitted normalizer needed
    trend_features = np.stack([
        (temps - temps[-1]) / 10.0,
        days / 10.0,
        step_temp / 10.0,
        step_days / 10.0,
    ], axis=1).astype(np.float32)

    prev_delta = 0.0
    if len(temps) >= 2:
        dt = max(0.25, float(days[-1] - days[-2]))
        slope = float((temps[-1] - temps[-2]) / dt)
        prev_delta = clipped(slope * target_offset, clip_c)

    linear_delta = 0.0
    if len(temps) >= 2 and float(np.std(days)) > 1e-6:
        try:
            slope, intercept = np.polyfit(days.astype(np.float64), temps.astype(np.float64), 1)
            linear_delta = clipped(float(slope) * target_offset, clip_c)
        except Exception:
            linear_delta = 0.0

    return {
        "trend_features": trend_features,
        "previous_snapshot_trend_delta_c": float(prev_delta),
        "window_linear_trend_delta_c": float(linear_delta),
        "target_offset_days": float(target_offset),
    }


def solar_clear_sky_proxy_wm2(lat: float, lon: float, record: Dict[str, Any]) -> float:
    """Simple clear-sky shortwave proxy from record anchor/date.

    Used as fallback for old datasets that do not yet contain radiation_* fields.
    """
    import datetime as _dt
    raw = str(record.get("anchor") or record.get("date") or "")
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        dt = dt.astimezone(_dt.timezone.utc)
    except Exception:
        return 0.0

    day = int(dt.strftime("%j"))
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.001480 * math.sin(3 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    true_solar_time_min = hour * 60.0 + eqtime + 4.0 * float(lon)
    hour_angle = math.radians(true_solar_time_min / 4.0 - 180.0)
    lat_rad = math.radians(float(lat))
    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    if cos_zenith <= 0.0:
        return 0.0
    ecc = 1.0 + 0.033 * math.cos(2.0 * math.pi * day / 365.0)
    return float(max(0.0, 1361.0 * ecc * cos_zenith * 0.72))


def radiation_bundle(record: Dict[str, Any]) -> Dict[str, float]:
    """Return cloud-centered radiation target for the scene time.

    New v6 datasets store radiation_* fields directly. For older datasets,
    derive them from world_shortwave_radiation + solar clear-sky proxy.
    """
    if "radiation_cloud_loss_wm2" in record:
        return {
            "loss_wm2": float(record.get("radiation_cloud_loss_wm2", 0.0)),
            "attenuation": float(record.get("radiation_cloud_attenuation", 0.0)),
            "observed_wm2": float(record.get("radiation_shortwave_observed_wm2", 0.0)),
            "clear_wm2": float(record.get("radiation_clear_sky_proxy_wm2", 0.0)),
            "valid": float(record.get("radiation_daylight_valid", 0.0)),
        }

    inputs = record.get("inputs") or {}
    observed = float(inputs.get("world_shortwave_radiation", 0.0))
    clear = solar_clear_sky_proxy_wm2(float(record.get("lat", 0.0)), float(record.get("lon", 0.0)), record)
    valid = 1.0 if clear >= 50.0 else 0.0
    if valid:
        trans = max(0.0, min(1.5, observed / max(clear, 1e-6)))
        attenuation = max(-0.5, min(1.2, 1.0 - trans))
        loss = max(-250.0, min(1200.0, clear - observed))
    else:
        attenuation = 0.0
        loss = 0.0
    return {"loss_wm2": float(loss), "attenuation": float(attenuation), "observed_wm2": observed, "clear_wm2": clear, "valid": valid}


def feature_vector(r: Dict[str, Any], names: List[str]) -> np.ndarray:
    inputs = r.get("inputs")
    if inputs:
        return np.asarray([float(inputs[n]) for n in names], dtype=np.float32)
    raw = r.get("feature_vector")
    if raw is None:
        raise KeyError("record missing inputs/feature_vector")
    return np.asarray(raw, dtype=np.float32)


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std[~np.isfinite(self.std)] = 1.0
        self.std[self.std == 0.0] = 1.0
    @classmethod
    def fit(cls, arr: np.ndarray) -> "Normalizer":
        return cls(arr.mean(0), arr.std(0))
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float32) - self.mean) / self.std
    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "Normalizer":
        return cls(np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32))


class TargetNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if math.isfinite(float(std)) and float(std) != 0 else 1.0
    @classmethod
    def fit(cls, vals: Iterable[float]) -> "TargetNormalizer":
        a = np.asarray(list(vals), dtype=np.float32)
        return cls(float(a.mean()), float(a.std()))
    def transform_tensor(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std
    def inverse_tensor(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean
    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}
    @classmethod
    def from_state_dict(cls, d: Dict[str, float]) -> "TargetNormalizer":
        return cls(float(d["mean"]), float(d["std"]))


@dataclass
class Window:
    records: List[Dict[str, Any]]


def build_windows(records: List[Dict[str, Any]], lookback: int, max_gap_days: float) -> List[Window]:
    by_loc: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_loc.setdefault(str(r.get("location", r.get("city", ""))), []).append(r)
    max_gap = np.timedelta64(int(round(max_gap_days * 24)), "h")
    wins: List[Window] = []
    for rows in by_loc.values():
        rows.sort(key=parse_dt)
        if lookback <= 1:
            wins.extend(Window([r]) for r in rows); continue
        for i in range(lookback - 1, len(rows)):
            chunk = rows[i-lookback+1:i+1]
            if all(parse_dt(b) - parse_dt(a) <= max_gap for a, b in zip(chunk[:-1], chunk[1:])):
                wins.append(Window(chunk))
    return wins


class CloudTempResidualSequenceDataset(Dataset):
    def __init__(self, root: Path, records: List[Dict[str, Any]], raw_names: List[str],
                 cloud_names: List[str], world_names: List[str], x_norm: Normalizer,
                 delta_norm: TargetNormalizer, image_height: int, image_width: int,
                 lookback: int, max_gap_days: float, augment: bool, cache_images: bool = False):
        self.root = root; self.raw_names = raw_names; self.cloud_names = cloud_names; self.world_names = world_names
        self.x_norm = x_norm; self.delta_norm = delta_norm
        self.image_height = image_height; self.image_width = image_width; self.augment = augment
        raw_to_idx = {n:i for i,n in enumerate(raw_names)}
        self.cloud_idx = [raw_to_idx[n] for n in cloud_names]
        self.world_idx = [raw_to_idx[n] for n in world_names]
        self.windows = build_windows(records, lookback, max_gap_days)
        self.cache_images = cache_images
        self.cache: Dict[str, np.ndarray] = {}
        if cache_images:
            paths = sorted({str(r["mask_path"]) for w in self.windows for r in w.records})
            for p in tqdm(paths, desc="caching masks", leave=False):
                self.cache[p] = self._load_mask_array(p)

    def __len__(self) -> int:
        return len(self.windows)

    def _load_mask_array(self, rel: str) -> np.ndarray:
        img = Image.open(self.root / rel).convert("L")
        if img.size != (self.image_width, self.image_height):
            img = img.resize((self.image_width, self.image_height), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def load_mask(self, rel: str) -> torch.Tensor:
        arr = self.cache[rel].copy() if self.cache_images and rel in self.cache else self._load_mask_array(rel)
        if self.augment:
            if random.random() < 0.5: arr = np.ascontiguousarray(arr[:, ::-1])
            if random.random() < 0.20:
                arr = np.clip(arr + np.random.normal(0, 0.01, arr.shape).astype(np.float32), 0, 1)
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]
        masks, feats = [], []
        for r in win.records:
            masks.append(self.load_mask(str(r["mask_path"])))
            feats.append(self.x_norm.transform(feature_vector(r, self.raw_names)))
        x = np.stack(feats).astype(np.float32)
        last = win.records[-1]
        cur = get_current_temp(last)
        tgt = get_target_temp(last)
        delta = tgt - cur
        tb = trend_bundle(win.records)
        rb = radiation_bundle(last)
        return {
            "mask": torch.stack(masks, 0),
            "cloud_features": torch.from_numpy(x[:, self.cloud_idx]),
            "world_features": torch.from_numpy(x[:, self.world_idx]),
            "trend_features": torch.from_numpy(tb["trend_features"]),
            "target": torch.tensor([(delta - self.delta_norm.mean) / self.delta_norm.std], dtype=torch.float32),
            "target_delta_raw": torch.tensor([delta], dtype=torch.float32),
            "radiation_loss_raw": torch.tensor([rb["loss_wm2"]], dtype=torch.float32),
            "radiation_attenuation_raw": torch.tensor([rb["attenuation"]], dtype=torch.float32),
            "radiation_observed_wm2": torch.tensor([rb["observed_wm2"]], dtype=torch.float32),
            "radiation_clear_wm2": torch.tensor([rb["clear_wm2"]], dtype=torch.float32),
            "radiation_valid": torch.tensor([rb["valid"]], dtype=torch.float32),
            "target_raw": torch.tensor([tgt], dtype=torch.float32),
            "current_temp_raw": torch.tensor([cur], dtype=torch.float32),
            "previous_snapshot_trend_delta_raw": torch.tensor([tb["previous_snapshot_trend_delta_c"]], dtype=torch.float32),
            "window_linear_trend_delta_raw": torch.tensor([tb["window_linear_trend_delta_c"]], dtype=torch.float32),
            "target_offset_days": torch.tensor([tb["target_offset_days"]], dtype=torch.float32),
            "sample_id": last.get("sample_id", str(idx)),
            "location": last.get("location", last.get("city", "")),
            "anchor": last.get("anchor", last.get("date", "")),
        }


# -----------------------------
# Model
# -----------------------------

class ResBlock(nn.Module):
    def __init__(self, c: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
        )
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(x + self.net(x))


class Down(nn.Module):
    def __init__(self, a: int, b: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU(inplace=True),
            ResBlock(b, dropout),
        )
    def forward(self, x): return self.net(x)


class CloudStem(nn.Module):
    def __init__(self, out_ch: int = 64, dropout: float = 0.03):
        super().__init__()
        self.net = nn.Sequential(Down(1, 32, dropout), Down(32, 48, dropout), Down(48, out_ch, dropout))
    def forward(self, x): return self.net(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid: int, k: int = 3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k//2)
    def forward(self, x, state):
        if state is None:
            b, _, h, w = x.shape
            h0 = torch.zeros(b, self.hid, h, w, dtype=x.dtype, device=x.device)
            c0 = torch.zeros_like(h0)
        else:
            h0, c0 = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h0], 1)), 4, 1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c1 = f * c0 + i * g
        h1 = o * torch.tanh(c1)
        return h1, c1


class ConvLSTM(nn.Module):
    def __init__(self, in_ch: int, hidden: List[int], dropout: float):
        super().__init__()
        dims = [in_ch] + hidden
        self.cells = nn.ModuleList([ConvLSTMCell(dims[i], dims[i+1]) for i in range(len(hidden))])
        self.drop = nn.Dropout2d(dropout)
    def forward(self, x):
        # [B,T,C,H,W] -> last hidden map
        layer = x
        last = None
        for li, cell in enumerate(self.cells):
            state = None; outs = []
            for t in range(layer.size(1)):
                state = cell(layer[:, t], state)
                h, _ = state
                if li < len(self.cells) - 1: h = self.drop(h)
                outs.append(h)
            layer = torch.stack(outs, 1)
            last = outs[-1]
        return last


class AttnPool(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.score = nn.Conv2d(c, 1, 1)
        self.norm = nn.LayerNorm(c)
    def forward(self, x):
        b, c, h, w = x.shape
        a = torch.softmax(self.score(x).view(b, 1, h*w), -1)
        v = x.view(b, c, h*w)
        return self.norm(torch.bmm(v, a.transpose(1,2)).squeeze(-1))


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: List[int], out: int, dropout: float):
        super().__init__()
        layers = []; p = inp
        for h in hidden:
            layers += [nn.Linear(p,h), nn.LayerNorm(h), nn.SiLU(inplace=True), nn.Dropout(dropout)]
            p = h
        layers += [nn.Linear(p,out), nn.LayerNorm(out), nn.SiLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class DeltaHead(nn.Module):
    def __init__(self, inp: int, hidden: int = 96, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.SiLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.normal_(self.net[-1].weight, 0.0, 1e-3)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return self.net(x)


class ResidualTrendSplitConvLSTM(nn.Module):
    def __init__(self, num_cloud_features: int, num_world_features: int, stem_dim: int = 64,
                 convlstm_dims: Optional[List[int]] = None, cloud_scalar_dim: int = 128,
                 world_dim: int = 160, trend_dim: int = 96, fusion_dim: int = 192, dropout: float = 0.14,
                 delta_norm_limit: Optional[float] = None):
        super().__init__()
        convlstm_dims = convlstm_dims or [64, 96]
        self.delta_norm_limit = delta_norm_limit

        self.stem = CloudStem(stem_dim, dropout * 0.25)
        self.convlstm = ConvLSTM(stem_dim, convlstm_dims, dropout * 0.5)
        self.pool = AttnPool(convlstm_dims[-1])
        self.image_proj = MLP(convlstm_dims[-1], [fusion_dim], fusion_dim, dropout)

        self.cloud_lstm = nn.LSTM(num_cloud_features, cloud_scalar_dim, batch_first=True)
        self.cloud_scalar_proj = MLP(cloud_scalar_dim, [fusion_dim], fusion_dim, dropout)
        self.cloud_fuse = MLP(fusion_dim * 2, [fusion_dim], fusion_dim, dropout)

        self.world_lstm = nn.LSTM(num_world_features, world_dim, batch_first=True)
        self.world_proj = MLP(world_dim, [fusion_dim], fusion_dim, dropout)

        # Temperature-trend branch from the existing lookback temperatures.
        # This is not future leakage: it only sees current/past temperatures in the window.
        self.trend_lstm = nn.LSTM(4, trend_dim, batch_first=True)
        self.trend_proj = MLP(trend_dim, [fusion_dim], fusion_dim, dropout)
        self.context_fuse = MLP(fusion_dim * 2, [fusion_dim], fusion_dim, dropout)

        self.gate = nn.Sequential(nn.Linear(fusion_dim*2, fusion_dim), nn.LayerNorm(fusion_dim),
                                  nn.SiLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
        self.interaction = MLP(fusion_dim * 4, [fusion_dim, fusion_dim], fusion_dim, dropout)

        self.image_head = DeltaHead(fusion_dim, 96, dropout)
        self.cloud_head = DeltaHead(fusion_dim, 96, dropout)
        self.world_head = DeltaHead(fusion_dim, 96, dropout)
        self.trend_head = DeltaHead(fusion_dim, 96, dropout)
        self.interaction_head = DeltaHead(fusion_dim, 96, dropout)

        # Radiation heads are intentionally attached to the image/cloud states.
        # They force the Sentinel-2 cloud image branch to learn a target that
        # clouds physically control more directly than +5 day temperature.
        self.image_radiation_head = DeltaHead(fusion_dim, 96, dropout)
        self.cloud_radiation_head = DeltaHead(fusion_dim, 96, dropout)

        self.final_bias = nn.Parameter(torch.zeros(1))

    def encode_image(self, mask):
        b, t, c, h, w = mask.shape
        x = mask.reshape(b*t, c, h, w).contiguous(memory_format=torch.channels_last)
        fmap = self.stem(x)
        _, cc, hh, ww = fmap.shape
        seq = fmap.view(b, t, cc, hh, ww)
        return self.image_proj(self.pool(self.convlstm(seq)))

    def forward(self, mask, cloud_features, world_features, trend_features,
                disable_interaction=False, disable_image_delta=False,
                disable_cloud_delta=False, disable_world_delta=False, disable_trend_delta=False,
                context_scale: float = 0.15):
        image_state = self.encode_image(mask)
        _, (hc, _) = self.cloud_lstm(cloud_features)
        cloud_scalar = self.cloud_scalar_proj(hc[-1])
        cloud_state = self.cloud_fuse(torch.cat([image_state, cloud_scalar], -1))

        _, (hw, _) = self.world_lstm(world_features)
        world_state = self.world_proj(hw[-1])

        _, (ht, _) = self.trend_lstm(trend_features)
        trend_state = self.trend_proj(ht[-1])
        context_state = self.context_fuse(torch.cat([world_state, trend_state], -1))

        g = self.gate(torch.cat([cloud_state, context_state], -1))
        mixed = g * cloud_state + (1.0 - g) * context_state
        inter_state = self.interaction(torch.cat([cloud_state, context_state, cloud_state * context_state, mixed], -1))

        image_delta = self.image_head(image_state)
        cloud_delta = self.cloud_head(cloud_state)
        world_delta = self.world_head(world_state)
        trend_delta = self.trend_head(trend_state)
        interaction_delta = self.interaction_head(inter_state)

        if disable_image_delta: image_delta = torch.zeros_like(image_delta)
        if disable_cloud_delta: cloud_delta = torch.zeros_like(cloud_delta)
        if disable_world_delta: world_delta = torch.zeros_like(world_delta)
        if disable_trend_delta: trend_delta = torch.zeros_like(trend_delta)
        if disable_interaction: interaction_delta = torch.zeros_like(interaction_delta)

        image_radiation = self.image_radiation_head(image_state)
        cloud_radiation = self.cloud_radiation_head(cloud_state)

        # Cloud-first formula. World/trend/interaction are allowed to correct,
        # but cannot freely dominate unless context_scale is high.
        cloud_primary_delta = image_delta + cloud_delta
        context_delta = world_delta + trend_delta + interaction_delta
        final_delta = cloud_primary_delta + float(context_scale) * context_delta + self.final_bias
        if self.delta_norm_limit is not None and self.delta_norm_limit > 0:
            lim = float(self.delta_norm_limit)
            final_delta = lim * torch.tanh(final_delta / lim)

        return {
            "final_delta": final_delta,
            "image_delta": image_delta,
            "cloud_delta": cloud_delta,
            "world_delta": world_delta,
            "trend_delta": trend_delta,
            "interaction_delta": interaction_delta,
            "cloud_primary_delta": cloud_primary_delta,
            "context_delta": context_delta,
            "image_radiation": image_radiation,
            "cloud_radiation": cloud_radiation,
            "gate_mean": g.mean(-1, keepdim=True),
        }


CloudWorldInteractionModel = ResidualTrendSplitConvLSTM
ResidualSplitConvLSTM = ResidualTrendSplitConvLSTM


# -----------------------------
# Train / eval
# -----------------------------

@torch.no_grad()
def metrics_from_pred(pred_raw: torch.Tensor, target_raw: torch.Tensor) -> Dict[str, float]:
    p = pred_raw.float().view(-1, 1); y = target_raw.float().view(-1, 1)
    e = p - y
    out = {"mae_c": float(e.abs().mean()), "rmse_c": float(torch.sqrt((e*e).mean())), "bias_c": float(e.mean())}
    if p.numel() > 1 and float(p.std()) > 1e-8 and float(y.std()) > 1e-8:
        out["corr"] = float(torch.corrcoef(torch.stack([p.flatten(), y.flatten()]))[0,1])
    else:
        out["corr"] = float("nan")
    return out


def future_from_delta(delta_normed: torch.Tensor, current: torch.Tensor, delta_norm: TargetNormalizer) -> torch.Tensor:
    return current.float() + delta_norm.inverse_tensor(delta_normed.float())


@torch.no_grad()
def evaluate(model, loader, device, delta_norm: TargetNormalizer, amp_dtype: torch.dtype, context_scale: float = 0.15) -> Dict[str, float]:
    model.eval()
    preds=[]; targs=[]; curs=[]; losses=[]
    loss_fn = nn.SmoothL1Loss(beta=0.75)
    for batch in loader:
        mask = batch["mask"].to(device, non_blocking=True)
        cloud = batch["cloud_features"].to(device, non_blocking=True)
        world = batch["world_features"].to(device, non_blocking=True)
        trend = batch["trend_features"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        targ_raw = batch["target_raw"].to(device, non_blocking=True)
        cur = batch["current_temp_raw"].to(device, non_blocking=True)
        with autocast("cuda", enabled=device.type=="cuda", dtype=amp_dtype):
            out = model(mask, cloud, world, trend, context_scale=context_scale)
            loss = loss_fn(out["final_delta"], target)
        preds.append(future_from_delta(out["final_delta"], cur, delta_norm).detach().cpu())
        targs.append(targ_raw.detach().cpu()); curs.append(cur.detach().cpu()); losses.append(float(loss))
    p = torch.cat(preds); y = torch.cat(targs); c = torch.cat(curs)
    m = metrics_from_pred(p, y)
    pm = metrics_from_pred(c, y)
    m["loss_norm_delta"] = float(np.mean(losses))
    m["persistence_mae_c"] = pm["mae_c"]
    m["improvement_vs_persistence_c"] = pm["mae_c"] - m["mae_c"]
    return m


def save_ckpt(path, model, opt, scheduler, epoch, args, metadata, x_norm, delta_norm, metrics,
              train_target_mean_c, train_city_means):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "args": vars(args),
        "raw_feature_names": metadata["raw_feature_names"],
        "cloud_feature_names": metadata["cloud_feature_names"],
        "world_feature_names": metadata["world_feature_names"],
        "normalizer": x_norm.state_dict(),
        "delta_normalizer": delta_norm.state_dict(),
        "target_normalizer": delta_norm.state_dict(),
        "target_is_delta": True,
        "train_target_mean_c": train_target_mean_c,
        "train_city_means": train_city_means,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "lookback": args.lookback,
        "model_kwargs": {
            "num_cloud_features": len(metadata["cloud_feature_names"]),
            "num_world_features": len(metadata["world_feature_names"]),
            "stem_dim": args.stem_dim,
            "convlstm_dims": [int(x) for x in args.convlstm_dims.split(",") if x.strip()],
            "cloud_scalar_dim": args.cloud_scalar_dim,
            "world_dim": args.world_dim,
            "trend_dim": args.trend_dim,
            "fusion_dim": args.fusion_dim,
            "dropout": args.dropout,
            "delta_norm_limit": (args.delta_limit_c / delta_norm.std) if args.delta_limit_c > 0 else None,
        },
        "metrics": metrics,
        "architecture": "CloudForcedRadiationSplitConvLSTM_v6",
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-height", type=int, default=160); ap.add_argument("--image-width", type=int, default=160)
    ap.add_argument("--lookback", type=int, default=4); ap.add_argument("--max-gap-days", type=float, default=12.0)
    ap.add_argument("--epochs", type=int, default=140); ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0); ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight-decay", type=float, default=4e-4)
    ap.add_argument("--grad-clip", type=float, default=0.75)
    ap.add_argument("--stem-dim", type=int, default=64); ap.add_argument("--convlstm-dims", default="64,96")
    ap.add_argument("--cloud-scalar-dim", type=int, default=128); ap.add_argument("--world-dim", type=int, default=160)
    ap.add_argument("--trend-dim", type=int, default=96)
    ap.add_argument("--fusion-dim", type=int, default=192); ap.add_argument("--dropout", type=float, default=0.16)
    ap.add_argument("--delta-limit-c", type=float, default=12.0)
    ap.add_argument("--image-aux-weight", type=float, default=0.90); ap.add_argument("--cloud-aux-weight", type=float, default=1.10)
    ap.add_argument("--world-aux-weight", type=float, default=0.03); ap.add_argument("--trend-aux-weight", type=float, default=0.05)
    ap.add_argument("--interaction-aux-weight", type=float, default=0.08)
    ap.add_argument("--component-l2-weight", type=float, default=0.004)

    # Cloud-forcing controls. context_scale=0 gives a strict cloud-only final model.
    # Small values let world/trend correct but not dominate.
    ap.add_argument("--context-scale", type=float, default=0.15)
    ap.add_argument("--image-radiation-weight", type=float, default=2.0)
    ap.add_argument("--cloud-radiation-weight", type=float, default=2.5)
    ap.add_argument("--cloud-primary-weight", type=float, default=0.75, help="Extra loss on current_temp + image_delta + cloud_delta.")
    ap.add_argument("--radiation-loss-scale", type=float, default=300.0, help="Normalize W/m2 radiation loss for radiation heads.")
    ap.add_argument("--image-drop-prob", type=float, default=0.05); ap.add_argument("--cloud-scalar-drop-prob", type=float, default=0.10)
    ap.add_argument("--world-drop-prob", type=float, default=0.08); ap.add_argument("--world-feature-drop-prob", type=float, default=0.04)
    ap.add_argument("--trend-drop-prob", type=float, default=0.05); ap.add_argument("--trend-feature-drop-prob", type=float, default=0.03)
    ap.add_argument("--augment", action="store_true"); ap.add_argument("--cache-images", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=35)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--amp-dtype", choices=["bf16","fp16"], default="bf16")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    root = Path(args.data_root).resolve(); out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(root)
    raw_names = list(meta["raw_feature_names"]); cloud_names = list(meta["cloud_feature_names"]); world_names = list(meta["world_feature_names"])
    train_records = load_records(root, "train"); val_records = load_records(root, "val"); test_records = load_records(root, "test")

    x_norm = Normalizer.fit(np.asarray([feature_vector(r, raw_names) for r in train_records], dtype=np.float32))
    delta_norm = TargetNormalizer.fit(get_target_temp(r) - get_current_temp(r) for r in train_records)
    train_target_mean_c = float(np.mean([get_target_temp(r) for r in train_records]))
    city_vals: Dict[str, List[float]] = {}
    for r in train_records:
        city_vals.setdefault(str(r.get("location", r.get("city", ""))), []).append(get_target_temp(r))
    train_city_means = {k: float(np.mean(v)) for k,v in city_vals.items()}

    ds_args = dict(root=root, raw_names=raw_names, cloud_names=cloud_names, world_names=world_names,
                   x_norm=x_norm, delta_norm=delta_norm, image_height=args.image_height, image_width=args.image_width,
                   lookback=args.lookback, max_gap_days=args.max_gap_days)
    train_ds = CloudTempResidualSequenceDataset(records=train_records, augment=args.augment, cache_images=args.cache_images, **ds_args)
    val_ds = CloudTempResidualSequenceDataset(records=val_records, augment=False, cache_images=args.cache_images, **ds_args)
    test_ds = CloudTempResidualSequenceDataset(records=test_records, augment=False, cache_images=args.cache_images, **ds_args)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=device.type=="cuda", persistent_workers=args.num_workers>0)
    if args.num_workers > 0: loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = ResidualTrendSplitConvLSTM(
        len(cloud_names), len(world_names), stem_dim=args.stem_dim,
        convlstm_dims=[int(x) for x in args.convlstm_dims.split(",") if x.strip()],
        cloud_scalar_dim=args.cloud_scalar_dim, world_dim=args.world_dim, trend_dim=args.trend_dim, fusion_dim=args.fusion_dim,
        dropout=args.dropout, delta_norm_limit=(args.delta_limit_c / delta_norm.std if args.delta_limit_c > 0 else None),
    ).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.55, patience=10)
    loss_fn = nn.SmoothL1Loss(beta=0.75)
    scaler = GradScaler("cuda", enabled=(args.amp_dtype=="fp16" and device.type=="cuda"))

    print(json.dumps({
        "architecture": "CloudForcedRadiationSplitConvLSTM_v6",
        "device": str(device),
        "num_train_windows": len(train_ds), "num_val_windows": len(val_ds), "num_test_windows": len(test_ds),
        "cloud_features": cloud_names, "world_features": world_names,
        "delta_mean_c": delta_norm.mean, "delta_std_c": delta_norm.std,
        "context_scale": args.context_scale,
        "formula": "future_temp = current_temperature_c + inverse_delta(image_delta + cloud_delta + context_scale*(world_delta+trend_delta+interaction_delta))",
        "trend_features": ["temp_rel_current", "days_rel_current", "step_temp_change", "step_days"],
    }, indent=2))

    best_mae = float("inf"); no_improve = 0; hist = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total=0; total_loss=0.0; total_mae=0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            mask = batch["mask"].to(device, non_blocking=True)
            cloud = batch["cloud_features"].to(device, non_blocking=True)
            world = batch["world_features"].to(device, non_blocking=True)
            trend = batch["trend_features"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_raw = batch["target_raw"].to(device, non_blocking=True)
            current = batch["current_temp_raw"].to(device, non_blocking=True)

            if args.image_drop_prob > 0 and random.random() < args.image_drop_prob: mask = torch.zeros_like(mask)
            if args.world_drop_prob > 0 and random.random() < args.world_drop_prob: world = torch.zeros_like(world)
            if args.world_feature_drop_prob > 0:
                drop = (torch.rand(world.shape[-1], device=device) < args.world_feature_drop_prob).float()
                world = world * (1.0 - drop.view(1,1,-1))
            if args.cloud_scalar_drop_prob > 0:
                drop = (torch.rand(cloud.shape[-1], device=device) < args.cloud_scalar_drop_prob).float()
                cloud = cloud * (1.0 - drop.view(1,1,-1))
            if args.trend_drop_prob > 0 and random.random() < args.trend_drop_prob:
                trend = torch.zeros_like(trend)
            if args.trend_feature_drop_prob > 0:
                drop = (torch.rand(trend.shape[-1], device=device) < args.trend_feature_drop_prob).float()
                trend = trend * (1.0 - drop.view(1,1,-1))

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type=="cuda", dtype=amp_dtype):
                out = model(mask, cloud, world, trend, context_scale=args.context_scale)
                loss = loss_fn(out["final_delta"], target)
                loss = loss + args.image_aux_weight * loss_fn(out["image_delta"], target)
                loss = loss + args.cloud_aux_weight * loss_fn(out["cloud_delta"], target)
                loss = loss + args.world_aux_weight * loss_fn(out["world_delta"], target)
                loss = loss + args.trend_aux_weight * loss_fn(out["trend_delta"], target)
                loss = loss + args.interaction_aux_weight * loss_fn(out["interaction_delta"], target)

                # Cloud-primary temp loss: force image+cloud to independently model the temperature change.
                cloud_primary_delta = out["cloud_primary_delta"]
                loss = loss + args.cloud_primary_weight * loss_fn(cloud_primary_delta, target)

                # Radiation target: strongest direct supervision for Sentinel-2 cloud image.
                rad_valid = batch["radiation_valid"].to(device, non_blocking=True)
                rad_loss_raw = batch["radiation_loss_raw"].to(device, non_blocking=True)
                rad_target = rad_loss_raw / max(1e-6, float(args.radiation_loss_scale))
                if float(rad_valid.sum().item()) > 0:
                    image_rad_loss = (loss_fn(out["image_radiation"], rad_target) * rad_valid).sum() / rad_valid.sum().clamp_min(1.0)
                    cloud_rad_loss = (loss_fn(out["cloud_radiation"], rad_target) * rad_valid).sum() / rad_valid.sum().clamp_min(1.0)
                    loss = loss + args.image_radiation_weight * image_rad_loss
                    loss = loss + args.cloud_radiation_weight * cloud_rad_loss

                loss = loss + args.component_l2_weight * (
                    out["image_delta"].pow(2).mean() + out["cloud_delta"].pow(2).mean()
                    + out["world_delta"].pow(2).mean() + out["trend_delta"].pow(2).mean()
                    + out["interaction_delta"].pow(2).mean()
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()

            with torch.no_grad():
                pred_raw = future_from_delta(out["final_delta"], current, delta_norm)
                mae = (pred_raw - target_raw).abs().mean()
            bs = target.size(0); total += bs; total_loss += float(loss)*bs; total_mae += float(mae)*bs
            pbar.set_postfix(loss=total_loss/max(1,total), mae_c=total_mae/max(1,total))

        val = evaluate(model, val_loader, device, delta_norm, amp_dtype, context_scale=args.context_scale); scheduler.step(val["mae_c"])
        row = {"epoch": epoch, "train_loss": total_loss/max(1,total), "train_mae_c": total_mae/max(1,total), "val": val, "lr": opt.param_groups[0]["lr"]}
        hist.append(row); write_json(out_dir/"history.json", {"history": hist})
        print(f"epoch={epoch:03d} train_mae={row['train_mae_c']:.3f}C val_mae={val['mae_c']:.3f}C val_rmse={val['rmse_c']:.3f}C corr={val['corr']:.3f} persist={val['persistence_mae_c']:.3f}C improve={val['improvement_vs_persistence_c']:.3f}C lr={row['lr']:.2e}")
        save_ckpt(out_dir/"last.pt", model, opt, scheduler, epoch, args, meta, x_norm, delta_norm, row, train_target_mean_c, train_city_means)
        if val["mae_c"] < best_mae:
            best_mae = val["mae_c"]; no_improve = 0
            save_ckpt(out_dir/"best.pt", model, opt, scheduler, epoch, args, meta, x_norm, delta_norm, row, train_target_mean_c, train_city_means)
            print(f"saved best.pt val_mae={best_mae:.3f}C improve_vs_persistence={val['improvement_vs_persistence_c']:.3f}C")
        else:
            no_improve += 1
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"early stop: {no_improve} epochs without improvement")
            break

    if len(test_ds) and (out_dir/"best.pt").exists():
        ckpt = torch.load(out_dir/"best.pt", map_location=device)
        state = ckpt["model_state"]
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.","",1): v for k,v in state.items()}
        model.load_state_dict(state)
        test = evaluate(model, test_loader, device, delta_norm, amp_dtype, context_scale=args.context_scale)
        write_json(out_dir/"test_metrics.json", test)
        print("test:", json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
