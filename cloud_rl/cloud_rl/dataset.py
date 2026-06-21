from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

FEATURE_KEYS: List[str] = [
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

DEFAULT_FEATURE_MEAN = np.array(
    [0.5, 20.0, 5000.0, 65000.0, 0.0, 70.0, 1013.25, 20.0, 5.0, 180.0, 15.0, 1.0, 50.0, 1000.0],
    dtype=np.float32,
)
DEFAULT_FEATURE_STD = np.array(
    [0.25, 20.0, 3000.0, 15000.0, 2.0, 20.0, 20.0, 15.0, 5.0, 104.0, 10.0, 5.0, 30.0, 30.0],
    dtype=np.float32,
)

DEFAULT_FEATURE_STATS = {
    name: (float(mean), float(std))
    for name, mean, std in zip(FEATURE_KEYS, DEFAULT_FEATURE_MEAN, DEFAULT_FEATURE_STD)
}

SKIP_JSON_NAMES = {
    "stats.json",
    "metadata.json",
    "config.resolved.json",
    "history.tail.json",
}


def discover_sample_sources(root: Path, split: Optional[str] = None, manifest_path: Optional[Path] = None) -> List[Path]:
    if manifest_path is not None:
        resolved = Path(manifest_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Manifest file not found: {resolved}")
        return [resolved]

    if root.is_file():
        return [root]

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if split and split not in {"", "all"}:
        split_manifest = root / "splits" / f"{split}.jsonl"
        if split_manifest.exists():
            return [split_manifest]

    dataset_manifest = root / "dataset.jsonl"
    if dataset_manifest.exists():
        return [dataset_manifest]

    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in SKIP_JSON_NAMES
        and path.suffix in {".json", ".jsonl"}
    ]
    if not files:
        raise FileNotFoundError(f"No dataset samples found under {root}")
    return files


def read_sample_records(source: Path) -> List[Dict]:
    if source.suffix == ".jsonl":
        records: List[Dict] = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        if not records:
            raise ValueError(f"Manifest file {source} did not contain any records.")
        return records

    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload
    return [payload]


def resolve_mask_path(data_root: Path, record_source: Path, mask_path: str) -> Path:
    p = Path(mask_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(data_root / p)
    candidates.append(record_source.parent / p)
    candidates.append(record_source.parent.parent / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve mask_path={mask_path!r} for source {record_source}")


def load_binary_mask(path: Path, size_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = size_hw
    img = Image.open(path).convert("L")
    # The exporter writes square 256x256 masks by default. Resize only to
    # keep the training code resilient when the exporter size changes.
    img = img.resize((w, h), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr >= 0.5).astype(np.float32)
    return torch.from_numpy(arr)[None, :, :]


def load_cloud_tensor_or_mask(data_root: Path, record_source: Path, sample: Dict, mask: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Load Sentinel cloud tensor [C,H,W] when present; otherwise synthesize from mask."""
    fallback = torch.cat([mask.float(), mask.float(), mask.float(), torch.zeros(5, *mask.shape[-2:], dtype=torch.float32)], dim=0)
    tensor_path = sample.get("cloud_tensor_path")
    h, w = size_hw
    if tensor_path:
        p = Path(str(tensor_path))
        candidates = [p] if p.is_absolute() else []
        candidates.extend([data_root / p, record_source.parent / p, record_source.parent.parent / p])
        for candidate in candidates:
            if candidate.exists():
                arr = np.load(candidate)["cloud_tensor"].astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(arr, 0.0, 1.0)
                channels = []
                for idx in range(arr.shape[-1]):
                    img = Image.fromarray((arr[..., idx] * 255.0).astype(np.uint8)).convert("L")
                    if img.size != (w, h):
                        img = img.resize((w, h), Image.Resampling.BILINEAR)
                    channels.append(np.asarray(img, dtype=np.float32) / 255.0)
                tensor = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))
                if tensor.shape[0] < 8:
                    pad = torch.zeros(8 - tensor.shape[0], h, w, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, pad], dim=0)
                return tensor[:8]
    return fallback


def load_feature_keys(data_root: Path) -> List[str]:
    metadata_path = data_root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("raw_feature_names", "feature_keys", "model_feature_names"):
            names = metadata.get(key)
            if names:
                return list(names)
    return list(FEATURE_KEYS)


def default_stats_for_keys(feature_keys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for name in feature_keys:
        mean, std = DEFAULT_FEATURE_STATS.get(name, (0.0, 1.0))
        means.append(mean)
        stds.append(std)
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def extract_feature_vector(sample: Dict, feature_keys: Sequence[str] = FEATURE_KEYS) -> np.ndarray:
    if "feature_vector" in sample and len(sample["feature_vector"]) == len(feature_keys):
        return np.asarray(sample["feature_vector"], dtype=np.float32)
    inputs = sample.get("inputs", {})
    return np.asarray([float(inputs.get(k, np.nan)) for k in feature_keys], dtype=np.float32)


def parse_sample_datetime(sample: Dict) -> np.datetime64:
    raw = str(sample.get("anchor") or sample.get("date") or "").replace("Z", "")
    if "+" in raw:
        raw = raw.split("+", 1)[0]
    if not raw:
        return np.datetime64("1970-01-01")
    return np.datetime64(raw)


def sample_location_key(sample: Dict) -> str:
    return str(sample.get("location") or sample.get("city") or "")


def current_temperature_c(sample: Dict) -> float:
    value = sample.get("current_temperature_c")
    if value is not None:
        return float(value)
    inputs = sample.get("inputs") or {}
    if "world_temperature_2m" in inputs:
        return float(inputs["world_temperature_2m"])
    return float(sample.get("target_temperature_c", sample.get("target", 20.0)))


def target_offset_days(sample: Dict) -> float:
    value = sample.get("target_offset_days")
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    if sample.get("target_timestamp") and (sample.get("anchor") or sample.get("date")):
        try:
            target_dt = np.datetime64(str(sample["target_timestamp"]).replace("Z", ""))
            anchor_dt = np.datetime64(str(sample.get("anchor") or sample.get("date")).replace("Z", ""))
            return float((target_dt - anchor_dt) / np.timedelta64(1, "D"))
        except Exception:
            pass
    return 5.0


def solar_clear_sky_proxy_wm2(sample: Dict) -> float:
    raw = str(sample.get("anchor") or sample.get("date") or "")
    try:
        import datetime as _dt
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        dt = dt.astimezone(_dt.timezone.utc)
    except Exception:
        return 0.0

    lat = float(sample.get("lat", 0.0))
    lon = float(sample.get("lon", 0.0))
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
    true_solar_time_min = hour * 60.0 + eqtime + 4.0 * lon
    hour_angle = math.radians(true_solar_time_min / 4.0 - 180.0)
    lat_rad = math.radians(lat)
    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    if cos_zenith <= 0.0:
        return 0.0
    ecc = 1.0 + 0.033 * math.cos(2.0 * math.pi * day / 365.0)
    return float(max(0.0, 1361.0 * ecc * cos_zenith * 0.72))


def radiation_bundle(sample: Dict) -> Dict[str, float]:
    if "radiation_cloud_loss_wm2" in sample:
        return {
            "loss_wm2": float(sample.get("radiation_cloud_loss_wm2", 0.0)),
            "attenuation": float(sample.get("radiation_cloud_attenuation", 0.0)),
            "observed_wm2": float(sample.get("radiation_shortwave_observed_wm2", 0.0)),
            "clear_wm2": float(sample.get("radiation_clear_sky_proxy_wm2", 0.0)),
            "valid": float(sample.get("radiation_daylight_valid", 0.0)),
        }
    inputs = sample.get("inputs") or {}
    observed = float(inputs.get("world_shortwave_radiation", inputs.get("shortwave_radiation", 0.0)))
    clear = solar_clear_sky_proxy_wm2(sample)
    valid = 1.0 if clear >= 50.0 else 0.0
    if valid:
        transmission = max(0.0, min(1.5, observed / max(clear, 1e-6)))
        attenuation = max(-0.5, min(1.2, 1.0 - transmission))
        loss = max(-250.0, min(1200.0, clear - observed))
    else:
        attenuation = 0.0
        loss = 0.0
    return {"loss_wm2": float(loss), "attenuation": float(attenuation), "observed_wm2": observed, "clear_wm2": clear, "valid": valid}


def radiation_context_features(sample: Dict) -> np.ndarray:
    rb = radiation_bundle(sample)
    raw = str(sample.get("anchor") or sample.get("date") or "")
    try:
        import datetime as _dt
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        dt = dt.astimezone(_dt.timezone.utc)
        doy = int(dt.strftime("%j"))
        hour = dt.hour + dt.minute / 60.0
    except Exception:
        doy = 180
        hour = 12.0
    day_ang = 2.0 * math.pi * doy / 365.0
    hour_ang = 2.0 * math.pi * hour / 24.0
    return np.asarray(
        [
            float(rb["clear_wm2"]) / 1000.0,
            float(rb["valid"]),
            math.sin(day_ang),
            math.cos(day_ang),
            math.sin(hour_ang),
            math.cos(hour_ang),
            float(sample.get("lat", 0.0)) / 90.0,
            float(sample.get("lon", 0.0)) / 180.0,
        ],
        dtype=np.float32,
    )


def days_between(a: Dict, b: Dict) -> float:
    try:
        return float((parse_sample_datetime(a) - parse_sample_datetime(b)) / np.timedelta64(1, "D"))
    except Exception:
        return 0.0


def build_trend_features(records: Sequence[Dict]) -> np.ndarray:
    last = records[-1]
    temps = np.asarray([current_temperature_c(r) for r in records], dtype=np.float32)
    days = np.asarray([days_between(r, last) for r in records], dtype=np.float32)
    step_temp = np.zeros_like(temps, dtype=np.float32)
    step_days = np.zeros_like(days, dtype=np.float32)
    if len(temps) > 1:
        step_temp[1:] = temps[1:] - temps[:-1]
        step_days[1:] = np.maximum(0.0, days[1:] - days[:-1])
    return np.stack([temps - temps[-1], days, step_temp, step_days], axis=1).astype(np.float32) / 10.0


def build_sample_windows(
    samples: Sequence[Tuple[Dict, Path]],
    lookback: int,
    max_gap_days: float,
) -> List[List[Tuple[Dict, Path]]]:
    if lookback <= 1:
        return [[item] for item in samples]

    by_loc: Dict[str, List[Tuple[Dict, Path]]] = {}
    for item in samples:
        by_loc.setdefault(sample_location_key(item[0]), []).append(item)

    max_gap = np.timedelta64(int(round(max_gap_days * 24)), "h")
    windows: List[List[Tuple[Dict, Path]]] = []
    for rows in by_loc.values():
        rows.sort(key=lambda item: parse_sample_datetime(item[0]))
        for i in range(lookback - 1, len(rows)):
            chunk = rows[i - lookback + 1 : i + 1]
            if all(
                parse_sample_datetime(b[0]) - parse_sample_datetime(a[0]) <= max_gap
                for a, b in zip(chunk[:-1], chunk[1:])
            ):
                windows.append(chunk)
    return windows


def compute_stats(
    data_root: str | Path,
    feature_keys: Optional[Sequence[str]] = None,
    split: Optional[str] = "train",
    manifest_path: Optional[str | Path] = None,
) -> Dict[str, List[float]]:
    root = Path(data_root)
    resolved_feature_keys = list(feature_keys) if feature_keys is not None else load_feature_keys(root)
    source = discover_sample_sources(root, split=split, manifest_path=Path(manifest_path) if manifest_path else None)
    samples = []
    for item in source:
        samples.extend(read_sample_records(item))

    xs, ts = [], []
    for sample in samples:
        x = extract_feature_vector(sample, resolved_feature_keys)
        if np.isfinite(x).all():
            xs.append(x)
        ts.append(float(sample.get("target_temperature_c", 20.0)))

    if not xs:
        mean, std = default_stats_for_keys(resolved_feature_keys)
    else:
        mat = np.stack(xs).astype(np.float32)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)

    t = np.asarray(ts, dtype=np.float32)
    return {
        "feature_keys": list(resolved_feature_keys),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "target_temp_mean": [float(t.mean()) if len(t) else 20.0],
        "target_temp_std": [float(max(t.std(), 1.0)) if len(t) else 10.0],
    }


class CloudFolderDataset(Dataset):
    """Loads the exporter output manifest and its mask PNGs.

    The dataset accepts either:
      - dataset_out/dataset.jsonl
      - dataset_out/splits/train.jsonl, val.jsonl, test.jsonl
      - a folder of individual JSON sample files
      - a single JSON or JSONL file
    """

    def __init__(
        self,
        data_root: str | Path,
        image_size: Tuple[int, int] = (256, 256),
        stats_path: Optional[str | Path] = None,
        include_feature_planes: bool = True,
        split: Optional[str] = "train",
        manifest_path: Optional[str | Path] = None,
        lookback: int = 1,
        max_gap_days: float = 12.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.include_feature_planes = include_feature_planes
        self.feature_keys = load_feature_keys(self.data_root)
        self.manifest_sources = discover_sample_sources(
            self.data_root,
            split=split,
            manifest_path=Path(manifest_path) if manifest_path is not None else None,
        )
        self.samples: List[Tuple[Dict, Path]] = []
        for source in self.manifest_sources:
            for sample in read_sample_records(source):
                self.samples.append((sample, source))
        if not self.samples:
            raise FileNotFoundError(f"No usable samples found in {self.data_root}")
        self.lookback = max(1, int(lookback))
        self.max_gap_days = float(max_gap_days)
        self.windows = build_sample_windows(self.samples, self.lookback, self.max_gap_days)
        if not self.windows:
            raise FileNotFoundError(
                f"No usable lookback windows found in {self.data_root}; "
                f"lookback={self.lookback}, max_gap_days={self.max_gap_days}"
            )

        stats = None
        if stats_path is not None and Path(stats_path).exists():
            stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        elif (self.data_root / "stats.json").exists():
            stats = json.loads((self.data_root / "stats.json").read_text(encoding="utf-8"))

        if stats is None:
            feature_rows = []
            targets = []
            for sample, _source in self.samples:
                x = extract_feature_vector(sample, self.feature_keys)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if np.isfinite(x).all():
                    feature_rows.append(x)
                targets.append(float(sample.get("target_temperature_c", sample.get("target", 20.0))))
            if feature_rows:
                mat = np.stack(feature_rows).astype(np.float32)
                self.feature_mean = mat.mean(axis=0)
                self.feature_std = mat.std(axis=0)
            else:
                self.feature_mean, self.feature_std = default_stats_for_keys(self.feature_keys)
            t = np.asarray(targets, dtype=np.float32)
            self.target_mean = float(t.mean()) if len(t) else 20.0
            self.target_std = float(max(t.std(), 1.0)) if len(t) else 10.0
        else:
            stats_feature_keys = list(stats.get("feature_keys") or self.feature_keys)
            if stats_feature_keys != self.feature_keys:
                raise ValueError(
                    "stats.json feature_keys do not match dataset metadata feature names. "
                    "Regenerate stats with train_rl.py --write-stats for this dataset."
                )
            self.feature_mean = np.asarray(stats["feature_mean"], dtype=np.float32)
            self.feature_std = np.asarray(stats["feature_std"], dtype=np.float32)
            self.target_mean = float(stats["target_temp_mean"][0])
            self.target_std = float(stats["target_temp_std"][0])

        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        self.target_std = max(self.target_std, 1e-6)
        # Policy state includes the latest normalized raw feature vector plus the
        # v6 trend sequence, current-temp anchor, and target offset. The reward
        # still receives raw feature tensors separately.
        self.policy_feature_dim = len(self.feature_keys) + self.lookback * 4 + 5
        self.feature_dim = self.policy_feature_dim

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | Dict]:
        window = self.windows[idx]
        sample, source = window[-1]
        masks = []
        cloud_tensors = []
        raw_sequence = []
        for row, row_source in window:
            mask_path = resolve_mask_path(self.data_root, row_source, row["mask_path"])
            mask = load_binary_mask(mask_path, self.image_size)
            masks.append(mask)
            cloud_tensors.append(load_cloud_tensor_or_mask(self.data_root, row_source, row, mask, self.image_size))
            raw = extract_feature_vector(row, self.feature_keys)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            raw_sequence.append(raw)

        raw_features = raw_sequence[-1]
        raw_feature_sequence = np.stack(raw_sequence, axis=0).astype(np.float32)
        norm_sequence = (raw_feature_sequence - self.feature_mean) / self.feature_std
        norm_sequence = np.nan_to_num(norm_sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        norm_sequence = np.clip(norm_sequence, -8.0, 8.0)
        norm_features = norm_sequence[-1]
        trend = build_trend_features([row for row, _ in window])
        if trend.shape[0] != self.lookback:
            pad = np.repeat(trend[:1], self.lookback - trend.shape[0], axis=0)
            trend = np.concatenate([pad, trend], axis=0)

        target = float(sample.get("target_temperature_c", sample.get("target", 20.0)))
        current = current_temperature_c(sample)
        rb = radiation_bundle(sample)
        rad_loss_norm = np.asarray([float(rb["loss_wm2"]) / 300.0], dtype=np.float32)
        rad_clear_norm = np.asarray([float(rb["clear_wm2"]) / 1000.0], dtype=np.float32)
        rad_attn = np.asarray([float(rb["attenuation"])], dtype=np.float32)
        target_norm = np.asarray([(target - self.target_mean) / self.target_std], dtype=np.float32)
        current_norm = np.asarray([(current - self.target_mean) / self.target_std], dtype=np.float32)
        offset_norm = np.asarray([target_offset_days(sample) / 10.0], dtype=np.float32)
        policy_features = np.concatenate([norm_features, trend.reshape(-1), current_norm, offset_norm, rad_loss_norm, rad_clear_norm, rad_attn]).astype(np.float32)
        policy_features = np.nan_to_num(policy_features, nan=0.0, posinf=0.0, neginf=0.0)
        policy_features = np.clip(policy_features, -8.0, 8.0).astype(np.float32)

        h, w = self.image_size
        mask_sequence = torch.stack(masks, dim=0).float()
        cloud_tensor_sequence = torch.stack(cloud_tensors, dim=0).float()
        mask = mask_sequence[-1]
        feature_planes = torch.from_numpy(policy_features).float()[:, None, None].expand(-1, h, w)
        target_plane = torch.full((1, h, w), float(target_norm[0]), dtype=torch.float32)
        obs_map = torch.cat([mask, feature_planes, target_plane], dim=0)

        return {
            "obs_map": obs_map.float(),
            "original_mask": mask.float(),
            "original_mask_sequence": mask_sequence,
            "cloud_tensor_sequence": cloud_tensor_sequence,
            "features": torch.from_numpy(policy_features).float(),
            "raw_features": torch.from_numpy(raw_features).float(),
            "raw_feature_sequence": torch.from_numpy(raw_feature_sequence).float(),
            "trend_features": torch.from_numpy(trend).float(),
            "current_temp": torch.tensor([current], dtype=torch.float32),
            "current_radiation_loss_wm2": torch.tensor([float(rb["loss_wm2"])], dtype=torch.float32),
            "radiation_clear_wm2": torch.tensor([float(rb["clear_wm2"])], dtype=torch.float32),
            "radiation_observed_wm2": torch.tensor([float(rb["observed_wm2"])], dtype=torch.float32),
            "radiation_attenuation": torch.tensor([float(rb["attenuation"])], dtype=torch.float32),
            "radiation_valid": torch.tensor([float(rb["valid"])], dtype=torch.float32),
            "radiation_context_features": torch.from_numpy(radiation_context_features(sample)).float(),
            "target_temp": torch.tensor([target], dtype=torch.float32),
            "target_temp_norm": torch.from_numpy(target_norm).float(),
            "sample_id": str(sample.get("sample_id", f"sample_{idx}")),
            "json_path": str(source),
            "meta": sample,
        }


def collate_cloud_batch(batch: List[Dict]) -> Dict:
    out: Dict = {}
    tensor_keys = [
        "obs_map",
        "original_mask",
        "original_mask_sequence",
        "cloud_tensor_sequence",
        "features",
        "raw_features",
        "raw_feature_sequence",
        "trend_features",
        "current_temp",
        "current_radiation_loss_wm2",
        "radiation_clear_wm2",
        "radiation_observed_wm2",
        "radiation_attenuation",
        "radiation_valid",
        "radiation_context_features",
        "target_temp",
        "target_temp_norm",
    ]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["sample_id"] = [item["sample_id"] for item in batch]
    out["json_path"] = [item["json_path"] for item in batch]
    out["meta"] = [item["meta"] for item in batch]
    return out
