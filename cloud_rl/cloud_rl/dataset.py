from __future__ import annotations

import json
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


def extract_feature_vector(sample: Dict, feature_keys: Sequence[str] = FEATURE_KEYS) -> np.ndarray:
    if "feature_vector" in sample and len(sample["feature_vector"]) == len(feature_keys):
        return np.asarray(sample["feature_vector"], dtype=np.float32)
    inputs = sample.get("inputs", {})
    return np.asarray([float(inputs.get(k, np.nan)) for k in feature_keys], dtype=np.float32)


def compute_stats(
    data_root: str | Path,
    feature_keys: Sequence[str] = FEATURE_KEYS,
    split: Optional[str] = "train",
    manifest_path: Optional[str | Path] = None,
) -> Dict[str, List[float]]:
    root = Path(data_root)
    source = discover_sample_sources(root, split=split, manifest_path=Path(manifest_path) if manifest_path else None)
    samples = []
    for item in source:
        samples.extend(read_sample_records(item))

    xs, ts = [], []
    for sample in samples:
        x = extract_feature_vector(sample, feature_keys)
        if np.isfinite(x).all():
            xs.append(x)
        ts.append(float(sample.get("target_temperature_c", 20.0)))

    if not xs:
        mean, std = DEFAULT_FEATURE_MEAN, DEFAULT_FEATURE_STD
    else:
        mat = np.stack(xs).astype(np.float32)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)

    t = np.asarray(ts, dtype=np.float32)
    return {
        "feature_keys": list(feature_keys),
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
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.include_feature_planes = include_feature_planes
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

        stats = None
        if stats_path is not None and Path(stats_path).exists():
            stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        elif (self.data_root / "stats.json").exists():
            stats = json.loads((self.data_root / "stats.json").read_text(encoding="utf-8"))

        if stats is None:
            self.feature_mean = DEFAULT_FEATURE_MEAN
            self.feature_std = DEFAULT_FEATURE_STD
            self.target_mean = 20.0
            self.target_std = 10.0
        else:
            self.feature_mean = np.asarray(stats["feature_mean"], dtype=np.float32)
            self.feature_std = np.asarray(stats["feature_std"], dtype=np.float32)
            self.target_mean = float(stats["target_temp_mean"][0])
            self.target_std = float(stats["target_temp_std"][0])

        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        self.target_std = max(self.target_std, 1e-6)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | Dict]:
        sample, source = self.samples[idx]
        mask_path = resolve_mask_path(self.data_root, source, sample["mask_path"])
        mask = load_binary_mask(mask_path, self.image_size)

        raw_features = extract_feature_vector(sample)
        raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        norm_features = (raw_features - self.feature_mean) / self.feature_std

        target = float(sample.get("target_temperature_c", 20.0))
        target_norm = np.asarray([(target - self.target_mean) / self.target_std], dtype=np.float32)

        h, w = self.image_size
        feature_planes = torch.from_numpy(norm_features).float()[:, None, None].expand(-1, h, w)
        target_plane = torch.full((1, h, w), float(target_norm[0]), dtype=torch.float32)
        obs_map = torch.cat([mask, feature_planes, target_plane], dim=0)

        return {
            "obs_map": obs_map.float(),
            "original_mask": mask.float(),
            "features": torch.from_numpy(norm_features).float(),
            "raw_features": torch.from_numpy(raw_features).float(),
            "target_temp": torch.tensor([target], dtype=torch.float32),
            "target_temp_norm": torch.from_numpy(target_norm).float(),
            "sample_id": str(sample.get("sample_id", f"sample_{idx}")),
            "json_path": str(source),
            "meta": sample,
        }


def collate_cloud_batch(batch: List[Dict]) -> Dict:
    out: Dict = {}
    tensor_keys = ["obs_map", "original_mask", "features", "raw_features", "target_temp", "target_temp_norm"]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["sample_id"] = [item["sample_id"] for item in batch]
    out["json_path"] = [item["json_path"] for item in batch]
    out["meta"] = [item["meta"] for item in batch]
    return out
