#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


S2_BANDS = {
    "cloud_mask_scl": 0,
    "cloud_gray": 1,
    "cloud_red_B04": 2,
    "cloud_green_B03": 3,
    "cloud_blue_B02": 4,
    "cloud_nir_B08": 5,
    "cloud_prob_CLP": 6,
    "aerosol_AOT": 7,
    "water_vapour_WVP": 8,
    "cirrus_flag": 9,
    "high_cloud_flag": 10,
    "medium_cloud_flag": 11,
}


# This is the recommended aligned profile.
# It expands the earlier 20-channel model to 24 channels because the downloader
# has useful Sentinel-2 cloud/aerosol/water-vapour bands and ERA5 wind-v.
BUILDER24_INPUT_CHANNELS = [
    # Sentinel-2 cloud-only image/tensor channels
    "s2_cloud_mask_scl",
    "s2_cloud_gray",
    "s2_cloud_prob_CLP",
    "s2_cirrus_flag",

    # ERA5 cloud fields
    "era5_total_cloud_cover",
    "era5_low_cloud_cover",
    "era5_medium_cloud_cover",
    "era5_high_cloud_cover",
    "era5_total_column_cloud_liquid_water",
    "era5_total_column_cloud_ice_water",

    # S2 atmospheric/cloud enrichments
    "s2_aerosol_AOT",
    "s2_water_vapour_WVP",

    # ERA5 radiation bridge
    "era5_surface_solar_radiation_downwards",
    "era5_surface_solar_radiation_downwards_clear_sky",
    "era5_surface_thermal_radiation_downwards",
    "era5_surface_thermal_radiation_downwards_clear_sky",
    "era5_surface_net_solar_radiation",
    "era5_surface_net_thermal_radiation",

    # ERA5 dynamic atmosphere / rain support
    "era5_2m_temperature_anomaly",
    "era5_2m_dewpoint_temperature",
    "era5_total_column_water_vapour",
    "era5_surface_pressure",
    "era5_total_precipitation",
    "era5_wind_speed_10m",
]

CORE20_INPUT_CHANNELS = [
    "s2_cloud_mask_scl",
    "era5_low_cloud_cover",
    "era5_medium_cloud_cover",
    "era5_high_cloud_cover",
    "era5_total_column_cloud_liquid_water",
    "era5_total_column_cloud_ice_water",
    "s2_cloud_gray",
    "s2_cloud_prob_CLP",
    "era5_surface_solar_radiation_downwards",
    "era5_surface_solar_radiation_downwards_clear_sky",
    "era5_surface_thermal_radiation_downwards",
    "era5_surface_thermal_radiation_downwards_clear_sky",
    "era5_surface_net_solar_radiation",
    "era5_surface_net_thermal_radiation",
    "era5_2m_temperature_anomaly",
    "era5_2m_dewpoint_temperature",
    "era5_total_column_water_vapour",
    "era5_surface_pressure",
    "era5_total_precipitation",
    "era5_wind_speed_10m",
]

TARGET_CHANNELS = [
    "shortwave_anomaly",
    "longwave_anomaly",
    "net_radiation_anomaly",
    "temperature_anomaly",
]


ERA5_ALIASES = {
    "era5_total_cloud_cover": "era5_single_total_cloud_cover",
    "era5_low_cloud_cover": "era5_single_low_cloud_cover",
    "era5_medium_cloud_cover": "era5_single_medium_cloud_cover",
    "era5_high_cloud_cover": "era5_single_high_cloud_cover",
    "era5_total_column_cloud_liquid_water": "era5_single_total_column_cloud_liquid_water",
    "era5_total_column_cloud_ice_water": "era5_single_total_column_cloud_ice_water",
    "era5_surface_solar_radiation_downwards": "era5_single_surface_solar_radiation_downwards",
    "era5_surface_solar_radiation_downwards_clear_sky": "era5_single_surface_solar_radiation_downwards_clear_sky",
    "era5_surface_thermal_radiation_downwards": "era5_single_surface_thermal_radiation_downwards",
    "era5_surface_thermal_radiation_downwards_clear_sky": "era5_single_surface_thermal_radiation_downwards_clear_sky",
    "era5_surface_net_solar_radiation": "era5_single_surface_net_solar_radiation",
    "era5_surface_net_solar_radiation_clear_sky": "era5_single_surface_net_solar_radiation_clear_sky",
    "era5_surface_net_thermal_radiation": "era5_single_surface_net_thermal_radiation",
    "era5_surface_net_thermal_radiation_clear_sky": "era5_single_surface_net_thermal_radiation_clear_sky",
    "era5_2m_temperature": "era5_single_2m_temperature",
    "era5_2m_dewpoint_temperature": "era5_single_2m_dewpoint_temperature",
    "era5_total_column_water_vapour": "era5_single_total_column_water_vapour",
    "era5_surface_pressure": "era5_single_surface_pressure",
    "era5_total_precipitation": "era5_single_total_precipitation",
    "era5_10m_u_component_of_wind": "era5_single_10m_u_component_of_wind",
    "era5_10m_v_component_of_wind": "era5_single_10m_v_component_of_wind",
    "era5_boundary_layer_height": "era5_single_boundary_layer_height",
    "era5_cape": "era5_single_convective_available_potential_energy",
}


def parse_dt(value: str) -> datetime:
    # pandas handles ISO with offsets robustly.
    return pd.to_datetime(value, utc=True).to_pydatetime()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def resize_2d(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Small dependency-free bilinear-ish resize for numeric grids.

    Uses 1D interpolation over array indices. Good enough for converting small
    ERA5 grids to the training tensor grid.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 0:
        return np.full((height, width), float(arr), dtype=np.float32)
    if arr.ndim == 1:
        # Broadcast a 1D variable into the map.
        return np.full((height, width), float(np.nanmean(arr)), dtype=np.float32)
    if arr.ndim > 2:
        # For pressure-level variables, average the leading dimensions unless
        # you build a profile-specific extractor.
        leading = tuple(range(arr.ndim - 2))
        arr = np.nanmean(arr, axis=leading)

    if arr.shape == (height, width):
        return arr.astype(np.float32, copy=False)

    in_h, in_w = arr.shape
    if in_h == 1 and in_w == 1:
        return np.full((height, width), float(arr[0, 0]), dtype=np.float32)

    # Fill NaNs before interpolation.
    if not np.isfinite(arr).all():
        fill = np.nanmean(arr)
        if not np.isfinite(fill):
            fill = 0.0
        arr = np.nan_to_num(arr, nan=float(fill), posinf=float(fill), neginf=float(fill))

    src_x = np.linspace(0, 1, in_w)
    dst_x = np.linspace(0, 1, width)
    tmp = np.empty((in_h, width), dtype=np.float32)
    for i in range(in_h):
        tmp[i] = np.interp(dst_x, src_x, arr[i]).astype(np.float32)

    src_y = np.linspace(0, 1, in_h)
    dst_y = np.linspace(0, 1, height)
    out = np.empty((height, width), dtype=np.float32)
    for j in range(width):
        out[:, j] = np.interp(dst_y, src_y, tmp[:, j]).astype(np.float32)

    return out


def get_s2_band(payload: dict[str, np.ndarray], band_name: str, height: int, width: int) -> np.ndarray:
    if "s2_cloud_tensor" not in payload:
        return np.zeros((height, width), dtype=np.float32)
    arr = np.asarray(payload["s2_cloud_tensor"], dtype=np.float32)
    band_idx = S2_BANDS[band_name]
    if arr.ndim != 3:
        return np.zeros((height, width), dtype=np.float32)
    # Downloader stores [H,W,12].
    band = arr[:, :, band_idx]
    return resize_2d(band, height, width)


def get_era5(payload: dict[str, np.ndarray], logical_name: str, height: int, width: int) -> np.ndarray:
    key = ERA5_ALIASES.get(logical_name, logical_name)
    if key not in payload:
        return np.zeros((height, width), dtype=np.float32)
    return resize_2d(payload[key], height, width)


def extract_raw_field(payload: dict[str, np.ndarray], channel_name: str, height: int, width: int) -> np.ndarray:
    if channel_name.startswith("s2_"):
        s2_name = channel_name.removeprefix("s2_")
        return get_s2_band(payload, s2_name, height, width)

    if channel_name == "era5_wind_speed_10m":
        u = get_era5(payload, "era5_10m_u_component_of_wind", height, width)
        v = get_era5(payload, "era5_10m_v_component_of_wind", height, width)
        return np.sqrt(u * u + v * v).astype(np.float32)

    if channel_name == "era5_2m_temperature_anomaly":
        # Temporary raw temperature. We convert to anomaly after all states
        # are loaded because anomaly needs same-location seasonal baseline.
        return get_era5(payload, "era5_2m_temperature", height, width)

    return get_era5(payload, channel_name, height, width)


def compute_target_raw(payload: dict[str, np.ndarray], height: int, width: int) -> np.ndarray:
    ssrd = get_era5(payload, "era5_surface_solar_radiation_downwards", height, width)
    ssrdc = get_era5(payload, "era5_surface_solar_radiation_downwards_clear_sky", height, width)
    strd = get_era5(payload, "era5_surface_thermal_radiation_downwards", height, width)
    strdc = get_era5(payload, "era5_surface_thermal_radiation_downwards_clear_sky", height, width)

    ssr = get_era5(payload, "era5_surface_net_solar_radiation", height, width)
    ssrc = get_era5(payload, "era5_surface_net_solar_radiation_clear_sky", height, width)
    strn = get_era5(payload, "era5_surface_net_thermal_radiation", height, width)
    strnc = get_era5(payload, "era5_surface_net_thermal_radiation_clear_sky", height, width)

    temp = get_era5(payload, "era5_2m_temperature", height, width)

    shortwave_anom = ssrd - ssrdc
    longwave_anom = strd - strdc

    # Prefer clear-sky net anomaly if available. If clear-sky fields are all
    # missing/zero, this will degrade to all-sky net after normalization.
    net_anom = (ssr - ssrc) + (strn - strnc)

    return np.stack([shortwave_anom, longwave_anom, net_anom, temp], axis=0).astype(np.float32)


def day_bin(dt: datetime, bin_days: int = 15) -> int:
    doy = int(dt.strftime("%j"))
    return int((doy - 1) // bin_days)


def apply_temperature_anomaly(
    x_series: np.ndarray,
    y_series: np.ndarray,
    rows: list[dict],
    input_channel_names: list[str],
    temp_input_name: str = "era5_2m_temperature_anomaly",
    target_temp_index: int = 3,
    bin_days: int = 15,
) -> None:
    """Convert raw 2m temperature maps into per-location seasonal anomalies.

    Modifies arrays in-place.
    """
    try:
        temp_input_idx = input_channel_names.index(temp_input_name)
    except ValueError:
        temp_input_idx = None

    # Build baseline by location and day-of-year bin.
    groups: dict[tuple[str, int], list[int]] = {}
    for i, row in enumerate(rows):
        key = (row["location"], day_bin(row["timestamp"], bin_days))
        groups.setdefault(key, []).append(i)

    for _, indices in groups.items():
        if len(indices) < 2:
            # Fallback to local group itself; anomaly will be near zero.
            pass
        target_base = np.nanmean(y_series[indices, target_temp_index], axis=0)
        y_series[indices, target_temp_index] -= target_base

        if temp_input_idx is not None:
            input_base = np.nanmean(x_series[indices, temp_input_idx], axis=0)
            x_series[indices, temp_input_idx] -= input_base


def normalize_and_save(
    x: np.ndarray,
    y: np.ndarray,
    train_state_indices: np.ndarray,
    out_dir: Path,
) -> None:
    # Channel-wise stats from training states only.
    x_train = x[train_state_indices]
    y_train = y[train_state_indices]

    x_mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    x_std = x_train.std(axis=(0, 2, 3), keepdims=True)
    y_mean = y_train.mean(axis=(0, 2, 3), keepdims=True)
    y_std = y_train.std(axis=(0, 2, 3), keepdims=True)

    x_std = np.maximum(x_std, 1e-6)
    y_std = np.maximum(y_std, 1e-6)

    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    y_norm = ((y - y_mean) / y_std).astype(np.float32)

    np.save(out_dir / "x_series.npy", x_norm)
    np.save(out_dir / "y_series.npy", y_norm)
    np.savez_compressed(
        out_dir / "normalization_stats.npz",
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
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


def build_windows_for_location(
    loc_rows: list[dict],
    input_len: int,
    horizon: int,
    train_frac: float,
    val_frac: float,
) -> tuple[list[dict], list[dict], list[dict], list[int]]:
    # loc_rows must be sorted by timestamp and contain global_idx.
    n = len(loc_rows)
    train_cut = int(n * train_frac)
    val_cut = int(n * (train_frac + val_frac))

    train_windows, val_windows, test_windows = [], [], []
    train_state_indices: list[int] = []

    for i, row in enumerate(loc_rows):
        if i < train_cut:
            train_state_indices.append(row["global_idx"])

    for target_start_i in range(input_len, n - horizon + 1):
        x_start_i = target_start_i - input_len
        x_end_i = target_start_i
        y_start_i = target_start_i
        y_end_i = target_start_i + horizon

        # Split by target start time, so validation/test future is not used in train.
        if target_start_i < train_cut:
            split = "train"
        elif target_start_i < val_cut:
            split = "val"
        else:
            split = "test"

        out = {
            "location": loc_rows[target_start_i]["location"],
            "target_timestamp": loc_rows[target_start_i]["timestamp"].isoformat(),
            "x_start": loc_rows[x_start_i]["global_idx"],
            "x_end": loc_rows[x_end_i - 1]["global_idx"] + 1,
            "y_start": loc_rows[y_start_i]["global_idx"],
            "y_end": loc_rows[y_end_i - 1]["global_idx"] + 1,
            "split": split,
        }

        # Require slices to be contiguous in global tensor. They are if we append
        # states by location in sorted order, which this script does.
        if split == "train":
            train_windows.append(out)
        elif split == "val":
            val_windows.append(out)
        else:
            test_windows.append(out)

    return train_windows, val_windows, test_windows, train_state_indices


def main():
    parser = argparse.ArgumentParser(
        description="Convert downloader states_npz into ConvLSTM training tensors and indexed windows."
    )
    parser.add_argument("--dataset-dir", required=True, help="Output directory from cloud_state_dataset_downloader.py")
    parser.add_argument("--out-dir", default="data/processed_aligned", help="Where to write x_series/y_series and window CSVs")
    parser.add_argument("--profile", choices=["core20", "builder24"], default="builder24")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=8, help="8 snapshots = about 40 days at 5-day cadence")
    parser.add_argument("--horizon", type=int, default=1, help="1 snapshot = next 5-day response")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seasonal-bin-days", type=int, default=15)
    parser.add_argument("--max-states", type=int, default=None, help="Optional cap for testing converter")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = dataset_dir / "metadata" / "states_metadata.csv"
    states_dir = dataset_dir / "states_npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {meta_path}")
    if not states_dir.exists():
        raise FileNotFoundError(f"Missing states_npz directory: {states_dir}. Run downloader with --build-npz.")

    input_channels = BUILDER24_INPUT_CHANNELS if args.profile == "builder24" else CORE20_INPUT_CHANNELS

    df = pd.read_csv(meta_path)
    if "s2_download_ok" in df.columns:
        df = df[df["s2_download_ok"].astype(str).str.lower().isin(["true", "1", "yes"])]
    df["timestamp"] = df["timestamp_utc"].map(parse_dt)
    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)

    # Keep only states with NPZ present.
    rows = []
    for _, r in df.iterrows():
        npz_path = states_dir / f"{r['state_id']}.npz"
        if npz_path.exists():
            rows.append({
                "state_id": str(r["state_id"]),
                "location": str(r["location"]),
                "timestamp": r["timestamp"],
                "npz_path": npz_path,
            })
        if args.max_states and len(rows) >= args.max_states:
            break

    if not rows:
        raise RuntimeError("No usable state NPZ files found.")

    n = len(rows)
    c_in = len(input_channels)
    c_out = len(TARGET_CHANNELS)
    h, w = args.height, args.width

    x_series = np.zeros((n, c_in, h, w), dtype=np.float32)
    y_series = np.zeros((n, c_out, h, w), dtype=np.float32)

    for i, row in enumerate(tqdm(rows, desc="Converting state NPZs")):
        payload = load_npz(row["npz_path"])

        for ci, name in enumerate(input_channels):
            x_series[i, ci] = extract_raw_field(payload, name, h, w)

        y_series[i] = compute_target_raw(payload, h, w)

        row["global_idx"] = i

    # Convert raw 2m temperature fields to seasonal anomalies.
    apply_temperature_anomaly(
        x_series,
        y_series,
        rows,
        input_channels,
        bin_days=args.seasonal_bin_days,
    )

    # Build safe per-location windows.
    train_windows, val_windows, test_windows = [], [], []
    train_state_indices: list[int] = []

    for location, loc_df in pd.DataFrame(rows).groupby("location", sort=False):
        loc_rows = loc_df.sort_values("timestamp").to_dict("records")
        tr, va, te, tr_idx = build_windows_for_location(
            loc_rows,
            input_len=args.input_len,
            horizon=args.horizon,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
        train_windows.extend(tr)
        val_windows.extend(va)
        test_windows.extend(te)
        train_state_indices.extend(tr_idx)

    if not train_windows:
        raise RuntimeError("No train windows created. Need more states per location or smaller input_len.")

    train_state_indices = np.array(sorted(set(train_state_indices)), dtype=np.int64)

    # Normalize using train state range only.
    normalize_and_save(x_series, y_series, train_state_indices, out_dir)

    write_csv(out_dir / "train_windows.csv", train_windows)
    write_csv(out_dir / "val_windows.csv", val_windows)
    write_csv(out_dir / "test_windows.csv", test_windows)

    # Full state index for debugging/reproducibility.
    state_rows = [
        {
            "global_idx": r["global_idx"],
            "state_id": r["state_id"],
            "location": r["location"],
            "timestamp_utc": r["timestamp"].isoformat(),
            "npz_path": str(r["npz_path"]),
        }
        for r in rows
    ]
    write_csv(out_dir / "states_index.csv", state_rows)

    manifest = {
        "profile": args.profile,
        "input_channels": input_channels,
        "target_channels": TARGET_CHANNELS,
        "x_series_shape": list(x_series.shape),
        "y_series_shape": list(y_series.shape),
        "input_len": args.input_len,
        "horizon": args.horizon,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "test_windows": len(test_windows),
        "notes": [
            "x_series/y_series are normalized with train-state statistics.",
            "Window CSVs prevent crossing location boundaries.",
            "Targets are current-state radiation/temp maps; training Dataset uses future windows.",
            "temperature_anomaly is computed as per-location day-of-year-bin anomaly.",
        ],
    }
    (out_dir / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("DONE")
    print(f"x_series: {x_series.shape}")
    print(f"y_series: {y_series.shape}")
    print(f"train windows: {len(train_windows)}")
    print(f"val windows:   {len(val_windows)}")
    print(f"test windows:  {len(test_windows)}")
    print(f"manifest: {out_dir / 'tensor_manifest.json'}")


if __name__ == "__main__":
    main()
