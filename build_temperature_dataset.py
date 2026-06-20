#!/usr/bin/env python3
"""
Build a temperature training dataset from aligned Copernicus and weather data.

The exporter expects the local Node proxy to be running at /api/* so it can reuse
the existing Copernicus auth and request plumbing:

    npm start

It downloads, for each selected Balkan city and each day in the window:
  - a Sentinel-2 cloud mask
  - Sentinel-5P cloud / atmospheric statistics
  - Sentinel-3 OLCI atmospheric statistics
  - Open-Meteo daily weather values for temperature, wind, radiation, rain, and cloud cover

Outputs:
  - JSONL manifest
  - downloaded PNGs
  - metadata.json schema
  - optional dataset.pt if torch is installed

The final samples are day-aligned to the rarest field available in the stack:
the satellite observation day.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
import zlib
import struct
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_SERVER_URL = "http://localhost:5173"
DEFAULT_DAYS = 30
DEFAULT_CITY_RADIUS_KM = 20.0
DEFAULT_IMAGE_WIDTH = 256
DEFAULT_IMAGE_HEIGHT = 256


@dataclass(frozen=True)
class City:
    name: str
    country: str
    lat: float
    lon: float


DEFAULT_CITIES = [
    City("Sofia", "Bulgaria", 42.6977, 23.3219),
    City("Plovdiv", "Bulgaria", 42.1354, 24.7453),
    City("Varna", "Bulgaria", 43.2141, 27.9147),
    City("Burgas", "Bulgaria", 42.5048, 27.4626),
    City("Skopje", "North Macedonia", 41.9973, 21.4280),
    City("Belgrade", "Serbia", 44.7866, 20.4489),
]


SATELLITE_FEATURES = [
    ("s5p_cloud_fraction", "sentinel5p", "cloud_fraction"),
    ("s5p_cloud_optical_thickness", "sentinel5p", "cloud_optical_thickness"),
    ("s5p_cloud_top_height_m", "sentinel5p", "cloud_top_height"),
    ("s5p_cloud_top_pressure_pa", "sentinel5p", "cloud_top_pressure"),
    ("s5p_aerosol_index", "sentinel5p", "aerosol_index"),
    ("s3_humidity_pct", "sentinel3olci", "humidity"),
    ("s3_sea_level_pressure_hpa", "sentinel3olci", "sea_level_pressure"),
    ("s3_water_vapour_kg_m2", "sentinel3olci", "water_vapour"),
]

WEATHER_FIELDS = [
    ("wind_speed_10m_mean", "wind_speed_10m_mean"),
    ("wind_direction_10m_dominant", "wind_direction_10m_dominant"),
    ("shortwave_radiation_sum", "shortwave_radiation_sum"),  # net radiation proxy
    ("precipitation_sum", "precipitation_sum"),
    ("cloud_cover_mean", "cloud_cover_mean"),
    ("surface_pressure_mean", "surface_pressure_mean"),
]

TARGET_FIELD = "temperature_2m_mean"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PyTorch-ready temperature dataset.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Local proxy URL.")
    parser.add_argument("--out", default="dataset_out", help="Output directory.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of days to export.")
    parser.add_argument(
        "--radius-km",
        type=float,
        default=DEFAULT_CITY_RADIUS_KM,
        help="Half-width of the per-city bounding box in kilometers.",
    )
    parser.add_argument(
        "--cities",
        default=",".join(city.name for city in DEFAULT_CITIES),
        help="Comma-separated city names from the built-in Balkan city list.",
    )
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip optional .pt export even if torch is installed.",
    )
    args = parser.parse_args()

    server_url = args.server_url.rstrip("/")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    check_server(server_url)

    city_lookup = {city.name.lower(): city for city in DEFAULT_CITIES}
    selected_cities = []
    for raw_name in [entry.strip() for entry in args.cities.split(",") if entry.strip()]:
        key = raw_name.lower()
        if key not in city_lookup:
            raise SystemExit(f"Unknown city '{raw_name}'. Valid names: {', '.join(city_lookup)}")
        selected_cities.append(city_lookup[key])

    date_range = build_date_range(args.days)
    print(f"Building {len(selected_cities)} cities x {len(date_range)} days")

    manifest: List[Dict[str, object]] = []
    feature_names = [name for name, _, _ in SATELLITE_FEATURES] + [name for name, _ in WEATHER_FIELDS]

    for city in selected_cities:
      # Keep the city loop outermost so weather and satellite stats can be reused per city.
        bbox = bbox_from_center(city.lat, city.lon, args.radius_km)
        weather_by_date = fetch_open_meteo_daily(city, date_range[0], date_range[-1])
        satellite_series = fetch_satellite_series(server_url, city, bbox, args.days)
        merged = merge_city_series(city, bbox, date_range, weather_by_date, satellite_series)

        city_slug = slugify(city.name)
        city_dir = masks_dir / city_slug
        city_dir.mkdir(exist_ok=True)

        for sample in merged:
            mask_path = city_dir / f"{sample['date']}.png"
            if not download_satellite_mask(
                server_url,
                bbox,
                sample["date"],
                mask_path,
                args.image_width,
                args.image_height,
            ):
                continue

            record = {
                "sample_id": f"{city_slug}_{sample['date']}",
                "city": city.name,
                "country": city.country,
                "lat": city.lat,
                "lon": city.lon,
                "bbox": bbox,
                "date": sample["date"],
                "anchor": f"{sample['date']}T00:00:00Z",
                "mask_path": str(mask_path.relative_to(out_dir)).replace("\\", "/"),
                "target_temperature_c": sample["weather"].get(TARGET_FIELD),
                "inputs": sample["inputs"],
                "feature_vector": [float("nan") if sample["inputs"].get(name) is None else sample["inputs"].get(name) for name in feature_names],
                "source_notes": {
                    "mask": "Sentinel-2 cloud mask",
                    "satellite_stats": "Sentinel-5P + Sentinel-3 OLCI daily statistics",
                    "weather": "Open-Meteo daily weather (ERA5-based)",
                    "radiation_note": "shortwave_radiation_sum used as a net-radiation proxy",
                },
            }
            manifest.append(record)

    if not manifest:
        raise SystemExit("No samples were produced. Check network access and the configured dates.")

    manifest.sort(key=lambda row: (row["date"], row["city"]))
    splits = split_records(manifest)

    write_jsonl(out_dir / "dataset.jsonl", manifest)
    write_json(out_dir / "metadata.json", build_metadata(feature_names, selected_cities, args.days, args.radius_km, date_range))
    write_splits(out_dir, splits)

    if not args.skip_torch:
        try:
            import torch  # type: ignore
        except Exception:
            print("torch is not installed, so dataset.pt was not written. The JSONL manifest is ready.")
        else:
            export_torch_bundle(out_dir / "dataset.pt", manifest, splits, feature_names, torch)

    print(f"Wrote {len(manifest)} samples to {out_dir}")
    return 0


def check_server(server_url: str) -> None:
    try:
        with urllib.request.urlopen(f"{server_url}/api/config", timeout=10) as response:
            if response.status != 200:
                raise RuntimeError
    except Exception as exc:
        raise SystemExit(
            f"Could not reach {server_url}. Start the local proxy first with `npm start`."
        ) from exc


def build_date_range(days: int) -> List[str]:
    end = dt.datetime.utcnow().date() - dt.timedelta(days=1)
    return [
        (end - dt.timedelta(days=offset)).isoformat()
        for offset in range(days - 1, -1, -1)
    ]


def bbox_from_center(lat: float, lon: float, radius_km: float) -> List[float]:
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / (111.32 * max(0.2, math.cos(math.radians(lat))))
    return [
        round(lon - lon_delta, 6),
        round(lat - lat_delta, 6),
        round(lon + lon_delta, 6),
        round(lat + lat_delta, 6),
    ]


def fetch_open_meteo_daily(city: City, start_date: str, end_date: str) -> Dict[str, Dict[str, float]]:
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "daily": ",".join(
            [
                "temperature_2m_mean",
                "precipitation_sum",
                "cloud_cover_mean",
                "surface_pressure_mean",
                "shortwave_radiation_sum",
                "wind_speed_10m_mean",
                "wind_direction_10m_dominant",
            ]
        ),
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    payload = json.loads(http_get(url))
    daily = payload.get("daily", {})
    dates = daily.get("time", [])

    out: Dict[str, Dict[str, float]] = {}
    for index, date in enumerate(dates):
        out[date] = {
            TARGET_FIELD: safe_float(daily, "temperature_2m_mean", index),
            "precipitation_sum": safe_float(daily, "precipitation_sum", index),
            "cloud_cover_mean": safe_float(daily, "cloud_cover_mean", index),
            "surface_pressure_mean": safe_float(daily, "surface_pressure_mean", index),
            "shortwave_radiation_sum": safe_float(daily, "shortwave_radiation_sum", index),
            "wind_speed_10m_mean": safe_float(daily, "wind_speed_10m_mean", index),
            "wind_direction_10m_dominant": safe_float(daily, "wind_direction_10m_dominant", index),
        }

    return out


def fetch_satellite_series(
    server_url: str,
    city: City,
    bbox: List[float],
    days: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    series: Dict[str, Dict[str, Dict[str, float]]] = {"sentinel5p": {}, "sentinel3olci": {}}

    for _, source, field in SATELLITE_FEATURES:
        payload = {
            "source": source,
            "field": field,
            "days": days,
            "bbox": bbox,
        }
        response = http_post_json(f"{server_url}/api/statistics", payload)
        points = response.get("points", [])
        for point in points:
            date = point.get("date")
            if not date:
                continue
            series.setdefault(source, {}).setdefault(date, {})[field] = point.get("mean")

    return series


def merge_city_series(
    city: City,
    bbox: List[float],
    date_range: List[str],
    weather_by_date: Dict[str, Dict[str, float]],
    satellite_series: Dict[str, Dict[str, Dict[str, float]]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for date in date_range:
        weather = weather_by_date.get(date, {})
        inputs: Dict[str, Optional[float]] = {}

        for feature_name, source, source_field in SATELLITE_FEATURES:
            value = satellite_series.get(source, {}).get(date, {}).get(source_field)
            inputs[feature_name] = clean_number(value)

        for feature_name, source_field in WEATHER_FIELDS:
            inputs[feature_name] = clean_number(weather.get(source_field))

        target = clean_number(weather.get(TARGET_FIELD))
        required_values = [
            target,
            inputs["s5p_cloud_fraction"],
            inputs["s5p_cloud_optical_thickness"],
            inputs["s5p_cloud_top_height_m"],
            inputs["s5p_cloud_top_pressure_pa"],
            inputs["wind_speed_10m_mean"],
            inputs["shortwave_radiation_sum"],
            inputs["precipitation_sum"],
            inputs["cloud_cover_mean"],
        ]

        if any(value is None for value in required_values):
            continue

        rows.append(
            {
                "city": city.name,
                "bbox": bbox,
                "date": date,
                "weather": weather,
                "inputs": inputs,
                "target": target,
            }
        )

    return rows


def download_satellite_mask(
    server_url: str,
    bbox: List[float],
    date: str,
    mask_path: Path,
    width: int,
    height: int,
) -> bool:
    payload = {
        "source": "sentinel2",
        "field": "cloud_mask",
        "item": {"date": date, "label": date},
        "bbox": bbox,
        "width": width,
        "height": height,
    }

    try:
        data = http_post_bytes(f"{server_url}/api/process", payload)
    except Exception:
        return False

    if not data.startswith(PNG_SIGNATURE):
        return False

    if is_empty_mask_png(data):
        if mask_path.exists():
            mask_path.unlink()
        return False

    mask_path.write_bytes(data)
    return True


def is_empty_mask_png(data: bytes) -> bool:
    if not data.startswith(PNG_SIGNATURE):
        return False

    try:
        width, height, bit_depth, color_type, interlace, pixels = decode_png_pixels(data)
    except Exception:
        return False

    if width <= 0 or height <= 0:
        return True

    if not pixels:
        return True

    # Fully transparent images are the most common "empty" Sentinel outputs.
    if all(pixel[3] == 0 for pixel in pixels):
        return True

    # Near-empty: if almost all pixels are transparent, treat it as no-data.
    opaque_pixels = [pixel for pixel in pixels if pixel[3] > 0]
    if len(opaque_pixels) <= max(1, int(len(pixels) * 0.01)):
        return True

    return False


def decode_png_pixels(data: bytes) -> Tuple[int, int, int, int, int, List[Tuple[int, int, int, int]]]:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG file.")

    offset = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = interlace = None
    idat_parts: List[bytes] = []

    while offset < len(data):
        if offset + 8 > len(data):
            break

        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + length
        chunk_data = data[chunk_data_start:chunk_data_end]
        offset = chunk_data_end + 4

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if None in {width, height, bit_depth, color_type, interlace}:
        raise ValueError("Missing PNG header.")
    if bit_depth != 8 or interlace != 0:
        raise ValueError("Unsupported PNG encoding.")

    raw = zlib.decompress(b"".join(idat_parts))
    pixels = decode_png_scanlines(raw, int(width), int(height), int(color_type))
    return int(width), int(height), int(bit_depth), int(color_type), int(interlace), pixels


def decode_png_scanlines(raw: bytes, width: int, height: int, color_type: int) -> List[Tuple[int, int, int, int]]:
    channels_by_color_type = {
        0: 1,  # grayscale
        2: 3,  # RGB
        4: 2,  # grayscale + alpha
        6: 4,  # RGBA
    }
    channels = channels_by_color_type.get(color_type)
    if channels is None:
        raise ValueError(f"Unsupported PNG color type: {color_type}")

    row_bytes = width * channels
    pixels: List[Tuple[int, int, int, int]] = []
    previous_row = bytes(row_bytes)
    index = 0

    for _ in range(height):
        if index >= len(raw):
            break

        filter_type = raw[index]
        index += 1
        row = bytearray(raw[index : index + row_bytes])
        index += row_bytes

        if len(row) < row_bytes:
            raise ValueError("Truncated PNG data.")

        if filter_type == 1:
            for i in range(channels, len(row)):
                row[i] = (row[i] + row[i - channels]) & 0xFF
        elif filter_type == 2:
            for i in range(len(row)):
                row[i] = (row[i] + previous_row[i]) & 0xFF
        elif filter_type == 3:
            for i in range(len(row)):
                left = row[i - channels] if i >= channels else 0
                up = previous_row[i]
                row[i] = (row[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(len(row)):
                left = row[i - channels] if i >= channels else 0
                up = previous_row[i]
                up_left = previous_row[i - channels] if i >= channels else 0
                row[i] = (row[i] + paeth_predictor(left, up, up_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"Unsupported PNG filter: {filter_type}")

        previous_row = bytes(row)

        for start in range(0, len(row), channels):
            chunk = row[start : start + channels]
            if channels == 1:
                gray = chunk[0]
                pixels.append((gray, gray, gray, 255))
            elif channels == 2:
                gray, alpha = chunk
                pixels.append((gray, gray, gray, alpha))
            elif channels == 3:
                r, g, b = chunk
                pixels.append((r, g, b, 255))
            else:
                r, g, b, a = chunk
                pixels.append((r, g, b, a))

    return pixels


def paeth_predictor(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)

    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def split_records(records: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    dates = sorted({str(record["date"]) for record in records})
    if not dates:
        return {"train": [], "val": [], "test": []}

    train_cut = max(1, int(len(dates) * 0.7))
    val_cut = max(train_cut + 1, int(len(dates) * 0.85))
    date_to_split = {}
    for idx, date in enumerate(dates):
        if idx < train_cut:
            date_to_split[date] = "train"
        elif idx < val_cut:
            date_to_split[date] = "val"
        else:
            date_to_split[date] = "test"

    splits = {"train": [], "val": [], "test": []}
    for record in records:
        splits[date_to_split[str(record["date"])]].append(record)
    return splits


def write_splits(out_dir: Path, splits: Dict[str, List[Dict[str, object]]]) -> None:
    split_dir = out_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(split_dir / f"{name}.jsonl", rows)


def build_metadata(
    feature_names: List[str],
    cities: List[City],
    days: int,
    radius_km: float,
    date_range: List[str],
) -> Dict[str, object]:
    return {
        "target": TARGET_FIELD,
        "feature_names": feature_names,
        "cities": [
            {"name": city.name, "country": city.country, "lat": city.lat, "lon": city.lon}
            for city in cities
        ],
        "days": days,
        "radius_km": radius_km,
        "date_range": {"start": date_range[0], "end": date_range[-1]},
        "notes": [
            "Sentinel-2 cloud masks are the visual spatial input.",
            "Cloud composition uses Sentinel-5P and Sentinel-3 OLCI statistics.",
            "Weather comes from Open-Meteo historical data, which is ERA5-based.",
            "shortwave_radiation_sum is used as a net-radiation proxy.",
            "Records are day-aligned to the satellite observation day.",
        ],
    }


def export_torch_bundle(
    out_path: Path,
    records: List[Dict[str, object]],
    splits: Dict[str, List[Dict[str, object]]],
    feature_names: List[str],
    torch,
) -> None:
    index_by_sample = {record["sample_id"]: idx for idx, record in enumerate(records)}
    feature_rows = []
    targets = []
    mask_paths = []
    sample_ids = []
    cities = []
    dates = []

    for record in records:
        feature_rows.append([record["inputs"].get(name) for name in feature_names])
        targets.append(record["target_temperature_c"])
        mask_paths.append(record["mask_path"])
        sample_ids.append(record["sample_id"])
        cities.append(record["city"])
        dates.append(record["date"])

    payload = {
        "feature_names": feature_names,
        "target_name": TARGET_FIELD,
        "mask_paths": mask_paths,
        "sample_ids": sample_ids,
        "cities": cities,
        "dates": dates,
        "features": torch.tensor(feature_rows, dtype=torch.float32),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "splits": {
            split_name: [index_by_sample[row["sample_id"]] for row in rows]
            for split_name, rows in splits.items()
        },
        "metadata": {
            "note": "Mask decoding is intentionally left on disk so the bundle stays small.",
        },
    }
    torch.save(payload, out_path)


def http_post_json(url: str, payload: Dict[str, object]) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def http_post_bytes(url: str, payload: Dict[str, object]) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8")


def safe_float(data: Dict[str, List[float]], key: str, index: int) -> Optional[float]:
    values = data.get(key, [])
    if index >= len(values):
        return None
    return clean_number(values[index])


def clean_number(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def slugify(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
