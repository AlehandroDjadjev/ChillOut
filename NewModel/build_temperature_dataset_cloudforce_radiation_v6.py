#!/usr/bin/env python3
"""
build_temperature_dataset_cloudforce.py

Fast cloud-focused dataset builder for the CNN/MLP temperature model family.

It replaces the old local-proxy daily builder with:
  - Copernicus Data Space / Sentinel Hub direct Sentinel-2 L2A search + Process API
  - cloud-only grayscale masks where non-cloud pixels are blacked out
  - image-derived cloud scalar features
  - Open-Meteo hourly historical weather for world-state features and future temperature target

Important design choices:
  - Temperature is NOT used as an input feature.
  - Open-Meteo scalar cloud-cover fields are NOT used as input features.
  - Only one solar/radiation context field is kept: world_shortwave_radiation.
  - Cloud signal comes mainly from Sentinel-2 cloud pixels + cloud-derived scalar features.
  - Output target is a future temperature number, default +5 days after the S2 scene timestamp.

Credentials:
  Put .env next to this script or export:
    SH_CLIENT_ID=...
    SH_CLIENT_SECRET=...

Install:
  pip install sentinelhub numpy pandas pillow pyproj python-dateutil tqdm requests python-dotenv torch

Example:
  python build_temperature_dataset_cloudforce.py ^
    --start-date 2023-01-01 ^
    --end-date 2026-01-01 ^
    --out dataset_cloudforce ^
    --max-scenes-per-location 200 ^
    --resolution-m 250 ^
    --patch-km 10 ^
    --s2-workers 8 ^
    --openmeteo-workers 1 ^
    --target-offset-days 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

SCRIPT_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(SCRIPT_DIR / ".env")

import numpy as np
import requests
from PIL import Image
from dateutil.parser import isoparse
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from tqdm import tqdm

from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
)


SENTINEL2_L2A_CDSE = DataCollection.SENTINEL2_L2A.define_from(
    name="sentinel-2-l2a-cdse",
    service_url="https://sh.dataspace.copernicus.eu",
)

DEFAULT_LOCATIONS = [
    {"name": "sofia_bg", "city": "Sofia", "country": "Bulgaria", "lat": 42.6977, "lon": 23.3219},
    {"name": "athens_gr", "city": "Athens", "country": "Greece", "lat": 37.9838, "lon": 23.7275},
    {"name": "london_uk", "city": "London", "country": "United Kingdom", "lat": 51.5074, "lon": -0.1278},
    {"name": "oslo_no", "city": "Oslo", "country": "Norway", "lat": 59.9139, "lon": 10.7522},
    {"name": "cairo_eg", "city": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357},
    {"name": "reykjavik_is", "city": "Reykjavik", "country": "Iceland", "lat": 64.1466, "lon": -21.9426},
    {"name": "lisbon_pt", "city": "Lisbon", "country": "Portugal", "lat": 38.7223, "lon": -9.1393},
    {"name": "milan_it", "city": "Milan", "country": "Italy", "lat": 45.4642, "lon": 9.1900},
    {"name": "casablanca_ma", "city": "Casablanca", "country": "Morocco", "lat": 33.5731, "lon": -7.5898},
    {"name": "helsinki_fi", "city": "Helsinki", "country": "Finland", "lat": 60.1699, "lon": 24.9384},
]

CLOUD_FEATURE_NAMES = [
    "cloud_s2_fraction",
    "cloud_s2_prob_mean",
    "cloud_s2_prob_std",
    "cloud_s2_prob_p90",
    "cloud_s2_aot_mean",
    "cloud_s2_cirrus_fraction",
    "cloud_s2_high_fraction",
    "cloud_s2_medium_fraction",
    "cloud_s2_edge_density",
    "cloud_s2_texture_std",
]

WORLD_FEATURE_NAMES = [
    # Current temperature is intentionally NOT a model input in v3.
    # It is still stored as current_temperature_c for persistence benchmarking.
    # No Open-Meteo cloud-cover inputs.
    # One solar/radiation context only.
    "world_shortwave_radiation",
    "world_relative_humidity_2m",
    "world_dew_point_2m",
    "world_surface_pressure",
    "world_pressure_msl",
    "world_wind_speed_10m",
    "world_wind_u_10m",
    "world_wind_v_10m",
    "world_precipitation",
    "world_snow_depth",
    "world_vapour_pressure_deficit",
]

RAW_FEATURE_NAMES = CLOUD_FEATURE_NAMES + WORLD_FEATURE_NAMES
TARGET_FIELD = "target_temperature_c"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snow_depth",
    "pressure_msl",
    "surface_pressure",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "vapour_pressure_deficit",
]

S2_CLOUD_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "SCL", "CLP", "AOT", "dataMask"]
    }],
    output: {
      bands: 8,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(s) {
  let mediumCloud = (s.SCL === 8) ? 1.0 : 0.0;
  let highCloud = (s.SCL === 9) ? 1.0 : 0.0;
  let cirrus = (s.SCL === 10) ? 1.0 : 0.0;
  let isCloud = ((mediumCloud + highCloud + cirrus) > 0.0 && s.dataMask > 0) ? 1.0 : 0.0;

  let clp = isCloud * Math.max(0.0, Math.min(1.0, s.CLP / 255.0));
  let gray = isCloud * ((s.B02 + s.B03 + s.B04) / 3.0);
  let cloudWhite = isCloud;

  return [
    isCloud,              // 0 hard SCL cloud mask
    clp,                  // 1 cloud probability, non-cloud black
    cloudWhite,           // 2 white cloud, black world
    cirrus,               // 3 thin cirrus
    highCloud,            // 4 high-prob cloud
    mediumCloud,          // 5 medium-prob cloud
    isCloud * s.AOT,      // 6 AOT only on cloud pixels
    gray                  // 7 cloud-only visible gray
  ];
}
"""


@dataclass(frozen=True)
class Location:
    name: str
    city: str
    country: str
    lat: float
    lon: float


@dataclass(frozen=True)
class SentinelScene:
    location: str
    item_id: str
    timestamp: str
    cloud_cover: Optional[float]
    bbox_wgs84: Tuple[float, float, float, float]


def parse_date(value: str) -> datetime:
    if len(value) == 10:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    dt = isoparse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def nearest_hour(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    base = dt.replace(minute=0, second=0, microsecond=0)
    if dt.minute >= 30:
        base += timedelta(hours=1)
    return base


def safe_name(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "unnamed"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def openmeteo_wait_seconds(status_code: int, body: str, attempt: int) -> float:
    """Conservative retry delay for Open-Meteo transient errors.

    429s can happen even for a small number of requests if they land inside the
    same minute window. Do not fail the build; wait through the window.
    """
    if status_code == 429:
        return 70.0 + random.uniform(0.0, 10.0)
    if status_code in {408, 500, 502, 503, 504}:
        return min(120.0, 10.0 * (2 ** max(0, attempt - 1))) + random.uniform(0.0, 5.0)
    return 0.0


def read_locations(path: Optional[str]) -> List[Location]:
    if path is None:
        return [Location(**x) for x in DEFAULT_LOCATIONS]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Location(**x) for x in data]


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def bbox_around_point_wgs84(lon: float, lat: float, patch_km: float) -> Tuple[float, float, float, float]:
    epsg = utm_epsg_for_lonlat(lon, lat)
    wgs84 = PyprojCRS.from_epsg(4326)
    utm = PyprojCRS.from_epsg(epsg)
    to_utm = Transformer.from_crs(wgs84, utm, always_xy=True)
    to_wgs = Transformer.from_crs(utm, wgs84, always_xy=True)
    x, y = to_utm.transform(lon, lat)
    half_m = patch_km * 1000.0 / 2.0
    corners = [
        to_wgs.transform(px, py)
        for px, py in [
            (x - half_m, y - half_m),
            (x - half_m, y + half_m),
            (x + half_m, y - half_m),
            (x + half_m, y + half_m),
        ]
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return min(lons), min(lats), max(lons), max(lats)


def image_size_from_patch(patch_km: float, resolution_m: float) -> Tuple[int, int]:
    n = max(16, int(round((patch_km * 1000.0) / resolution_m)))
    return n, n


def make_sh_config() -> SHConfig:
    client_id = os.environ.get("SH_CLIENT_ID")
    client_secret = os.environ.get("SH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Missing SH_CLIENT_ID / SH_CLIENT_SECRET")
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return config


def catalog_search_s2(
    config: SHConfig,
    loc: Location,
    bbox_wgs84: Tuple[float, float, float, float],
    start_dt: datetime,
    end_dt: datetime,
    max_cloud: float,
    limit: Optional[int],
) -> List[SentinelScene]:
    catalog = SentinelHubCatalog(config=config)
    bbox = BBox(bbox_wgs84, crs=CRS.WGS84)
    search_iter = catalog.search(
        SENTINEL2_L2A_CDSE,
        bbox=bbox,
        time=(start_dt.isoformat(), end_dt.isoformat()),
        filter=f"eo:cloud_cover <= {float(max_cloud)}",
        fields={
            "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
            "exclude": [],
        },
    )

    scenes: List[SentinelScene] = []
    seen: set[str] = set()
    for item in search_iter:
        props = item.get("properties", {})
        ts = props.get("datetime")
        item_id = item.get("id", "")
        if not ts or not item_id:
            continue
        key = f"{item_id}|{ts}"
        if key in seen:
            continue
        seen.add(key)
        scenes.append(
            SentinelScene(
                location=loc.name,
                item_id=item_id,
                timestamp=ts,
                cloud_cover=props.get("eo:cloud_cover"),
                bbox_wgs84=bbox_wgs84,
            )
        )
        if limit is not None and len(scenes) >= limit:
            break

    scenes.sort(key=lambda s: s.timestamp)
    return scenes


def download_s2_cloud_tensor(
    config: SHConfig,
    scene: SentinelScene,
    size: Tuple[int, int],
    out_npz: Path,
    out_png: Path,
    retries: int = 3,
) -> Optional[Dict[str, float]]:
    if out_npz.exists() and out_png.exists():
        try:
            arr = np.load(out_npz)["cloud_tensor"]
            return compute_cloud_features(arr)
        except Exception:
            pass

    ts = parse_date(scene.timestamp)
    start = (ts - timedelta(minutes=30)).isoformat()
    end = (ts + timedelta(minutes=30)).isoformat()

    request = SentinelHubRequest(
        evalscript=S2_CLOUD_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=SENTINEL2_L2A_CDSE,
                time_interval=(start, end),
                mosaicking_order="leastCC",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=BBox(scene.bbox_wgs84, crs=CRS.WGS84),
        size=size,
        config=config,
    )

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            arr = np.asarray(request.get_data()[0], dtype=np.float32)
            if arr.ndim != 3 or arr.shape[-1] != 8:
                raise RuntimeError(f"Unexpected cloud tensor shape {arr.shape}")

            ensure_dir(out_npz.parent)
            ensure_dir(out_png.parent)
            np.savez_compressed(out_npz, cloud_tensor=arr)

            # Cloud-focused image: non-cloud black, cloud intensity from probability if available.
            cloud_mask = np.clip(arr[..., 0], 0.0, 1.0)
            cloud_prob = np.clip(arr[..., 1], 0.0, 1.0)
            cloud_img = np.where(cloud_prob > 0, cloud_prob, cloud_mask)
            png = np.clip(cloud_img * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(png, mode="L").save(out_png)

            return compute_cloud_features(arr)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(2.0 * attempt)
                continue

    print(f"[S2 ERROR] {scene.location} {scene.timestamp}: {last_error}", file=sys.stderr)
    return None


def _mean_on_cloud(values: np.ndarray, mask: np.ndarray) -> float:
    if float(mask.sum()) <= 0:
        return 0.0
    return float(values[mask > 0.5].mean())


def compute_edge_density(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    m = (mask > 0.5).astype(np.float32)
    dx = np.abs(m[:, 1:] - m[:, :-1]).mean() if m.shape[1] > 1 else 0.0
    dy = np.abs(m[1:, :] - m[:-1, :]).mean() if m.shape[0] > 1 else 0.0
    return float((dx + dy) / 2.0)


def compute_cloud_features(arr: np.ndarray) -> Dict[str, float]:
    mask = np.clip(arr[..., 0], 0.0, 1.0)
    prob = np.clip(arr[..., 1], 0.0, 1.0)
    cirrus = np.clip(arr[..., 3], 0.0, 1.0)
    high = np.clip(arr[..., 4], 0.0, 1.0)
    medium = np.clip(arr[..., 5], 0.0, 1.0)
    aot = np.clip(arr[..., 6], 0.0, None)
    gray = np.clip(arr[..., 7], 0.0, None)

    cloud_pixels = prob[mask > 0.5]
    if cloud_pixels.size == 0:
        prob_mean = prob_std = prob_p90 = texture_std = 0.0
    else:
        prob_mean = float(cloud_pixels.mean())
        prob_std = float(cloud_pixels.std())
        prob_p90 = float(np.percentile(cloud_pixels, 90))
        texture_std = float(gray[mask > 0.5].std())

    return {
        "cloud_s2_fraction": float(mask.mean()),
        "cloud_s2_prob_mean": prob_mean,
        "cloud_s2_prob_std": prob_std,
        "cloud_s2_prob_p90": prob_p90,
        "cloud_s2_aot_mean": _mean_on_cloud(aot, mask),
        "cloud_s2_cirrus_fraction": float(cirrus.mean()),
        "cloud_s2_high_fraction": float(high.mean()),
        "cloud_s2_medium_fraction": float(medium.mean()),
        "cloud_s2_edge_density": compute_edge_density(mask),
        "cloud_s2_texture_std": texture_std,
    }


def fetch_openmeteo_hourly(
    loc: Location,
    start_date: str,
    end_date: str,
    cache_path: Path,
    model: str,
    retries: int = 8,
    request_sleep_s: float = 0.0,
) -> Dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    ensure_dir(cache_path.parent)
    params = {
        "latitude": loc.lat,
        "longitude": loc.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(OPEN_METEO_HOURLY_VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "cell_selection": "nearest",
    }
    if model and model.lower() not in {"best_match", "default", "auto"}:
        params["models"] = model

    last_error = None
    for attempt in range(1, retries + 1):
        if request_sleep_s > 0:
            time.sleep(request_sleep_s + random.uniform(0.0, 1.5))

        try:
            r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=120)
            if r.status_code >= 400:
                body = r.text[:1000]
                wait_s = openmeteo_wait_seconds(r.status_code, body, attempt)
                if wait_s > 0 and attempt < retries:
                    print(
                        f"[Open-Meteo retry] {loc.name}: HTTP {r.status_code}; "
                        f"waiting {wait_s:.0f}s then retrying ({attempt}/{retries})",
                        flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(f"Open-Meteo HTTP {r.status_code}: {body}")

            payload = r.json()
            if payload.get("error"):
                body = json.dumps(payload)[:1000]
                wait_s = openmeteo_wait_seconds(429 if "limit" in body.lower() else 500, body, attempt)
                if wait_s > 0 and attempt < retries:
                    print(
                        f"[Open-Meteo retry] {loc.name}: API error; "
                        f"waiting {wait_s:.0f}s then retrying ({attempt}/{retries})",
                        flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(body)

            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload

        except requests.RequestException as e:
            last_error = e
            if attempt < retries:
                wait_s = min(120.0, 10.0 * (2 ** max(0, attempt - 1))) + random.uniform(0.0, 5.0)
                print(
                    f"[Open-Meteo retry] {loc.name}: network error {e}; "
                    f"waiting {wait_s:.0f}s then retrying ({attempt}/{retries})",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
        except Exception as e:
            last_error = e
            if attempt < retries:
                wait_s = min(120.0, 10.0 * (2 ** max(0, attempt - 1))) + random.uniform(0.0, 5.0)
                print(
                    f"[Open-Meteo retry] {loc.name}: {e}; "
                    f"waiting {wait_s:.0f}s then retrying ({attempt}/{retries})",
                    flush=True,
                )
                time.sleep(wait_s)
                continue

    raise RuntimeError(f"Open-Meteo failed for {loc.name}: {last_error}")


def nearest_hour_row(payload: Dict[str, Any], timestamp: datetime) -> Optional[Dict[str, float]]:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None

    key = nearest_hour(timestamp).strftime("%Y-%m-%dT%H:00")
    try:
        idx = times.index(key)
    except ValueError:
        return None

    row: Dict[str, float] = {}
    for name, values in hourly.items():
        if name == "time":
            continue
        if isinstance(values, list) and idx < len(values):
            value = values[idx]
            if value is None:
                return None
            try:
                row[name] = float(value)
            except Exception:
                return None
    return row


def wind_uv(speed_ms: float, direction_deg: float) -> Tuple[float, float]:
    rad = math.radians(direction_deg)
    return -speed_ms * math.sin(rad), -speed_ms * math.cos(rad)


def world_features_from_row(row: Dict[str, float]) -> Dict[str, float]:
    speed = float(row.get("wind_speed_10m", 0.0))
    direction = float(row.get("wind_direction_10m", 0.0))
    u, v = wind_uv(speed, direction)

    return {
        "world_shortwave_radiation": float(row.get("shortwave_radiation", 0.0)),
        "world_relative_humidity_2m": float(row.get("relative_humidity_2m", 0.0)) / 100.0,
        "world_dew_point_2m": float(row.get("dew_point_2m", 0.0)),
        "world_surface_pressure": float(row.get("surface_pressure", 0.0)) * 100.0,
        "world_pressure_msl": float(row.get("pressure_msl", 0.0)) * 100.0,
        "world_wind_speed_10m": speed,
        "world_wind_u_10m": float(u),
        "world_wind_v_10m": float(v),
        "world_precipitation": float(row.get("precipitation", 0.0)),
        "world_snow_depth": float(row.get("snow_depth", 0.0)),
        "world_vapour_pressure_deficit": float(row.get("vapour_pressure_deficit", 0.0)),
    }



def solar_clear_sky_proxy_wm2(lat: float, lon: float, dt: datetime) -> float:
    """Simple clear-sky shortwave proxy.

    It is not a full radiative-transfer model. It gives a physically grounded
    daylight-scale reference so the cloud target can be:

        cloud_radiative_loss = clear_sky_proxy - observed_shortwave

    This makes the target much more cloud-centered than raw future temperature.
    """
    dt = dt.astimezone(timezone.utc)
    day = int(dt.strftime("%j"))
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    # NOAA-style approximation.
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

    cos_zenith = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    )
    if cos_zenith <= 0.0:
        return 0.0

    # Extraterrestrial correction + clear-sky transmittance proxy.
    ecc = 1.0 + 0.033 * math.cos(2.0 * math.pi * day / 365.0)
    toa = 1361.0 * ecc * cos_zenith
    clear = toa * 0.72
    return float(max(0.0, clear))


def radiation_targets_from_shortwave(loc: Location, timestamp: datetime, observed_shortwave_wm2: float) -> Dict[str, float]:
    clear = solar_clear_sky_proxy_wm2(loc.lat, loc.lon, timestamp)
    observed = float(observed_shortwave_wm2)
    daylight = 1.0 if clear >= 50.0 else 0.0

    if daylight > 0:
        transmission = max(0.0, min(1.5, observed / max(clear, 1e-6)))
        attenuation = max(-0.5, min(1.2, 1.0 - transmission))
        loss = max(-250.0, min(1200.0, clear - observed))
    else:
        transmission = 1.0
        attenuation = 0.0
        loss = 0.0

    return {
        "radiation_shortwave_observed_wm2": observed,
        "radiation_clear_sky_proxy_wm2": clear,
        "radiation_cloud_loss_wm2": loss,
        "radiation_cloud_attenuation": attenuation,
        "radiation_cloud_transmission": transmission,
        "radiation_daylight_valid": daylight,
    }


def split_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    # Date split to avoid same-day leakage across cities.
    dates = sorted({str(r["date"]) for r in records})
    if not dates:
        return {"train": [], "val": [], "test": []}
    train_cut = max(1, int(len(dates) * 0.70))
    val_cut = max(train_cut + 1, int(len(dates) * 0.85))
    date_split = {}
    for i, d in enumerate(dates):
        date_split[d] = "train" if i < train_cut else ("val" if i < val_cut else "test")

    out = {"train": [], "val": [], "test": []}
    for r in records:
        out[date_split[str(r["date"])]].append(r)
    return out


def export_torch_bundle(out_path: Path, records: List[Dict[str, Any]], splits: Dict[str, List[Dict[str, Any]]]) -> None:
    try:
        import torch
    except Exception:
        print("torch not installed; skipped dataset.pt")
        return

    features = np.asarray([[float(r["inputs"][name]) for name in RAW_FEATURE_NAMES] for r in records], dtype=np.float32)
    targets = np.asarray([float(r[TARGET_FIELD]) for r in records], dtype=np.float32)

    index_by_id = {r["sample_id"]: i for i, r in enumerate(records)}
    train_idx = np.asarray([index_by_id[r["sample_id"]] for r in splits["train"]], dtype=np.int64)
    train_features = features[train_idx] if len(train_idx) else features
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[~np.isfinite(std)] = 1.0
    std[std == 0.0] = 1.0

    payload = {
        "features": torch.from_numpy((features - mean) / std),
        "raw_features": torch.from_numpy(features),
        "targets": torch.from_numpy(targets),
        "raw_feature_names": RAW_FEATURE_NAMES,
        "feature_names": RAW_FEATURE_NAMES,
        "cloud_feature_names": CLOUD_FEATURE_NAMES,
        "world_feature_names": WORLD_FEATURE_NAMES,
        "target_name": TARGET_FIELD,
        "x_mean": torch.from_numpy(mean.astype(np.float32)),
        "x_std": torch.from_numpy(std.astype(np.float32)),
        "mask_paths": [r["mask_path"] for r in records],
        "sample_ids": [r["sample_id"] for r in records],
        "cities": [r["city"] for r in records],
        "dates": [r["date"] for r in records],
        "splits": {k: [index_by_id[r["sample_id"]] for r in rows] for k, rows in splits.items()},
    }
    torch.save(payload, out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out", default="dataset_cloudforce")
    parser.add_argument("--locations-json", default=None)
    parser.add_argument("--patch-km", type=float, default=10.0)
    parser.add_argument("--resolution-m", type=float, default=250.0)
    parser.add_argument("--max-cloud", type=float, default=100.0)
    parser.add_argument("--max-scenes-per-location", type=int, default=200)
    parser.add_argument("--s2-workers", type=int, default=8)
    parser.add_argument("--openmeteo-workers", type=int, default=1, help="Use 1 for Open-Meteo free API safety. Higher can hit 429.")
    parser.add_argument("--openmeteo-retries", type=int, default=8)
    parser.add_argument("--openmeteo-sleep-s", type=float, default=8.0, help="Sleep before each uncached Open-Meteo request.")
    parser.add_argument("--openmeteo-continue-on-fail", action="store_true", help="Skip failed locations instead of aborting.")
    parser.add_argument("--openmeteo-model", default="best_match")
    parser.add_argument("--target-offset-days", type=float, default=5.0)
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    start_dt = parse_date(args.start_date)
    end_dt = parse_date(args.end_date)
    out_dir = Path(args.out).resolve()

    masks_dir = out_dir / "masks"
    tensors_dir = out_dir / "cloud_tensors"
    meteo_dir = out_dir / "openmeteo"
    for p in [out_dir, masks_dir, tensors_dir, meteo_dir, out_dir / "splits"]:
        ensure_dir(p)

    locations = read_locations(args.locations_json)
    size = image_size_from_patch(args.patch_km, args.resolution_m)

    sh_config = make_sh_config()

    loc_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    scenes_by_loc: Dict[str, List[SentinelScene]] = {}

    print("[1/4] Searching Sentinel-2 scenes")
    for loc in locations:
        bbox = bbox_around_point_wgs84(loc.lon, loc.lat, args.patch_km)
        loc_bboxes[loc.name] = bbox
        scenes = catalog_search_s2(
            sh_config,
            loc,
            bbox,
            start_dt,
            end_dt,
            max_cloud=args.max_cloud,
            limit=args.max_scenes_per_location,
        )
        scenes_by_loc[loc.name] = scenes
        print(f"  {loc.name}: {len(scenes)} scenes")

    print("[2/4] Downloading Open-Meteo hourly data")
    meteo_by_loc: Dict[str, Dict[str, Any]] = {}
    target_end_dt = end_dt + timedelta(days=float(args.target_offset_days) + 2)
    start_date = start_dt.date().isoformat()
    target_end_date = target_end_dt.date().isoformat()

    def fetch_loc(loc: Location) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        cache_path = meteo_dir / f"openmeteo_{loc.name}_{start_date}_{target_end_date}.json"
        try:
            payload = fetch_openmeteo_hourly(
                loc,
                start_date,
                target_end_date,
                cache_path,
                args.openmeteo_model,
                retries=int(args.openmeteo_retries),
                request_sleep_s=float(args.openmeteo_sleep_s),
            )
            return loc.name, payload, None
        except Exception as exc:
            return loc.name, None, str(exc)

    # Open-Meteo can rate-limit bursts very aggressively. Default is sequential,
    # but cached files still return instantly, so reruns remain fast.
    with ThreadPoolExecutor(max_workers=max(1, args.openmeteo_workers)) as ex:
        futures = [ex.submit(fetch_loc, loc) for loc in locations]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Open-Meteo"):
            name, payload, error = fut.result()
            if payload is not None:
                meteo_by_loc[name] = payload
            else:
                message = f"[OPEN-METEO FAILED] {name}: {error}"
                if args.openmeteo_continue_on_fail:
                    print(message + " -- skipping this location", file=sys.stderr)
                    continue
                raise RuntimeError(message)

    print("[3/4] Downloading Sentinel-2 cloud-only masks/tensors")
    loc_by_name = {loc.name: loc for loc in locations}
    usable_locations = [loc for loc in locations if loc.name in meteo_by_loc]
    if not usable_locations:
        raise SystemExit("No Open-Meteo payloads available; cannot build samples.")

    all_pairs: List[Tuple[Location, SentinelScene]] = []
    for loc in usable_locations:
        all_pairs.extend((loc, scene) for scene in scenes_by_loc[loc.name])

    def process_scene(pair: Tuple[Location, SentinelScene]) -> Optional[Dict[str, Any]]:
        loc, scene = pair
        ts = parse_date(scene.timestamp)
        target_ts = ts + timedelta(days=float(args.target_offset_days))

        weather_now = nearest_hour_row(meteo_by_loc[loc.name], ts)
        weather_target = nearest_hour_row(meteo_by_loc[loc.name], target_ts)
        if weather_now is None or weather_target is None:
            return None

        sample_id = f"{safe_name(loc.name)}_{nearest_hour(ts).strftime('%Y%m%dT%H%M%SZ')}_{safe_name(scene.item_id)[:24]}"
        rel_png = Path("masks") / loc.name / f"{sample_id}.png"
        rel_npz = Path("cloud_tensors") / loc.name / f"{sample_id}.npz"

        cloud_features = download_s2_cloud_tensor(
            sh_config,
            scene,
            size=size,
            out_npz=out_dir / rel_npz,
            out_png=out_dir / rel_png,
        )
        if cloud_features is None:
            return None

        world_features = world_features_from_row(weather_now)
        radiation_targets = radiation_targets_from_shortwave(loc, ts, float(weather_now.get("shortwave_radiation", 0.0)))
        target_radiation_targets = radiation_targets_from_shortwave(loc, target_ts, float(weather_target.get("shortwave_radiation", 0.0)))
        inputs = {**cloud_features, **world_features}

        if any(name not in inputs or not math.isfinite(float(inputs[name])) for name in RAW_FEATURE_NAMES):
            return None

        target_temp = float(weather_target["temperature_2m"])

        return {
            "sample_id": sample_id,
            "city": loc.city,
            "location": loc.name,
            "country": loc.country,
            "lat": loc.lat,
            "lon": loc.lon,
            "bbox": list(scene.bbox_wgs84),
            "date": nearest_hour(ts).date().isoformat(),
            "anchor": nearest_hour(ts).isoformat(),
            "target_timestamp": nearest_hour(target_ts).isoformat(),
            "target_offset_days": float(args.target_offset_days),
            "sentinel_item_id": scene.item_id,
            "sentinel_scene_cloud_cover": scene.cloud_cover,
            "mask_path": str(rel_png).replace("\\", "/"),
            "cloud_tensor_path": str(rel_npz).replace("\\", "/"),
            "inputs": {name: float(inputs[name]) for name in RAW_FEATURE_NAMES},
            "feature_vector": [float(inputs[name]) for name in RAW_FEATURE_NAMES],
            "current_temperature_c": float(weather_now["temperature_2m"]),
            TARGET_FIELD: target_temp,
            # Radiation-centered cloud targets at the Sentinel-2 scene time.
            # These are the targets that should be most directly controlled by clouds.
            **radiation_targets,
            "target_radiation_shortwave_observed_wm2": float(target_radiation_targets["radiation_shortwave_observed_wm2"]),
            "target_radiation_clear_sky_proxy_wm2": float(target_radiation_targets["radiation_clear_sky_proxy_wm2"]),
            "target_radiation_cloud_loss_wm2": float(target_radiation_targets["radiation_cloud_loss_wm2"]),
            "target_radiation_cloud_attenuation": float(target_radiation_targets["radiation_cloud_attenuation"]),
            "target_radiation_cloud_transmission": float(target_radiation_targets["radiation_cloud_transmission"]),
            "target_radiation_daylight_valid": float(target_radiation_targets["radiation_daylight_valid"]),
            "source_notes": {
                "mask": "Sentinel-2 SCL/CLP cloud-only mask; non-cloud pixels are black.",
                "cloud_features": "Derived from the Sentinel-2 cloud-only tensor, not Open-Meteo cloud cover.",
                "world_features": "Open-Meteo hourly non-temperature, non-cloud-cover context.",
                "target": f"Open-Meteo temperature_2m at anchor + {args.target_offset_days} days.",
            },
        }

    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.s2_workers)) as ex:
        futures = [ex.submit(process_scene, pair) for pair in all_pairs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="S2 cloud samples"):
            row = fut.result()
            if row is not None:
                records.append(row)

    records.sort(key=lambda r: (r["date"], r["location"], r["sample_id"]))
    if not records:
        raise SystemExit("No samples produced.")

    splits = split_records(records)

    print("[4/4] Writing outputs")
    write_jsonl(out_dir / "dataset.jsonl", records)
    for split, rows in splits.items():
        write_jsonl(out_dir / "splits" / f"{split}.jsonl", rows)

    metadata = {
        "target": TARGET_FIELD,
        "target_description": f"temperature_2m at anchor + {args.target_offset_days} days",
        "radiation_targets": [
            "radiation_cloud_loss_wm2",
            "radiation_cloud_attenuation",
            "radiation_cloud_transmission",
            "radiation_shortwave_observed_wm2",
            "radiation_clear_sky_proxy_wm2",
            "radiation_daylight_valid"
        ],
        "radiation_target_description": "Scene-time shortwave cloud effect derived from Open-Meteo shortwave radiation and a simple solar clear-sky proxy.",
        "raw_feature_names": RAW_FEATURE_NAMES,
        "model_feature_names": RAW_FEATURE_NAMES,
        "cloud_feature_names": CLOUD_FEATURE_NAMES,
        "world_feature_names": WORLD_FEATURE_NAMES,
        "benchmark_inputs": [
            "current_temperature_c is stored outside model inputs for a concrete persistence benchmark.",
            "The model must beat current-temperature persistence without receiving current temperature as an input.",
        ],
        "removed_shortcuts": [
            "future temperature target leakage",
            "current temperature model input",
            "Open-Meteo cloud_cover input",
            "Open-Meteo cloud_cover_low/mid/high input",
            "duplicate direct/diffuse/terrestrial radiation inputs",
            "fake longwave/net-radiation placeholders",
        ],
        "image": {
            "height": size[1],
            "width": size[0],
            "meaning": "single-channel cloud-only image; black=non-cloud, white/probability=cloud",
        },
        "locations": [asdict(loc) for loc in locations],
        "date_range": {"start": args.start_date, "end": args.end_date},
        "target_offset_days": float(args.target_offset_days),
        "splits": {k: len(v) for k, v in splits.items()},
    }
    write_json(out_dir / "metadata.json", metadata)

    if not args.skip_torch:
        export_torch_bundle(out_dir / "dataset.pt", records, splits)

    print(f"Wrote {len(records)} samples to {out_dir}")
    print("Split sizes:", {k: len(v) for k, v in splits.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
