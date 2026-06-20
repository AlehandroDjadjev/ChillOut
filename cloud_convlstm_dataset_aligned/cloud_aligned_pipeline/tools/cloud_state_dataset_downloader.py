#!/usr/bin/env python3
"""
cloud_state_dataset_downloader.py

Full dataset downloader for cloud-state / radiation / Open-Meteo-aligned ConvLSTM training data.

What it does:
  1. Uses Copernicus Data Space Sentinel Hub Catalog API to find Sentinel-2 L2A acquisitions
     for 10 km x 10 km patches around configured coordinates.
  2. Uses Sentinel Hub Process API to download a fixed-size, aligned Sentinel-2 cloud-only tensor
     for each acquisition timestamp. Ground pixels are hard-gated to black/zero using SCL classes.
  3. Uses Open-Meteo Historical Weather API to download nearest-hour weather/radiation
     values for the same locations/time period. This replaces slow CDS/ERA5 downloads.
  4. Builds per-state .npz files that link each Sentinel-2 cloud tensor with nearest-hour
     Open-Meteo numeric values stored under ERA5-compatible keys for the existing converter.
  5. Writes metadata.csv and run_config.json so the dataset is reproducible.

Important hard fact:
  Sentinel-2 does NOT have 30 years of imagery. Sentinel-2 starts in 2015; L2A global availability
  is generally from 2017 onward. For 30-year histories you can use ERA5, but not Sentinel-2 images.

Credentials expected:
  Sentinel Hub / Copernicus Data Space OAuth client credentials:
    export SH_CLIENT_ID="..."
    export SH_CLIENT_SECRET="..."

  CDS API token, either via ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <YOUR_CDS_PERSONAL_ACCESS_TOKEN>

  or environment variables:
    export CDSAPI_URL="https://cds.climate.copernicus.eu/api"
    export CDSAPI_KEY="..."

Install in a venv:
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install --upgrade pip
  pip install sentinelhub cdsapi numpy pandas pyproj python-dateutil tifffile pillow xarray netcdf4 tqdm requests python-dotenv

Run example:
  python cloud_state_dataset_downloader.py \
    --start-date 2022-01-01 \
    --end-date 2022-03-01 \
    --out-dir ./cloud_dataset \
    --max-cloud 100 \
    --resolution-m 100 \
    --patch-km 10 \
    --build-npz \
    --s2-workers 4 \
    --era5-workers 2

For a full Sentinel-2 era run, use e.g. --start-date 2017-01-01 --end-date 2026-01-01.
Expect this to be slow and quota-heavy. It is a real downloader, not a demo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Optional local .env loading.
# Put .env next to this script:
#   SH_CLIENT_ID=...
#   SH_CLIENT_SECRET=...
#   CDSAPI_URL=https://cds.climate.copernicus.eu/api
#   CDSAPI_KEY=...
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

SCRIPT_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(SCRIPT_DIR / ".env")

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import tifffile
from dateutil.parser import isoparse
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from tqdm import tqdm

try:
    import cdsapi
except Exception:  # pragma: no cover
    cdsapi = None

try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None

from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
)


# Use the Copernicus Data Space Ecosystem Sentinel Hub endpoint, not the old
# services.sentinel-hub.com endpoint. Without this, Catalog may work but
# Process API can fail with 401 Unauthorized because it calls the wrong host.
SENTINEL2_L2A_CDSE = DataCollection.SENTINEL2_L2A.define_from(
    name="sentinel-2-l2a-cdse",
    service_url="https://sh.dataspace.copernicus.eu",
)

# -----------------------------
# Default 10 locations
# -----------------------------
# Picked to include different cloud/temperature regimes: coast, continental, dry, humid, high latitude.
DEFAULT_LOCATIONS = [
    {"name": "sofia_bg", "lat": 42.6977, "lon": 23.3219},
    {"name": "athens_gr", "lat": 37.9838, "lon": 23.7275},
    {"name": "london_uk", "lat": 51.5074, "lon": -0.1278},
    {"name": "oslo_no", "lat": 59.9139, "lon": 10.7522},
    {"name": "cairo_eg", "lat": 30.0444, "lon": 31.2357},
    {"name": "reykjavik_is", "lat": 64.1466, "lon": -21.9426},
    {"name": "lisbon_pt", "lat": 38.7223, "lon": -9.1393},
    {"name": "milan_it", "lat": 45.4642, "lon": 9.1900},
    {"name": "casablanca_ma", "lat": 33.5731, "lon": -7.5898},
    {"name": "helsinki_fi", "lat": 60.1699, "lon": 24.9384},
]


# -----------------------------
# ERA5 variables
# -----------------------------
# Split ERA5 single-level variables by NetCDF stepType class.
# The new CDS GRIB->NetCDF converter splits output by stepType (instant vs accum).
# Requesting instant + accumulated variables together as one NetCDF is brittle and can fail.
ERA5_SINGLE_INSTANT_VARIABLES = [
    # Cloud fields - instantaneous/state-like
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "total_column_cloud_liquid_water",
    "total_column_cloud_ice_water",

    # Dynamic atmosphere / rain support - instantaneous/state-like
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_column_water_vapour",
    "surface_pressure",
    "mean_sea_level_pressure",
    "boundary_layer_height",
    "convective_available_potential_energy",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",

    # Optional lightweight surface state/context.
    # Do NOT use "snow_cover" here. In CDS/MARS it is ambiguous and fails:
    # "snow_cover could be SNOW EVAPORATION or SNOW EVAPORATION VARIABLE RESOLUTION".
    # Use "snow_depth" if you want a snow-state proxy.
    "skin_temperature",
    "sea_surface_temperature",
    "snow_depth",
    "soil_temperature_level_1",
    "volumetric_soil_water_layer_1",
    "forecast_albedo",
    "land_sea_mask",
    "geopotential",
]

ERA5_SINGLE_ACCUM_VARIABLES = [
    # Radiation / accumulated flux fields
    "surface_solar_radiation_downwards",
    "surface_solar_radiation_downward_clear_sky",
    "surface_thermal_radiation_downwards",
    "surface_thermal_radiation_downward_clear_sky",
    "surface_net_solar_radiation",
    "surface_net_solar_radiation_clear_sky",
    "surface_net_thermal_radiation",
    "surface_net_thermal_radiation_clear_sky",
    "toa_incident_solar_radiation",
    "top_net_solar_radiation",
    "top_net_thermal_radiation",
    "total_precipitation",
]

ERA5_SINGLE_LEVEL_VARIABLES = ERA5_SINGLE_INSTANT_VARIABLES + ERA5_SINGLE_ACCUM_VARIABLES

ERA5_PRESSURE_LEVEL_VARIABLES = [
    "relative_humidity",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "fraction_of_cloud_cover",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
]

DEFAULT_PRESSURE_LEVELS = ["1000", "925", "850", "700", "500", "300"]


# -----------------------------
# Sentinel-2 cloud tensor evalscript
# -----------------------------
# Output bands, FLOAT32, all non-cloud / invalid ground pixels are zeroed:
# 0 cloud_mask_scl       : 1 for SCL 8/9/10, else 0
# 1 cloud_gray           : visible grayscale only on cloud pixels
# 2 cloud_red_B04        : gated red
# 3 cloud_green_B03      : gated green
# 4 cloud_blue_B02       : gated blue
# 5 cloud_nir_B08        : gated NIR
# 6 cloud_prob_CLP       : gated cloud probability, normalized 0..1 where available
# 7 aerosol_AOT          : gated aerosol optical thickness proxy from S2 L2A
# 8 water_vapour_WVP     : placeholder 0.0; use ERA5 total_column_water_vapour instead
# 9 cirrus_flag          : 1 for SCL 10 thin cirrus, else 0
# 10 high_cloud_flag     : 1 for SCL 9, else 0
# 11 medium_cloud_flag   : 1 for SCL 8, else 0
S2_CLOUD_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      // Do NOT force all bands to REFLECTANCE.
      // B02/B03/B04/B08 are reflectance-like by default, but SCL and CLP are DN/classification bands.
      // Forcing SCL to REFLECTANCE causes:
      // "Band 'SCL' requested in unsupported units 'REFLECTANCE'".
      bands: ["B02", "B03", "B04", "B08", "SCL", "CLP", "AOT", "dataMask"]
    }],
    output: {
      bands: 12,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(s) {
  // ESA Sentinel-2 L2A Scene Classification Layer:
  // 8 = cloud medium probability, 9 = cloud high probability, 10 = thin cirrus.
  let mediumCloud = (s.SCL === 8) ? 1.0 : 0.0;
  let highCloud = (s.SCL === 9) ? 1.0 : 0.0;
  let cirrus = (s.SCL === 10) ? 1.0 : 0.0;
  let isCloud = ((mediumCloud + highCloud + cirrus) > 0.0 && s.dataMask > 0) ? 1.0 : 0.0;

  let r = isCloud * s.B04;
  let g = isCloud * s.B03;
  let b = isCloud * s.B02;
  let nir = isCloud * s.B08;
  let gray = isCloud * ((s.B02 + s.B03 + s.B04) / 3.0);

  // CLP is typically 0..255 in Sentinel Hub; normalize defensively.
  let clp = isCloud * Math.max(0.0, Math.min(1.0, s.CLP / 255.0));

  return [
    isCloud,
    gray,
    r,
    g,
    b,
    nir,
    clp,
    isCloud * s.AOT,
    0.0, // water_vapour_WVP removed: S2L2A CDSE collection has no WVP band here; use ERA5 total_column_water_vapour instead.
    cirrus,
    highCloud,
    mediumCloud
  ];
}
"""


@dataclass
class Location:
    name: str
    lat: float
    lon: float


@dataclass
class SentinelScene:
    location: str
    item_id: str
    timestamp: str
    cloud_cover: Optional[float]
    bbox_wgs84: Tuple[float, float, float, float]


# -----------------------------
# Utility functions
# -----------------------------
def safe_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


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


def month_key(dt: datetime) -> Tuple[int, int]:
    return dt.year, dt.month


def daterange_months(start: datetime, end: datetime) -> Iterable[Tuple[int, int]]:
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    end_m = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    while cur <= end_m:
        yield cur.year, cur.month
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def bbox_around_point_wgs84(lon: float, lat: float, patch_km: float) -> Tuple[float, float, float, float]:
    """Return WGS84 bbox around lon/lat with square side patch_km in local UTM meters."""
    epsg = utm_epsg_for_lonlat(lon, lat)
    wgs84 = PyprojCRS.from_epsg(4326)
    utm = PyprojCRS.from_epsg(epsg)
    to_utm = Transformer.from_crs(wgs84, utm, always_xy=True)
    to_wgs = Transformer.from_crs(utm, wgs84, always_xy=True)
    x, y = to_utm.transform(lon, lat)
    half_m = patch_km * 1000.0 / 2.0
    minx, miny, maxx, maxy = x - half_m, y - half_m, x + half_m, y + half_m
    corners = [to_wgs.transform(px, py) for px, py in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return min(lons), min(lats), max(lons), max(lats)


def image_size_from_patch(patch_km: float, resolution_m: float) -> Tuple[int, int]:
    n = max(8, int(round((patch_km * 1000.0) / resolution_m)))
    return n, n


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_locations(path: Optional[str]) -> List[Location]:
    if not path:
        return [Location(name=safe_name(x["name"]), lat=float(x["lat"]), lon=float(x["lon"])) for x in DEFAULT_LOCATIONS]
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Locations file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    locations = []
    for item in data:
        locations.append(Location(name=safe_name(item["name"]), lat=float(item["lat"]), lon=float(item["lon"])))
    return locations


# -----------------------------
# Sentinel Hub setup
# -----------------------------
def make_sh_config() -> SHConfig:
    client_id = os.environ.get("SH_CLIENT_ID")
    client_secret = os.environ.get("SH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Sentinel Hub credentials. Set SH_CLIENT_ID and SH_CLIENT_SECRET. "
            "Create them in the Copernicus Data Space / Sentinel Hub Dashboard as an OAuth client."
        )
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    # Copernicus Data Space Ecosystem OAuth + Sentinel Hub Process/Catalog endpoint.
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
    limit: Optional[int] = None,
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
        if not ts:
            continue
        item_id = item.get("id", "")
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
    out_path: Path,
    retries: int = 3,
    sleep_s: float = 2.0,
) -> bool:
    """Download one Sentinel-2 cloud-only tensor as TIFF-like multi-band array saved with tifffile."""
    if out_path.exists():
        return True

    ts = parse_date(scene.timestamp)
    # Tight one-hour interval around the acquisition. Sentinel Hub will select available data in the interval.
    start = (ts - timedelta(minutes=30)).isoformat()
    end = (ts + timedelta(minutes=30)).isoformat()
    bbox = BBox(scene.bbox_wgs84, crs=CRS.WGS84)

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
        bbox=bbox,
        size=size,
        config=config,
    )

    for attempt in range(1, retries + 1):
        try:
            data = request.get_data()[0]
            arr = np.asarray(data, dtype=np.float32)
            # SentinelHub sometimes returns [H,W,C], ensure it.
            if arr.ndim != 3 or arr.shape[-1] != 12:
                raise RuntimeError(f"Unexpected Sentinel tensor shape {arr.shape}")
            ensure_dir(out_path.parent)
            tifffile.imwrite(str(out_path), arr, dtype=np.float32)
            return True
        except Exception as e:
            if attempt == retries:
                print(f"[S2 ERROR] {scene.location} {scene.timestamp} {scene.item_id}: {e}", file=sys.stderr)
                return False
            time.sleep(sleep_s * attempt)
    return False



# -----------------------------
# Open-Meteo historical weather replacement for ERA5/CDS
# -----------------------------
# Open-Meteo Historical Weather API is much faster than CDS for this use-case.
# It returns point/time-series weather values, not gridded ERA5 maps.
# We store those scalar values under the same era5_single_* keys expected by
# the existing tensor converter. The converter will broadcast scalars to maps.
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

OPEN_METEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "terrestrial_radiation",
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "terrestrial_radiation_instant",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
    "weather_code",
]


def fetch_openmeteo_hourly(
    loc: Location,
    start_date: str,
    end_date: str,
    target: Path,
    model: str = "best_match",
    retries: int = 4,
    timeout_s: int = 120,
) -> bool:
    """Fetch Open-Meteo hourly historical weather JSON for one location/date range.

    model:
      - "best_match" or "" => omit model parameter; Open-Meteo picks the highest/best source.
      - "era5", "era5_land", etc. can be passed through if you want consistency over speed/resolution.
    """
    if target.exists():
        return True

    ensure_dir(target.parent)

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

    # Best match is the default behavior. For 2017+ this normally gives the best available
    # global historical model, currently ECMWF IFS 9 km where available.
    if model and model.lower() not in {"best_match", "default", "auto"}:
        params["models"] = model

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout_s)
            if r.status_code >= 400:
                raise RuntimeError(f"Open-Meteo HTTP {r.status_code}: {r.text[:1000]}")
            payload = r.json()
            if payload.get("error"):
                raise RuntimeError(f"Open-Meteo error: {payload}")
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return True
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(min(60, 2 ** attempt))
                continue

    print(
        f"[OPEN-METEO ERROR] {loc.name} {start_date}..{end_date} -> {target}: {last_error}",
        file=sys.stderr,
    )
    return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        x = float(value)
        if not math.isfinite(x):
            return float(default)
        return x
    except Exception:
        return float(default)


def _openmeteo_hour_row(payload: Dict[str, Any], timestamp: datetime) -> Dict[str, float]:
    """Return hourly values closest to the nearest-hour timestamp."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {}

    key = nearest_hour(timestamp).strftime("%Y-%m-%dT%H:00")

    try:
        idx = times.index(key)
    except ValueError:
        # Fallback: build parsed nearest index. This is slower but robust.
        target = nearest_hour(timestamp).replace(tzinfo=None)
        best_idx = 0
        best_delta = None
        for i, raw in enumerate(times):
            try:
                t = datetime.fromisoformat(str(raw))
            except Exception:
                continue
            delta = abs((t - target).total_seconds())
            if best_delta is None or delta < best_delta:
                best_idx, best_delta = i, delta
        idx = best_idx

    row: Dict[str, float] = {}
    for name, values in hourly.items():
        if name == "time":
            continue
        if isinstance(values, list) and idx < len(values):
            row[name] = _safe_float(values[idx], 0.0)
    return row


def _wind_uv_from_speed_dir(speed_ms: float, direction_deg: float) -> Tuple[float, float]:
    """Convert meteorological wind direction/speed to u/v components.

    Direction is where wind comes FROM. u positive eastward, v positive northward.
    """
    rad = math.radians(direction_deg)
    u = -speed_ms * math.sin(rad)
    v = -speed_ms * math.cos(rad)
    return float(u), float(v)


def _scalar(value: float) -> np.ndarray:
    return np.array(float(value), dtype=np.float32)


def openmeteo_to_era5_compatible_npz_payload(payload: Dict[str, Any], timestamp: datetime) -> Dict[str, np.ndarray]:
    """Map Open-Meteo hourly point values to the era5_single_* keys expected downstream.

    This is intentionally scalar. The tensor builder already broadcasts scalar/0D arrays to HxW maps.
    """
    row = _openmeteo_hour_row(payload, timestamp)

    temp_c = _safe_float(row.get("temperature_2m"))
    dew_c = _safe_float(row.get("dew_point_2m"))
    rh_frac = _safe_float(row.get("relative_humidity_2m")) / 100.0

    cloud_total = _safe_float(row.get("cloud_cover")) / 100.0
    cloud_low = _safe_float(row.get("cloud_cover_low")) / 100.0
    cloud_mid = _safe_float(row.get("cloud_cover_mid")) / 100.0
    cloud_high = _safe_float(row.get("cloud_cover_high")) / 100.0

    pressure_msl_pa = _safe_float(row.get("pressure_msl")) * 100.0
    surface_pressure_pa = _safe_float(row.get("surface_pressure")) * 100.0

    wind_speed = _safe_float(row.get("wind_speed_10m"))
    wind_dir = _safe_float(row.get("wind_direction_10m"))
    wind_u, wind_v = _wind_uv_from_speed_dir(wind_speed, wind_dir)

    precip_mm = _safe_float(row.get("precipitation"))
    snow_depth_m = _safe_float(row.get("snow_depth"))

    # Open-Meteo solar radiation is W/m^2. The ERA5 converter just normalizes channels,
    # so W/m^2 is fine. For clear-sky shortwave, Open-Meteo does not provide a true
    # clear-sky all-sky pair. Terrestrial radiation is used as a top-of-atmosphere proxy.
    sw = _safe_float(row.get("shortwave_radiation"))
    sw_inst = _safe_float(row.get("shortwave_radiation_instant"), sw)
    toa_sw = _safe_float(row.get("terrestrial_radiation"), _safe_float(row.get("terrestrial_radiation_instant"), sw_inst))

    # Open-Meteo Historical Weather does not provide downwelling longwave radiation.
    # Keep thermal fields zero so the target longwave anomaly is neutral rather than bogus.
    lw = 0.0
    lw_clear = 0.0

    # Open-Meteo does not provide cloud liquid/ice water. Keep zeros, but keep keys present.
    clw = 0.0
    ciw = 0.0

    # Total-column water vapour is not available in the stable hourly list; use RH fraction
    # as a weak moisture proxy so the channel is not dead.
    tcwv_proxy = rh_frac

    # Net solar is approximated by surface shortwave. Net thermal unavailable.
    net_solar = sw
    net_solar_clear_proxy = toa_sw
    net_thermal = 0.0
    net_thermal_clear = 0.0

    out = {
        "era5_single_total_cloud_cover": _scalar(cloud_total),
        "era5_single_low_cloud_cover": _scalar(cloud_low),
        "era5_single_medium_cloud_cover": _scalar(cloud_mid),
        "era5_single_high_cloud_cover": _scalar(cloud_high),
        "era5_single_total_column_cloud_liquid_water": _scalar(clw),
        "era5_single_total_column_cloud_ice_water": _scalar(ciw),

        "era5_single_2m_temperature": _scalar(temp_c),
        "era5_single_2m_dewpoint_temperature": _scalar(dew_c),
        "era5_single_total_column_water_vapour": _scalar(tcwv_proxy),
        "era5_single_surface_pressure": _scalar(surface_pressure_pa),
        "era5_single_mean_sea_level_pressure": _scalar(pressure_msl_pa),

        "era5_single_10m_u_component_of_wind": _scalar(wind_u),
        "era5_single_10m_v_component_of_wind": _scalar(wind_v),
        "era5_single_total_precipitation": _scalar(precip_mm),
        "era5_single_snow_depth": _scalar(snow_depth_m),

        "era5_single_surface_solar_radiation_downwards": _scalar(sw),
        "era5_single_surface_solar_radiation_downwards_clear_sky": _scalar(toa_sw),
        "era5_single_surface_solar_radiation_downward_clear_sky": _scalar(toa_sw),

        "era5_single_surface_thermal_radiation_downwards": _scalar(lw),
        "era5_single_surface_thermal_radiation_downwards_clear_sky": _scalar(lw_clear),
        "era5_single_surface_thermal_radiation_downward_clear_sky": _scalar(lw_clear),

        "era5_single_surface_net_solar_radiation": _scalar(net_solar),
        "era5_single_surface_net_solar_radiation_clear_sky": _scalar(net_solar_clear_proxy),
        "era5_single_surface_net_thermal_radiation": _scalar(net_thermal),
        "era5_single_surface_net_thermal_radiation_clear_sky": _scalar(net_thermal_clear),

        "era5_single_toa_incident_solar_radiation": _scalar(toa_sw),

        # Misc/context fallbacks expected by some configs.
        "era5_single_skin_temperature": _scalar(temp_c),
        "era5_single_sea_surface_temperature": _scalar(temp_c),
        "era5_single_soil_temperature_level_1": _scalar(_safe_float(row.get("soil_temperature_0_to_7cm"), temp_c)),
        "era5_single_volumetric_soil_water_layer_1": _scalar(_safe_float(row.get("soil_moisture_0_to_7cm"))),
        "era5_single_boundary_layer_height": _scalar(0.0),
        "era5_single_convective_available_potential_energy": _scalar(0.0),
        "era5_single_forecast_albedo": _scalar(0.2),
        "era5_single_land_sea_mask": _scalar(1.0),
        "era5_single_geopotential": _scalar(0.0),
    }
    return out


# -----------------------------
# CDS / ERA5 setup
# -----------------------------
def make_cds_client() -> Any:
    if cdsapi is None:
        raise RuntimeError("cdsapi is not installed. pip install cdsapi")
    url = os.environ.get("CDSAPI_URL")
    key = os.environ.get("CDSAPI_KEY")
    if url and key:
        return cdsapi.Client(url=url, key=key)
    return cdsapi.Client()


def build_cds_area(bbox_wgs84: Tuple[float, float, float, float], buffer_deg: float = 0.1) -> List[float]:
    minlon, minlat, maxlon, maxlat = bbox_wgs84
    # CDS area format: [North, West, South, East]
    return [maxlat + buffer_deg, minlon - buffer_deg, minlat - buffer_deg, maxlon + buffer_deg]


def group_hours_by_month(timestamps: Sequence[datetime]) -> Dict[Tuple[int, int], Dict[str, List[str]]]:
    grouped: Dict[Tuple[int, int], Dict[str, set[str]]] = {}
    for dt in timestamps:
        h = nearest_hour(dt)
        key = (h.year, h.month)
        if key not in grouped:
            grouped[key] = {"days": set(), "times": set()}
        grouped[key]["days"].add(f"{h.day:02d}")
        grouped[key]["times"].add(f"{h.hour:02d}:00")
    return {k: {"days": sorted(v["days"]), "times": sorted(v["times"])} for k, v in grouped.items()}


def chunked_sequence(values: Sequence[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0 or chunk_size >= len(values):
        return [list(values)]
    return [list(values[i:i + chunk_size]) for i in range(0, len(values), chunk_size)]


def retrieve_era5_single_month(
    client: Any,
    year: int,
    month: int,
    days: List[str],
    times: List[str],
    area: List[float],
    target: Path,
    variables: Sequence[str] = ERA5_SINGLE_LEVEL_VARIABLES,
    retries: int = 3,
) -> bool:
    if target.exists():
        return True
    ensure_dir(target.parent)
    request = {
        "product_type": ["reanalysis"],
        "variable": list(variables),
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": times,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    for attempt in range(1, retries + 1):
        try:
            client.retrieve("reanalysis-era5-single-levels", request, str(target))
            return True
        except Exception as e:
            if attempt == retries:
                print(f"[ERA5 SINGLE ERROR] {year}-{month:02d} {target}: {e}\n  variables={list(variables)}\n  days={days}\n  times={times}\n  area={area}", file=sys.stderr)
                return False
            time.sleep(5 * attempt)
    return False


def retrieve_era5_pressure_month(
    client: Any,
    year: int,
    month: int,
    days: List[str],
    times: List[str],
    area: List[float],
    target: Path,
    variables: Sequence[str] = ERA5_PRESSURE_LEVEL_VARIABLES,
    pressure_levels: Sequence[str] = DEFAULT_PRESSURE_LEVELS,
    retries: int = 3,
) -> bool:
    if target.exists():
        return True
    ensure_dir(target.parent)
    request = {
        "product_type": ["reanalysis"],
        "variable": list(variables),
        "pressure_level": list(pressure_levels),
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": times,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    for attempt in range(1, retries + 1):
        try:
            client.retrieve("reanalysis-era5-pressure-levels", request, str(target))
            return True
        except Exception as e:
            if attempt == retries:
                print(f"[ERA5 PRESSURE ERROR] {year}-{month:02d} {target}: {e}", file=sys.stderr)
                return False
            time.sleep(5 * attempt)
    return False


def find_time_coord(ds: Any) -> str:
    for name in ["valid_time", "time", "forecast_reference_time"]:
        if name in ds.coords or name in ds.dims:
            return name
    # fallback: first datetime-like coord
    for name, coord in ds.coords.items():
        if np.issubdtype(coord.dtype, np.datetime64):
            return name
    raise RuntimeError("Could not find time coordinate in ERA5 dataset")


def crop_ds_to_time(ds: Any, timestamp: datetime) -> Any:
    tcoord = find_time_coord(ds)
    target_np = np.datetime64(nearest_hour(timestamp).replace(tzinfo=None))
    return ds.sel({tcoord: target_np}, method="nearest")


def ds_to_npz_payload(ds: Any, prefix: str) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {}
    for var_name in ds.data_vars:
        arr = ds[var_name].values
        # Remove scalar time dims already selected; keep pressure/lat/lon if present.
        arr = np.asarray(arr)
        payload[f"{prefix}_{var_name}"] = arr.astype(np.float32, copy=False) if np.issubdtype(arr.dtype, np.number) else arr
    return payload


def build_state_npz_files(metadata_csv: Path, out_dir: Path, overwrite: bool = False) -> None:
    df = pd.read_csv(metadata_csv)
    states_dir = out_dir / "states_npz"
    ensure_dir(states_dir)

    # cache data files
    openmeteo_cache: Dict[str, Dict[str, Any]] = {}
    single_cache: Dict[str, Any] = {}
    pressure_cache: Dict[str, Any] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building per-state NPZ"):
        if not bool(row.get("s2_download_ok", False)):
            continue
        state_id = str(row["state_id"])
        npz_path = states_dir / f"{state_id}.npz"
        if npz_path.exists() and not overwrite:
            continue
        ts = parse_date(str(row["timestamp_utc"]))
        payload: Dict[str, Any] = {}
        payload["state_id"] = np.array(state_id)
        payload["timestamp_utc"] = np.array(str(row["timestamp_utc"]))
        payload["location"] = np.array(str(row["location"]))

        s2_path = Path(str(row["s2_cloud_tif"]))
        if s2_path.exists():
            payload["s2_cloud_tensor"] = tifffile.imread(str(s2_path)).astype(np.float32)

        # Default path: Open-Meteo hourly weather/radiation values mapped into era5_single_* keys.
        openmeteo_path = str(row.get("openmeteo_json", ""))
        if openmeteo_path and Path(openmeteo_path).exists():
            if openmeteo_path not in openmeteo_cache:
                openmeteo_cache[openmeteo_path] = json.loads(Path(openmeteo_path).read_text(encoding="utf-8"))
            payload.update(openmeteo_to_era5_compatible_npz_payload(openmeteo_cache[openmeteo_path], ts))

        # Backward-compatible fallback: if old ERA5 NetCDFs exist, merge them too.
        for single_col in ["era5_single_instant_nc", "era5_single_accum_nc", "era5_single_nc"]:
            single_path = str(row.get(single_col, ""))
            if single_path and Path(single_path).exists():
                if xr is None:
                    raise RuntimeError("xarray is required to read legacy ERA5 NetCDF files. pip install xarray netcdf4")
                if single_path not in single_cache:
                    single_cache[single_path] = xr.open_dataset(single_path)
                ds_single = crop_ds_to_time(single_cache[single_path], ts)
                payload.update(ds_to_npz_payload(ds_single, "era5_single"))

        pressure_path = str(row.get("era5_pressure_nc", ""))
        if pressure_path and Path(pressure_path).exists():
            if xr is None:
                raise RuntimeError("xarray is required to read legacy ERA5 pressure NetCDF files. pip install xarray netcdf4")
            if pressure_path not in pressure_cache:
                pressure_cache[pressure_path] = xr.open_dataset(pressure_path)
            ds_pressure = crop_ds_to_time(pressure_cache[pressure_path], ts)
            payload.update(ds_to_npz_payload(ds_pressure, "era5_pressure"))

        np.savez_compressed(npz_path, **payload)

    for ds in list(single_cache.values()) + list(pressure_cache.values()):
        try:
            ds.close()
        except Exception:
            pass


# -----------------------------
# Main orchestration
# -----------------------------
def write_metadata_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    start_dt = parse_date(args.start_date)
    end_dt = parse_date(args.end_date)
    if end_dt <= start_dt:
        raise ValueError("--end-date must be after --start-date")

    if start_dt.year < 2015:
        print(
            "[WARN] Sentinel-2 imagery does not exist before 2015. ERA5 can go earlier, but S2 cannot. "
            "The Sentinel query will return no products before availability.",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    ensure_dir(out_dir / "sentinel2_cloud_tensors")
    ensure_dir(out_dir / "era5_single")
    ensure_dir(out_dir / "openmeteo_hourly")
    ensure_dir(out_dir / "era5_pressure")
    ensure_dir(out_dir / "metadata")

    locations = read_locations(args.locations_json)
    size = image_size_from_patch(args.patch_km, args.resolution_m)

    run_config = {
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "patch_km": args.patch_km,
        "resolution_m": args.resolution_m,
        "image_size": size,
        "max_cloud": args.max_cloud,
        "locations": [asdict(loc) for loc in locations],
        "era5_single_variables": ERA5_SINGLE_LEVEL_VARIABLES,
        "era5_single_instant_variables": ERA5_SINGLE_INSTANT_VARIABLES,
        "era5_single_accum_variables": ERA5_SINGLE_ACCUM_VARIABLES,
        "weather_source": "open-meteo-historical",
        "openmeteo_model": args.openmeteo_model,
        "openmeteo_hourly_variables": OPEN_METEO_HOURLY_VARIABLES,
        "era5_pressure_variables": ERA5_PRESSURE_LEVEL_VARIABLES if args.include_pressure_levels else [],
        "pressure_levels": DEFAULT_PRESSURE_LEVELS if args.include_pressure_levels else [],
        "sentinel_cloud_tensor_bands": [
            "cloud_mask_scl",
            "cloud_gray",
            "cloud_red_B04",
            "cloud_green_B03",
            "cloud_blue_B02",
            "cloud_nir_B08",
            "cloud_prob_CLP",
            "aerosol_AOT",
            "water_vapour_WVP_placeholder_zero",
            "cirrus_flag",
            "high_cloud_flag",
            "medium_cloud_flag",
        ],
    }
    write_json(out_dir / "metadata" / "run_config.json", run_config)

    print("[1/5] Configuring Sentinel Hub...")
    sh_config = make_sh_config()

    all_scenes: List[SentinelScene] = []
    loc_bboxes: Dict[str, Tuple[float, float, float, float]] = {}

    print("[2/5] Searching Sentinel-2 L2A acquisitions...")
    for loc in locations:
        bbox = bbox_around_point_wgs84(loc.lon, loc.lat, args.patch_km)
        loc_bboxes[loc.name] = bbox
        try:
            scenes = catalog_search_s2(
                sh_config,
                loc,
                bbox,
                start_dt,
                end_dt,
                args.max_cloud,
                limit=args.max_scenes_per_location,
            )
            all_scenes.extend(scenes)
            print(f"  {loc.name}: {len(scenes)} Sentinel-2 scenes")
        except Exception as e:
            print(f"[CATALOG ERROR] {loc.name}: {e}", file=sys.stderr)

    scenes_json = [asdict(s) for s in all_scenes]
    write_json(out_dir / "metadata" / "sentinel_scenes.json", scenes_json)

    print("[3/5] Downloading Sentinel-2 cloud-only tensors...")
    rows: List[Dict[str, Any]] = []

    def prepare_scene_row(scene: SentinelScene) -> Dict[str, Any]:
        ts = parse_date(scene.timestamp)
        state_id = f"{safe_name(scene.location)}_{ts.strftime('%Y%m%dT%H%M%SZ')}_{safe_name(scene.item_id)[:32]}"
        s2_path = out_dir / "sentinel2_cloud_tensors" / scene.location / f"{state_id}.tif"
        bbox = loc_bboxes[scene.location]
        near = nearest_hour(ts)
        y, m = near.year, near.month
        openmeteo_path = out_dir / "openmeteo_hourly" / scene.location / f"openmeteo_{scene.location}_{y}_{m:02d}.json"
        era5_single_instant_path = out_dir / "era5_single" / scene.location / f"era5_single_instant_{scene.location}_{y}_{m:02d}.nc"
        era5_single_accum_path = out_dir / "era5_single" / scene.location / f"era5_single_accum_{scene.location}_{y}_{m:02d}.nc"
        era5_pressure_path = out_dir / "era5_pressure" / scene.location / f"era5_pressure_{scene.location}_{y}_{m:02d}.nc"
        return {
            "state_id": state_id,
            "location": scene.location,
            "timestamp_utc": ts.isoformat(),
            "nearest_era5_hour_utc": near.isoformat(),
            "sentinel_item_id": scene.item_id,
            "sentinel_scene_cloud_cover": scene.cloud_cover,
            "bbox_minlon": bbox[0],
            "bbox_minlat": bbox[1],
            "bbox_maxlon": bbox[2],
            "bbox_maxlat": bbox[3],
            "s2_cloud_tif": str(s2_path),
            "s2_download_ok": False,
            # Open-Meteo replacement path. This is the default weather/radiation source now.
            "openmeteo_json": str(openmeteo_path),

            # Backward-compatible ERA5 columns. Kept empty unless you deliberately use old CDS mode later.
            "era5_single_instant_nc": "",
            "era5_single_accum_nc": "",
            "era5_single_nc": "",
            "era5_pressure_nc": "",
        }

    rows = [prepare_scene_row(scene) for scene in all_scenes]

    if args.skip_sentinel_download:
        for row in rows:
            row["s2_download_ok"] = Path(str(row["s2_cloud_tif"])).exists()
    else:
        def download_one(pair: Tuple[SentinelScene, Dict[str, Any]]) -> Tuple[str, bool]:
            scene, row = pair
            ok = download_s2_cloud_tensor(
                sh_config,
                scene,
                size=size,
                out_path=Path(str(row["s2_cloud_tif"])),
            )
            return row["state_id"], ok

        scene_by_id = {row["state_id"]: row for row in rows}
        pairs = list(zip(all_scenes, rows))

        workers = max(1, int(args.s2_workers))
        if workers == 1:
            for scene, row in tqdm(pairs, desc="S2 tensors"):
                ok = download_s2_cloud_tensor(
                    sh_config,
                    scene,
                    size=size,
                    out_path=Path(str(row["s2_cloud_tif"])),
                )
                row["s2_download_ok"] = ok
        else:
            print(f"  Using {workers} parallel Sentinel-2 workers")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(download_one, pair) for pair in pairs]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="S2 tensors"):
                    state_id, ok = fut.result()
                    scene_by_id[state_id]["s2_download_ok"] = ok

    metadata_csv = out_dir / "metadata" / "states_metadata.csv"
    write_metadata_csv(metadata_csv, rows)

    if args.skip_openmeteo or args.skip_era5:
        print("[4/5] Skipping Open-Meteo weather/radiation downloads.")
    else:
        print("[4/5] Downloading Open-Meteo historical hourly weather/radiation...")
        rows_df = pd.DataFrame(rows)

        om_tasks: List[Dict[str, Any]] = []
        for loc in locations:
            loc_rows = rows_df[rows_df["location"] == loc.name]
            if loc_rows.empty:
                continue

            timestamps = [parse_date(x) for x in loc_rows["timestamp_utc"].tolist()]
            grouped = group_hours_by_month(timestamps)

            for (year, month), dt_info in grouped.items():
                days_int = sorted(int(d) for d in dt_info["days"])
                start_date = f"{year}-{month:02d}-{days_int[0]:02d}"
                end_date = f"{year}-{month:02d}-{days_int[-1]:02d}"
                target = out_dir / "openmeteo_hourly" / loc.name / f"openmeteo_{loc.name}_{year}_{month:02d}.json"
                om_tasks.append({
                    "location": loc,
                    "year": year,
                    "month": month,
                    "start_date": start_date,
                    "end_date": end_date,
                    "target": target,
                })

        def run_openmeteo_task(task: Dict[str, Any]) -> Tuple[str, bool]:
            loc = task["location"]
            label = f"{loc.name} {task['year']}-{task['month']:02d}"
            if Path(task["target"]).exists():
                return label, True
            print(f"  Open-Meteo {label}: {task['start_date']}..{task['end_date']}", flush=True)
            ok = fetch_openmeteo_hourly(
                loc,
                task["start_date"],
                task["end_date"],
                Path(task["target"]),
                model=args.openmeteo_model,
            )
            return label, ok

        workers = max(1, int(args.openmeteo_workers))
        print(f"  Open-Meteo tasks: {len(om_tasks)}")
        print(f"  Open-Meteo workers: {workers}")
        if workers == 1:
            for task in tqdm(om_tasks, desc="Open-Meteo tasks"):
                label, ok = run_openmeteo_task(task)
                if not ok:
                    print(f"[OPEN-METEO TASK FAILED] {label}", file=sys.stderr)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(run_openmeteo_task, task) for task in om_tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Open-Meteo tasks"):
                    label, ok = fut.result()
                    if not ok:
                        print(f"[OPEN-METEO TASK FAILED] {label}", file=sys.stderr)

    if args.build_npz:
        print("[5/5] Building per-state NPZ bundles...")
        build_state_npz_files(metadata_csv, out_dir, overwrite=args.overwrite_npz)
    else:
        print("[5/5] Skipping per-state NPZ build. Metadata still links S2 and ERA5 files.")

    print("DONE")
    print(f"Output folder: {out_dir}")
    print(f"Metadata: {metadata_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download aligned Sentinel-2 cloud tensors and Open-Meteo historical weather/radiation fields for cloud-state ConvLSTM datasets.")
    p.add_argument("--start-date", required=True, help="UTC start date, e.g. 2017-01-01")
    p.add_argument("--end-date", required=True, help="UTC end date, e.g. 2026-01-01")
    p.add_argument("--out-dir", default="./cloud_state_dataset", help="Output dataset folder")
    p.add_argument("--locations-json", default=None, help="Optional JSON list of {name,lat,lon}; defaults to 10 built-in locations")
    p.add_argument("--patch-km", type=float, default=10.0, help="Square patch side length in km around each coordinate")
    p.add_argument("--resolution-m", type=float, default=100.0, help="Sentinel output pixel size in meters. 100m gives 100x100 for 10 km.")
    p.add_argument("--max-cloud", type=float, default=100.0, help="Catalog max scene cloud cover percent. Use 100 to keep cloudy scenes.")
    p.add_argument("--max-scenes-per-location", type=int, default=None, help="Optional cap per location for testing/quota control")
    p.add_argument("--era5-buffer-deg", type=float, default=0.15, help="ERA5 area buffer in degrees around 10 km patch")
    p.add_argument("--era5-variable-chunk-size", type=int, default=0, help="If >0, split ERA5 single-level variables into chunks to isolate CDS failures.")
    p.add_argument("--era5-workers", type=int, default=1, help="Legacy/no-op for Open-Meteo mode. Kept for backwards compatibility.")
    p.add_argument("--openmeteo-model", default="best_match", help="Open-Meteo historical model. Default best_match uses best/highest available source; try era5 or era5_land for consistency.")
    p.add_argument("--openmeteo-workers", type=int, default=4, help="Parallel Open-Meteo fetch workers. Usually 4-8 is fine.")
    p.add_argument("--skip-openmeteo", action="store_true", help="Skip Open-Meteo weather/radiation downloads")
    p.add_argument("--include-pressure-levels", action="store_true", help="Also fetch ERA5 pressure-level fields: humidity, winds, vertical velocity, etc.")
    p.add_argument("--build-npz", action="store_true", help="Build per-state .npz bundles containing S2 tensor + nearest-hour ERA5 arrays")
    p.add_argument("--overwrite-npz", action="store_true", help="Overwrite existing per-state NPZ files")
    p.add_argument("--skip-sentinel-download", action="store_true", help="Only use existing Sentinel tensor files")
    p.add_argument("--s2-workers", type=int, default=1, help="Parallel Sentinel-2 Process API workers. Start with 4; too high may hit rate limits.")
    p.add_argument("--skip-era5", action="store_true", help="Skip ERA5 downloads")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    run(parser.parse_args())
