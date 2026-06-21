/**
 * Single city + date sample builder — the server-side equivalent of one (city, date) row from
 * build_temperature_dataset.py. Connects the real satellite scientific features (Copernicus
 * Statistics API) with the real weather features (Open-Meteo archive, forecast fallback) into the
 * 14-feature contract the RL/temperature models expect.
 *
 * The Sentinel-2 cloud mask itself is fetched separately by the browser via the `sentinel2` action.
 */
import { fetchSatelliteStat } from './sentinel_stats.mjs';

// 14-feature contract, in order. Satellite features carry their (source, field); weather features
// are pulled from Open-Meteo.
const SATELLITE_FEATURES = [
  ['s5p_cloud_fraction', 'sentinel5p', 'cloud_fraction'],
  ['s5p_cloud_optical_thickness', 'sentinel5p', 'cloud_optical_thickness'],
  ['s5p_cloud_top_height_m', 'sentinel5p', 'cloud_top_height'],
  ['s5p_cloud_top_pressure_pa', 'sentinel5p', 'cloud_top_pressure'],
  ['s5p_aerosol_index', 'sentinel5p', 'aerosol_index'],
  ['s3_humidity_pct', 'sentinel3olci', 'humidity'],
  ['s3_sea_level_pressure_hpa', 'sentinel3olci', 'sea_level_pressure'],
  ['s3_water_vapour_kg_m2', 'sentinel3olci', 'water_vapour'],
];

const WEATHER_FEATURES = [
  'wind_speed_10m_mean',
  'wind_direction_10m_dominant',
  'shortwave_radiation_sum',
  'precipitation_sum',
  'cloud_cover_mean',
  'surface_pressure_mean',
];

const FEATURE_NAMES = [...SATELLITE_FEATURES.map((f) => f[0]), ...WEATHER_FEATURES];

function round(value, digits) {
  const p = Math.pow(10, digits);
  return Math.round(value * p) / p;
}

function slug(value) {
  return String(value || 'sample').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

function bboxFromCenter(lat, lon, radiusKm) {
  const dLat = radiusKm / 111.32;
  const dLon = radiusKm / (111.32 * Math.max(0.2, Math.cos((lat * Math.PI) / 180)));
  return [round(lon - dLon, 6), round(lat - dLat, 6), round(lon + dLon, 6), round(lat + dLat, 6)];
}

function mean(values) {
  const v = values.filter((x) => Number.isFinite(x));
  if (!v.length) return null;
  return v.reduce((s, x) => s + x, 0) / v.length;
}

function circularMeanDegrees(values) {
  const rad = values.filter((x) => Number.isFinite(x)).map((x) => (x * Math.PI) / 180);
  if (!rad.length) return null;
  const sin = mean(rad.map(Math.sin));
  const cos = mean(rad.map(Math.cos));
  const deg = (Math.atan2(sin, cos) * 180) / Math.PI;
  return (deg + 360) % 360;
}

// Pull one day's weather aggregate from an Open-Meteo response (archive or forecast share schema).
function aggregateWeather(payload, date) {
  const daily = payload.daily || {};
  const dailyTimes = daily.time || [];
  let di = dailyTimes.indexOf(date);
  if (di < 0) return null;

  const dailyAt = (name) => {
    const arr = daily[name] || [];
    const n = Number(arr[di]);
    return Number.isFinite(n) ? n : null;
  };

  const hourly = payload.hourly || {};
  const hourlyTimes = hourly.time || [];
  const idx = hourlyTimes.reduce((acc, t, i) => {
    if (String(t).slice(0, 10) === date) acc.push(i);
    return acc;
  }, []);
  const hourlyPick = (name) => idx.map((i) => Number((hourly[name] || [])[i]));

  const temperature = dailyAt('temperature_2m_mean');
  if (temperature === null) return null;

  return {
    observed_temperature_c: temperature,
    weather: {
      wind_speed_10m_mean: dailyAt('wind_speed_10m_mean'),
      wind_direction_10m_dominant: dailyAt('wind_direction_10m_dominant'),
      shortwave_radiation_sum: dailyAt('shortwave_radiation_sum'),
      precipitation_sum: dailyAt('precipitation_sum'),
      cloud_cover_mean: mean(hourlyPick('cloud_cover')),
      surface_pressure_mean: mean(hourlyPick('surface_pressure')),
    },
  };
}

const DAILY_FIELDS = [
  'temperature_2m_mean',
  'precipitation_sum',
  'shortwave_radiation_sum',
  'wind_speed_10m_mean',
  'wind_direction_10m_dominant',
].join(',');
const HOURLY_FIELDS = ['cloud_cover', 'surface_pressure'].join(',');

async function fetchWeather(lat, lon, date) {
  // Archive matches the training pipeline exactly, but lags ~5 days. Try it first.
  const archiveUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}` +
    `&start_date=${date}&end_date=${date}&timezone=UTC&cell_selection=land` +
    `&daily=${DAILY_FIELDS}&hourly=${HOURLY_FIELDS}`;
  try {
    const res = await fetch(archiveUrl);
    if (res.ok) {
      const agg = aggregateWeather(await res.json(), date);
      if (agg) return { ...agg, source: 'open-meteo archive' };
    }
  } catch {
    /* fall through to forecast */
  }

  // Forecast covers recent/today (and a small past window). Use a wide past window to reach the date.
  const forecastUrl = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
    `&past_days=92&forecast_days=7&timezone=UTC` +
    `&daily=${DAILY_FIELDS}&hourly=${HOURLY_FIELDS}`;
  const res = await fetch(forecastUrl);
  if (!res.ok) throw new Error('Open-Meteo weather fetch failed (archive and forecast).');
  const agg = aggregateWeather(await res.json(), date);
  if (!agg) throw new Error(`No weather data available for ${date}.`);
  return { ...agg, source: 'open-meteo forecast' };
}

/**
 * Build one connected sample for (place, lat, lon, date).
 * Returns the same shape the frontend fetchWeatherSample produces, with real satellite + weather data.
 */
export async function buildSample({ place, lat, lon, date, bbox, windowDays = 7 }) {
  const latN = Number(lat);
  const lonN = Number(lon);
  if (!Number.isFinite(latN) || !Number.isFinite(lonN)) throw new Error('lat and lon are required numbers.');
  if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(String(date))) throw new Error('date must be YYYY-MM-DD.');
  const box = Array.isArray(bbox) && bbox.length === 4 ? bbox.map(Number) : bboxFromCenter(latN, lonN, 20);

  // Satellite stats (parallel; OAuth token is cached after the first request) + weather together.
  const [satResults, weather] = await Promise.all([
    Promise.all(
      SATELLITE_FEATURES.map(async ([featureName, source, field]) => {
        try {
          const stat = await fetchSatelliteStat({ source, field, date, bbox: box, windowDays });
          return { featureName, stat };
        } catch (error) {
          return { featureName, stat: null, error: error.message };
        }
      })
    ),
    fetchWeather(latN, lonN, date),
  ]);

  const inputs = {};
  const estimated = [];
  const satelliteOffsets = {};
  for (const { featureName, stat } of satResults) {
    if (stat && Number.isFinite(stat.value)) {
      inputs[featureName] = round(stat.value, 4);
      satelliteOffsets[featureName] = stat.offsetDays;
    } else {
      inputs[featureName] = null;
      estimated.push(featureName);
    }
  }

  const w = weather.weather;
  inputs.wind_speed_10m_mean = w.wind_speed_10m_mean === null ? null : round(w.wind_speed_10m_mean, 2);
  inputs.wind_direction_10m_dominant =
    w.wind_direction_10m_dominant === null
      ? round(circularMeanDegrees([w.wind_direction_10m_dominant].filter(Number.isFinite)) || 0, 1)
      : round(w.wind_direction_10m_dominant, 1);
  inputs.shortwave_radiation_sum = w.shortwave_radiation_sum === null ? null : round(w.shortwave_radiation_sum, 2);
  inputs.precipitation_sum = w.precipitation_sum === null ? 0 : round(w.precipitation_sum, 2);
  inputs.cloud_cover_mean = w.cloud_cover_mean === null ? null : round(w.cloud_cover_mean, 2);
  inputs.surface_pressure_mean = w.surface_pressure_mean === null ? null : round(w.surface_pressure_mean, 2);

  const feature_vector = FEATURE_NAMES.map((name) => inputs[name]);

  return {
    sample_id: `${slug(place || 'place')}_${date}`,
    city: place || '',
    lat: latN,
    lon: lonN,
    bbox: box,
    date,
    anchor: `${date}T00:00:00Z`,
    mask_path: 'sentinel2_cloud_mask.png',
    observed_temperature_c: round(weather.observed_temperature_c, 2),
    target_temperature_c: null,
    inputs,
    feature_vector,
    source_notes: {
      satellite_stats:
        estimated.length === 0
          ? `Live Copernicus Statistics (Sentinel-5P + Sentinel-3 OLCI), window ±${windowDays}d.`
          : `Live Copernicus stats with no pass for: ${estimated.join(', ')} (returned null).`,
      satellite_offsets_days: satelliteOffsets,
      weather: `${weather.source} daily aggregate.`,
      missing_features: estimated,
    },
  };
}
