/**
 * Copernicus Statistics API helper (ported from the local dev server, server.mjs).
 *
 * Pulls a single per-day aggregate value for a Sentinel-5P / Sentinel-3 OLCI band over a small
 * date window and returns the nearest non-empty day to the requested date. Satellite passes are
 * sparse, so a single date often has no acquisition — widening the window keeps the value real.
 *
 * Reuses the cached OAuth token from sentinel2.mjs (same Copernicus client credentials).
 */
import { getAccessToken } from './sentinel2.mjs';

const STATS_URL = process.env.COPERNICUS_STATS_URL || 'https://sh.dataspace.copernicus.eu/statistics/v1';

// Only the 8 fields that map into the 14-feature model contract (5 s5p_*, 3 s3_*).
const S5P_FIELDS = {
  cloud_fraction: { band: 'CLOUD_FRACTION' },
  cloud_optical_thickness: { band: 'CLOUD_OPTICAL_THICKNESS' },
  cloud_top_height: { band: 'CLOUD_TOP_HEIGHT' },
  cloud_top_pressure: { band: 'CLOUD_TOP_PRESSURE' },
  aerosol_index: { band: 'AER_AI_340_380' },
};

const S3_FIELDS = {
  humidity: { band: 'HUMIDITY' },
  sea_level_pressure: { band: 'SEA_LEVEL_PRESSURE' },
  water_vapour: { band: 'TOTAL_COLUMN_WATER_VAPOUR' },
};

function getStatFieldMeta(source, field) {
  if (source === 'sentinel5p') return S5P_FIELDS[field] || null;
  if (source === 'sentinel3olci') return S3_FIELDS[field] || null;
  return null;
}

function getStatisticsDataType(source) {
  if (source === 'sentinel5p') return 'sentinel-5p-l2';
  if (source === 'sentinel3olci') return 'sentinel-3-olci';
  throw new Error(`Statistics are not available for source ${source}.`);
}

function getStatisticsDataFilter(source) {
  if (source === 'sentinel5p') return { mosaickingOrder: 'mostRecent', timeliness: 'OFFL' };
  return { mosaickingOrder: 'mostRecent' };
}

function getStatisticsResolution(source) {
  if (source === 'sentinel5p') return 0.01;
  if (source === 'sentinel3olci') return 0.003;
  return 0.01;
}

function buildStatisticsEvalscript(band) {
  return `//VERSION=3
function setup() {
  return {
    input: ["${band}", "dataMask"],
    output: [
      { id: "value", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(sample) {
  return {
    value: [sample.${band}],
    dataMask: [sample.dataMask]
  };
}`;
}

function buildStatisticsRequest({ source, fieldMeta, from, to, bbox }) {
  const resolution = getStatisticsResolution(source);
  return {
    input: {
      bounds: {
        bbox,
        properties: { crs: 'http://www.opengis.net/def/crs/OGC/1.3/CRS84' },
      },
      data: [{ type: getStatisticsDataType(source), dataFilter: getStatisticsDataFilter(source) }],
    },
    aggregation: {
      timeRange: { from, to },
      aggregationInterval: { of: 'P1D' },
      resx: resolution,
      resy: resolution,
      evalscript: buildStatisticsEvalscript(fieldMeta.band),
    },
  };
}

function shiftIsoDate(date, days) {
  const d = new Date(`${date}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function daysBetween(a, b) {
  const da = new Date(`${a}T00:00:00Z`).getTime();
  const db = new Date(`${b}T00:00:00Z`).getTime();
  return Math.round((da - db) / 86400000);
}

/**
 * Fetch one daily-aggregate value for a satellite band over a window centered on `date`,
 * returning the nearest non-empty day.
 * @returns {Promise<{value:number, usedDate:string, offsetDays:number}|null>}
 */
export async function fetchSatelliteStat({ source, field, date, bbox, windowDays = 7 }) {
  const fieldMeta = getStatFieldMeta(source, field);
  if (!fieldMeta) throw new Error(`Unknown satellite field ${source}/${field}.`);
  if (!Array.isArray(bbox) || bbox.length !== 4) throw new Error('bbox must be [minLon, minLat, maxLon, maxLat].');

  const from = `${shiftIsoDate(date, -windowDays)}T00:00:00Z`;
  const to = `${shiftIsoDate(date, windowDays + 1)}T00:00:00Z`;

  const accessToken = await getAccessToken();
  const requestBody = buildStatisticsRequest({ source, fieldMeta, from, to, bbox: bbox.map(Number) });

  const response = await fetch(STATS_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Copernicus statistics request failed (${response.status}): ${text.slice(0, 300)}`);
  }

  const payload = JSON.parse(text);
  let best = null;
  for (const entry of payload.data || []) {
    const day = String(entry.interval?.from || '').slice(0, 10);
    const stats = entry.outputs?.value?.bands?.B0?.stats;
    if (!day || !stats) continue;
    const mean = Number(stats.mean ?? stats.average);
    const sampleCount = Number(stats.sampleCount ?? 0);
    if (!Number.isFinite(mean) || sampleCount <= 0) continue;
    const offset = Math.abs(daysBetween(day, date));
    if (!best || offset < best.offset) {
      best = { value: mean, usedDate: day, offset };
    }
  }

  if (!best) return null;
  return { value: best.value, usedDate: best.usedDate, offsetDays: daysBetween(best.usedDate, date) };
}
