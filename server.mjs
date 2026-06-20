import { createServer } from "node:http";
import { createReadStream, existsSync, readFileSync } from "node:fs";
import { stat } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";

const root = process.cwd();
const port = Number(process.env.PORT || 5173);
const maxBodyBytes = 2_000_000;
const latestItemCount = 7;

const defaults = {
  tokenUrl:
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
  processUrl: "https://sh.dataspace.copernicus.eu/process/v1",
  catalogUrl: "https://sh.dataspace.copernicus.eu/catalog/v1/search",
  statsUrl: "https://sh.dataspace.copernicus.eu/statistics/v1",
  meteoUrl: "https://archive-api.open-meteo.com/v1/archive",
};

const S5P_FIELDS = {
  cloud_fraction: {
    band: "CLOUD_FRACTION",
    min: 0,
    max: 1,
    label: "Cloud fraction",
    unit: "fraction",
  },
  cloud_optical_thickness: {
    band: "CLOUD_OPTICAL_THICKNESS",
    min: 0,
    max: 250,
    label: "Cloud thickness",
    unit: "a.u.",
  },
  cloud_top_height: {
    band: "CLOUD_TOP_HEIGHT",
    min: 0,
    max: 20000,
    label: "Cloud top height",
    unit: "m",
  },
  cloud_top_pressure: {
    band: "CLOUD_TOP_PRESSURE",
    min: 1000,
    max: 110000,
    label: "Cloud top pressure",
    unit: "Pa",
  },
  no2: { band: "NO2", min: 0, max: 0.0003, label: "NO2", unit: "mol/m^2" },
  o3: { band: "O3", min: 0, max: 0.36, label: "Ozone", unit: "mol/m^2" },
  co: { band: "CO", min: 0, max: 0.1, label: "CO", unit: "mol/m^2" },
  ch4: { band: "CH4", min: 1600, max: 2000, label: "Methane", unit: "ppb" },
  so2: { band: "SO2", min: 0, max: 0.01, label: "SO2", unit: "mol/m^2" },
  aerosol_index: {
    band: "AER_AI_340_380",
    min: -1,
    max: 5,
    label: "Aerosol index",
    unit: "index",
  },
};

const S3_FIELDS = {
  cloud_pressure_proxy: {
    band: "B15",
    min: 0,
    max: 1,
    label: "Cloud pressure proxy",
    unit: "reflectance",
  },
  humidity: { band: "HUMIDITY", min: 0, max: 100, label: "Humidity", unit: "%" },
  sea_level_pressure: {
    band: "SEA_LEVEL_PRESSURE",
    min: 980,
    max: 1030,
    label: "Sea level pressure",
    unit: "hPa",
  },
  water_vapour: {
    band: "TOTAL_COLUMN_WATER_VAPOUR",
    min: 0,
    max: 70,
    label: "Water vapour",
    unit: "kg/m^2",
  },
};

const SLSTR_FIELDS = {
  cloud_screening: {
    band: "S1",
    min: 0,
    max: 1,
    label: "Cloud screening",
    unit: "reflectance",
  },
  cloud_flagging: {
    band: "S3",
    min: 0,
    max: 1,
    label: "Cloud flagging",
    unit: "reflectance",
  },
  cirrus_detection: {
    band: "S4",
    min: 0,
    max: 1,
    label: "Cirrus detection",
    unit: "reflectance",
  },
  cloud_clearing: {
    band: "S5",
    min: 0,
    max: 1,
    label: "Cloud clearing",
    unit: "reflectance",
  },
};

const ERA5_FIELDS = {
  shortwave_radiation: {
    hourly: "shortwave_radiation",
    min: 0,
    max: 1000,
    label: "Shortwave radiation",
    unit: "W/m2",
    color: "#ffb454",
  },
  direct_radiation: {
    hourly: "direct_radiation",
    min: 0,
    max: 1000,
    label: "Direct radiation",
    unit: "W/m2",
    color: "#ff9b42",
  },
  diffuse_radiation: {
    hourly: "diffuse_radiation",
    min: 0,
    max: 1000,
    label: "Diffuse radiation",
    unit: "W/m2",
    color: "#f2c94c",
  },
  direct_normal_irradiance: {
    hourly: "direct_normal_irradiance",
    min: 0,
    max: 1200,
    label: "Direct normal irradiance",
    unit: "W/m2",
    color: "#f08b3c",
  },
  global_tilted_radiation: {
    hourly: "global_tilted_radiation",
    min: 0,
    max: 1200,
    label: "Global tilted radiation",
    unit: "W/m2",
    color: "#f5b45a",
  },
  terrestrial_radiation: {
    hourly: "terrestrial_radiation",
    min: 0,
    max: 700,
    label: "Terrestrial radiation",
    unit: "W/m2",
    color: "#f0a04b",
  },
  wind_speed_10m: { hourly: "wind_speed_10m", min: 0, max: 35, label: "Wind speed", unit: "m/s", color: "#73a5ff" },
  wind_direction_10m: { hourly: "wind_direction_10m", min: 0, max: 360, label: "Wind direction", unit: "deg", color: "#c58eff" },
  relative_humidity_2m: { hourly: "relative_humidity_2m", min: 0, max: 100, label: "Humidity", unit: "%", color: "#56d0d3" },
  cloud_cover: { hourly: "cloud_cover", min: 0, max: 100, label: "Cloud cover", unit: "%", color: "#a7b7cc" },
  cloud_cover_low: { hourly: "cloud_cover_low", min: 0, max: 100, label: "Cloud cover low", unit: "%", color: "#a7b7cc" },
  cloud_cover_mid: { hourly: "cloud_cover_mid", min: 0, max: 100, label: "Cloud cover mid", unit: "%", color: "#a7b7cc" },
  cloud_cover_high: { hourly: "cloud_cover_high", min: 0, max: 100, label: "Cloud cover high", unit: "%", color: "#a7b7cc" },
};

let tokenCache = null;

loadDotEnv();
loadLegacyConfig();

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", `http://${request.headers.host}`);

    if (request.method === "GET" && url.pathname === "/api/config") {
      sendJson(response, 200, getPublicConfig());
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/latest-scenes") {
      await handleLatestScenes(request, response);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/process") {
      await handleProcess(request, response);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/statistics") {
      await handleStatistics(request, response);
      return;
    }

    if (request.method === "GET" || request.method === "HEAD") {
      await serveStatic(url.pathname, response, request.method === "HEAD");
      return;
    }

    sendJson(response, 405, { message: "Method not allowed." });
  } catch (error) {
    sendJson(response, 500, {
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

server.listen(port, () => {
  console.log(`Stara Zagora viewer running at http://localhost:${port}`);
});

async function handleLatestScenes(request, response) {
  const body = await readJsonBody(request);
  const source = normalizeSource(body.source);
  const limit = clampNumber(body.limit, 1, 20, latestItemCount);

  if (source === "era5") {
    sendJson(response, 200, {
      source,
      scenes: buildRecentDays(limit).map((date) => ({
        source,
        date,
        label: date,
        detail: "ERA5 daily weather",
      })),
    });
    return;
  }

  if (source === "sentinel5p") {
    const scenes = await fetchCopernicusScenes({
      source,
      collection: "sentinel-5p-l2",
      limit,
    });
    sendJson(response, 200, { source, scenes });
    return;
  }

  if (source === "sentinel2") {
    const scenes = await fetchCopernicusScenes({
      source,
      collection: "sentinel-2-l2a",
      limit,
    });
    sendJson(response, 200, { source, scenes });
    return;
  }

  if (source === "sentinel3olci") {
    const scenes = await fetchCopernicusScenes({
      source,
      collection: "sentinel-3-olci",
      limit,
    });
    sendJson(response, 200, { source, scenes });
    return;
  }

  if (source === "sentinel3slstr") {
    const scenes = await fetchCopernicusScenes({
      source,
      collection: "sentinel-3-slstr",
      limit,
    });
    sendJson(response, 200, { source, scenes });
    return;
  }

  sendJson(response, 400, { message: `Unknown source: ${source}` });
}

async function handleProcess(request, response) {
  const body = await readJsonBody(request);
  const source = normalizeSource(body.source);

  if (source === "era5") {
    const field = body.field || "cloud_cover";
    const svg = await renderEra5Svg(body.item?.date, field, body);
    response.writeHead(200, {
      "Content-Type": "image/svg+xml; charset=utf-8",
      "Cache-Control": "no-store",
    });
    response.end(svg);
    return;
  }

  if (source === "sentinel5p") {
    await proxyCopernicusProcess(
      response,
      buildSentinel5pProcessRequest(body.item, body.field, body),
      body.item?.label || body.item?.date || "Sentinel-5P scene"
    );
    return;
  }

  if (source === "sentinel2") {
    await proxyCopernicusProcess(
      response,
      buildSentinel2ProcessRequest(body.item, body.field, body),
      body.item?.label || body.item?.date || "Sentinel-2 scene"
    );
    return;
  }

  if (source === "sentinel3olci") {
    await proxyCopernicusProcess(
      response,
      buildSentinel3OlciProcessRequest(body.item, body.field, body),
      body.item?.label || body.item?.date || "Sentinel-3 OLCI scene"
    );
    return;
  }

  if (source === "sentinel3slstr") {
    await proxyCopernicusProcess(
      response,
      buildSentinel3SlstrProcessRequest(body.item, body.field, body),
      body.item?.label || body.item?.date || "Sentinel-3 SLSTR scene"
    );
    return;
  }

  sendJson(response, 400, { message: `Unknown source: ${source}` });
}

async function handleStatistics(request, response) {
  const body = await readJsonBody(request);
  const source = normalizeSource(body.source);
  const field = String(body.field || "").trim();
  const days = clampNumber(body.days, 1, 30, latestItemCount);
  const bbox = parseBbox(body.bbox) || getPublicConfig().bbox;

  if (source === "era5") {
    const meta = ERA5_FIELDS[field] || ERA5_FIELDS.cloud_cover;
    const points = await buildEra5StatisticsSeries(meta.hourly, days, bbox);
    sendJson(response, 200, {
      source,
      field,
      label: meta.label,
      unit: meta.unit,
      points,
    });
    return;
  }

  const fieldMeta = getStatFieldMeta(source, field);

  if (!fieldMeta) {
    sendJson(response, 400, { message: `Statistics are not available for ${source} / ${field}.` });
    return;
  }

  const points = await fetchCopernicusStatisticsSeries({
    source,
    field,
    fieldMeta,
    days,
    bbox,
  });

  sendJson(response, 200, {
    source,
    field,
    label: fieldMeta.label,
    unit: fieldMeta.unit || null,
    points,
  });
}

async function proxyCopernicusProcess(response, requestBody, description) {
  if (!process.env.COPERNICUS_CLIENT_ID || !process.env.COPERNICUS_CLIENT_SECRET) {
    sendJson(response, 500, {
      message:
        "Missing COPERNICUS_CLIENT_ID or COPERNICUS_CLIENT_SECRET. Copy .env.example to .env and fill in your Copernicus OAuth credentials.",
    });
    return;
  }

  const accessToken = await getAccessToken();
  const upstreamResponse = await fetch(process.env.COPERNICUS_PROCESS_URL || defaults.processUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  const payload = Buffer.from(await upstreamResponse.arrayBuffer());
  response.writeHead(upstreamResponse.status, {
    "Content-Type": upstreamResponse.headers.get("content-type") || "application/octet-stream",
    "Cache-Control": "no-store",
  });
  response.end(payload);
}

async function fetchCopernicusScenes({ source, collection, limit }) {
  const accessToken = await getAccessToken();
  const end = new Date();
  const windowsInDays = [7, 14, 30, 60, 120, 240, 365];
  const byDate = new Map();
  const sourceDetail = {
    sentinel2: "Sentinel-2 surface scene",
    sentinel5p: "Sentinel-5P daily pass",
    sentinel3olci: "Sentinel-3 OLCI atmospheric scene",
    sentinel3slstr: "Sentinel-3 SLSTR cloud proxy",
  }[source] || "Scene";

  for (const days of windowsInDays) {
    const start = new Date(end);
    start.setUTCDate(start.getUTCDate() - days);

    const catalog = await searchCatalog(accessToken, {
      collection,
      from: start.toISOString(),
      to: end.toISOString(),
    });

    const features = [...(catalog.features || [])].sort((left, right) =>
      String(right.properties?.datetime || "").localeCompare(String(left.properties?.datetime || ""))
    );

    for (const feature of features) {
      const datetime = feature.properties?.datetime;
      if (!datetime) {
        continue;
      }

      const date = datetime.slice(0, 10);
      if (!byDate.has(date)) {
        byDate.set(date, {
          source,
          date,
          datetime,
          label: date,
          detail: sourceDetail,
        });
      }
    }

    if (byDate.size >= limit) {
      break;
    }
  }

  return [...byDate.values()]
    .sort((left, right) => right.datetime.localeCompare(left.datetime))
    .slice(0, limit);
}

async function fetchCopernicusStatisticsSeries({ source, field, fieldMeta, days, bbox }) {
  const accessToken = await getAccessToken();
  const end = new Date();
  end.setUTCHours(0, 0, 0, 0);
  end.setUTCDate(end.getUTCDate() + 1);
  const start = new Date(end);
  start.setUTCDate(start.getUTCDate() - days);
  const requestBody = buildStatisticsRequest({
    source,
    field,
    fieldMeta,
    from: start.toISOString(),
    to: end.toISOString(),
    days,
    bbox,
  });

  const statsResponse = await fetch(process.env.COPERNICUS_STATS_URL || defaults.statsUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  const text = await statsResponse.text();

  if (!statsResponse.ok) {
    throw new Error(`Copernicus statistics request failed: ${text}`);
  }

  const payload = JSON.parse(text);
  const statsByDate = new Map();

  for (const entry of payload.data || []) {
    const date = String(entry.interval?.from || "").slice(0, 10);
    const stats = entry.outputs?.value?.bands?.B0?.stats;

    if (!date || !stats) {
      continue;
    }

    statsByDate.set(date, {
      date,
      mean: Number(stats.mean ?? stats.average ?? 0),
      min: Number(stats.min ?? stats.minimum ?? 0),
      max: Number(stats.max ?? stats.maximum ?? 0),
      sampleCount: Number(stats.sampleCount ?? 0),
      noDataCount: Number(stats.noDataCount ?? 0),
    });
  }

  return buildRecentDays(days)
    .reverse()
    .map((date) =>
      statsByDate.get(date) || {
        date,
        mean: null,
        min: null,
        max: null,
        sampleCount: 0,
        noDataCount: 0,
      }
    );
}

async function searchCatalog(accessToken, { collection, from, to }) {
  const catalogBody = {
    bbox: getPublicConfig().bbox,
    datetime: `${from}/${to}`,
    collections: [collection],
    limit: 100,
    fields: {
      include: ["id", "properties.datetime"],
    },
  };

  const catalogResponse = await fetch(process.env.COPERNICUS_CATALOG_URL || defaults.catalogUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(catalogBody),
  });

  const text = await catalogResponse.text();

  if (!catalogResponse.ok) {
    throw new Error(`Copernicus catalog request failed: ${text}`);
  }

  return JSON.parse(text);
}

function buildSentinel5pProcessRequest(item, field, options) {
  const fieldMeta = S5P_FIELDS[field] || S5P_FIELDS.cloud_fraction;
  const date = item?.date;

  return {
    input: {
      bounds: {
        bbox: options.bbox || getPublicConfig().bbox,
      },
      data: [
        {
          type: "sentinel-5p-l2",
          dataFilter: {
            timeRange: {
              from: `${date}T00:00:00Z`,
              to: `${date}T23:59:59Z`,
            },
            mosaickingOrder: "mostRecent",
            timeliness: "OFFL",
          },
        },
      ],
    },
    output: {
      width: clampNumber(options.width, 32, 4096, 768),
      height: clampNumber(options.height, 32, 4096, 480),
      responses: [
        {
          identifier: "default",
          format: { type: "image/png" },
        },
      ],
    },
    evalscript: buildSingleBandEvalscript(fieldMeta.band, fieldMeta.min, fieldMeta.max),
  };
}

function buildSentinel3OlciProcessRequest(item, field, options) {
  const fieldMeta = S3_FIELDS[field] || S3_FIELDS.cloud_pressure_proxy;
  const date = item?.date || item?.datetime?.slice(0, 10);

  return {
    input: {
      bounds: {
        bbox: options.bbox || getPublicConfig().bbox,
      },
      data: [
        {
          type: "sentinel-3-olci",
          dataFilter: {
            timeRange: {
              from: `${date}T00:00:00Z`,
              to: `${date}T23:59:59Z`,
            },
            mosaickingOrder: "mostRecent",
          },
          processing: {
            upsampling: "NEAREST",
            downsampling: "NEAREST",
          },
        },
      ],
    },
    output: {
      width: clampNumber(options.width, 32, 4096, 768),
      height: clampNumber(options.height, 32, 4096, 480),
      responses: [
        {
          identifier: "default",
          format: { type: "image/png" },
        },
      ],
    },
    evalscript: buildSingleBandEvalscript(fieldMeta.band, fieldMeta.min, fieldMeta.max),
  };
}

function buildSentinel3SlstrProcessRequest(item, field, options) {
  const fieldMeta = SLSTR_FIELDS[field] || SLSTR_FIELDS.cloud_screening;
  const date = item?.date || item?.datetime?.slice(0, 10);

  return {
    input: {
      bounds: {
        bbox: options.bbox || getPublicConfig().bbox,
      },
      data: [
        {
          type: "sentinel-3-slstr",
          dataFilter: {
            timeRange: {
              from: `${date}T00:00:00Z`,
              to: `${date}T23:59:59Z`,
            },
            mosaickingOrder: "mostRecent",
          },
          processing: {
            upsampling: "NEAREST",
            downsampling: "NEAREST",
          },
        },
      ],
    },
    output: {
      width: clampNumber(options.width, 32, 4096, 768),
      height: clampNumber(options.height, 32, 4096, 480),
      responses: [
        {
          identifier: "default",
          format: { type: "image/png" },
        },
      ],
    },
    evalscript: buildSingleBandEvalscript(fieldMeta.band, fieldMeta.min, fieldMeta.max),
  };
}

function buildSentinel2ProcessRequest(item, field, options) {
  const date = item?.date || item?.datetime?.slice(0, 10);
  const mode = field || "natural_color";

  return {
    input: {
      bounds: {
        bbox: options.bbox || getPublicConfig().bbox,
      },
      data: [
        {
          type: "sentinel-2-l2a",
          dataFilter: {
            timeRange: {
              from: `${date}T00:00:00Z`,
              to: `${date}T23:59:59Z`,
            },
            mosaickingOrder: "leastCC",
          },
          processing: {
            upsampling: "NEAREST",
            downsampling: "NEAREST",
          },
        },
      ],
    },
    output: {
      width: clampNumber(options.width, 32, 4096, 768),
      height: clampNumber(options.height, 32, 4096, 480),
      responses: [
        {
          identifier: "default",
          format: { type: "image/png" },
        },
      ],
    },
    evalscript: buildSentinel2Evalscript(mode),
  };
}

function buildStatisticsRequest({ source, field, fieldMeta, from, to, days, bbox }) {
  const resolution = getStatisticsResolution(source, field, days);
  const dataType = getStatisticsDataType(source);

  return {
    input: {
      bounds: {
        bbox: bbox || getPublicConfig().bbox,
        properties: {
          crs: "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
        },
      },
      data: [
        {
          type: dataType,
          dataFilter: getStatisticsDataFilter(source),
        },
      ],
    },
    aggregation: {
      timeRange: {
        from,
        to,
      },
      aggregationInterval: {
        of: "P1D",
      },
      resx: resolution,
      resy: resolution,
      evalscript: buildStatisticsEvalscript(fieldMeta.band, fieldMeta.normalize),
    },
  };
}

function buildStatisticsEvalscript(band, normalizeValue) {
  const normalizeBody = typeof normalizeValue === "function" ? normalizeValue.toString() : null;

  return `//VERSION=3
function setup() {
  return {
    input: ["${band}", "dataMask"],
    output: [
      {
        id: "value",
        bands: 1,
        sampleType: "FLOAT32"
      },
      {
        id: "dataMask",
        bands: 1
      }
    ]
  };
}

function normalize(value) {
  ${normalizeBody ? `return (${normalizeBody})(value);` : "return value;"}
}

function evaluatePixel(sample) {
  return {
    value: [normalize(sample.${band})],
    dataMask: [sample.dataMask]
  };
}`;
}

async function buildEra5StatisticsSeries(hourlyVar, days, bbox) {
  const dates = buildRecentDays(days).reverse();
  const points = [];

  for (const date of dates) {
    const series = await fetchEra5Series(date, hourlyVar, bbox);
    points.push({
      date,
      mean: series.mean,
      min: null,
      max: null,
      sampleCount: series.values.length,
      noDataCount: 0,
    });
  }

  return points;
}

function buildSingleBandEvalscript(band, min, max) {
  return `//VERSION=3
function setup() {
  return {
    input: ["${band}", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [0, 0, 0, 0];
  var value = clamp(sample.${band}, ${min}, ${max});
  var t = (value - ${min}) / (${max} - ${min});
  return [
    0.08 + t * 0.86,
    0.18 + t * 0.72,
    0.35 + (1 - t) * 0.45,
    1
  ];
}`;
}

function buildSentinel2Evalscript(mode) {
  if (mode === "cloud_mask") {
    return `//VERSION=3
function setup() {
  return {
    input: ["CLM", "SCL", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [0, 0, 0, 0];
  if (sample.CLM === 1 || [8, 9, 10].includes(sample.SCL)) return [1, 1, 1, 1];
  return [0.05, 0.14, 0.2, 1];
}`;
  }

  if (mode === "cloud_probability") {
    return `//VERSION=3
function setup() {
  return {
    input: ["CLP", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [0, 0, 0, 0];
  var t = Math.max(0, Math.min(1, sample.CLP / 255));
  return [t, 0.5 + t * 0.2, 1 - t * 0.6, 1];
}`;
  }

  return `//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [0, 0, 0, 0];
  return [
    Math.min(1, sample.B04 * 2.5),
    Math.min(1, sample.B03 * 2.5),
    Math.min(1, sample.B02 * 2.5),
    1
  ];
}`;
}

async function renderEra5Svg(date, field, options) {
  const meta = ERA5_FIELDS[field] || ERA5_FIELDS.cloud_cover;
  const series = await fetchEra5Series(date, meta.hourly);
  const width = clampNumber(options.width, 32, 4096, 768);
  const height = clampNumber(options.height, 32, 4096, 480);
  const values = series.values;
  const bars = 24;
  const padding = 28;
  const chartTop = 108;
  const chartHeight = Math.max(120, height - 180);
  const innerWidth = width - padding * 2;
  const barWidth = innerWidth / bars;

  const max = meta.max;
  const min = meta.min;

  const barRects = Array.from({ length: bars }, (_, index) => {
    const value = clampNumber(values[index] ?? values[values.length - 1] ?? 0, min, max, min);
    const normalized = (value - min) / (max - min);
    const barHeight = Math.max(4, normalized * chartHeight);
    const x = padding + index * barWidth;
    const y = chartTop + chartHeight - barHeight;
    return `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${Math.max(
      1,
      barWidth - 2
    ).toFixed(2)}" height="${barHeight.toFixed(2)}" rx="3" fill="${meta.color}" />`;
  }).join("");

  const linePoints = Array.from({ length: bars }, (_, index) => {
    const value = clampNumber(values[index] ?? values[values.length - 1] ?? 0, min, max, min);
    const normalized = (value - min) / (max - min);
    const x = padding + index * barWidth + barWidth / 2;
    const y = chartTop + chartHeight - normalized * chartHeight;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(
    meta.label
  )} for ${escapeXml(date || "selected day")}">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#16374b" />
      <stop offset="45%" stop-color="#2f6f8b" />
      <stop offset="100%" stop-color="#d8eef2" />
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="${width}" height="${height}" fill="url(#sky)" />
  <rect x="18" y="18" width="${width - 36}" height="${height - 36}" rx="12" fill="rgba(255,255,255,0.14)" />
  <text x="32" y="54" fill="#f8fbfc" font-size="24" font-weight="700">${escapeXml(
    date || "ERA5"
  )}</text>
  <text x="32" y="80" fill="rgba(248,251,252,0.88)" font-size="14">${escapeXml(
    `${meta.label} average: ${series.mean.toFixed(meta.unit === "%" ? 0 : 1)}${meta.unit}`
  )}</text>
  <rect x="28" y="96" width="${width - 56}" height="${height - 124}" rx="10" fill="rgba(8,22,30,0.22)" />
  ${barRects}
  <polyline fill="none" stroke="${meta.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${linePoints}" />
  <text x="${width - 32}" y="${height - 28}" text-anchor="end" fill="rgba(248,251,252,0.82)" font-size="13">${escapeXml(
    meta.unit ? `${meta.unit} over 24 hours` : "24 hour series"
  )}</text>
</svg>`;
}

async function fetchEra5Series(date, hourlyVar, bbox) {
  const center = bboxCenter(bbox || getPublicConfig().bbox);
  const url = new URL(process.env.COPERNICUS_METEO_URL || defaults.meteoUrl);

  url.searchParams.set("latitude", String(center.lat));
  url.searchParams.set("longitude", String(center.lon));
  url.searchParams.set("start_date", date);
  url.searchParams.set("end_date", date);
  url.searchParams.set("hourly", hourlyVar);
  url.searchParams.set("timezone", "Europe/Sofia");
  url.searchParams.set("cell_selection", "land");

  const response = await fetch(url);
  const text = await response.text();

  if (!response.ok) {
    throw new Error(`ERA5 request failed: ${text}`);
  }

  const payload = JSON.parse(text);
  const values = payload.hourly?.[hourlyVar] || [];
  return {
    values,
    mean: average(values),
  };
}

async function getAccessToken() {
  const now = Math.floor(Date.now() / 1000);

  if (tokenCache && tokenCache.expiresAt - 60 > now) {
    return tokenCache.accessToken;
  }

  const body = new URLSearchParams({
    grant_type: "client_credentials",
    client_id: process.env.COPERNICUS_CLIENT_ID,
    client_secret: process.env.COPERNICUS_CLIENT_SECRET,
  });

  const tokenResponse = await fetch(process.env.COPERNICUS_TOKEN_URL || defaults.tokenUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body,
  });

  const text = await tokenResponse.text();

  if (!tokenResponse.ok) {
    throw new Error(`Copernicus token request failed: ${text}`);
  }

  const token = JSON.parse(text);
  tokenCache = {
    accessToken: token.access_token,
    expiresAt: now + Number(token.expires_in || 3600),
  };

  return tokenCache.accessToken;
}

async function readJsonBody(request) {
  const chunks = [];
  let size = 0;

  for await (const chunk of request) {
    size += chunk.length;

    if (size > maxBodyBytes) {
      throw new Error("Request body is too large.");
    }

    chunks.push(chunk);
  }

  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
}

async function serveStatic(pathname, response, headOnly) {
  if (pathname === "/config.js") {
    sendJson(response, 410, {
      message:
        "config.js is not served because credentials belong in .env and the local proxy.",
    });
    return;
  }

  const relativePath = pathname === "/" ? "index.html" : pathname.slice(1);
  const requestedPath = resolve(root, normalize(relativePath));

  if (!requestedPath.startsWith(root)) {
    sendJson(response, 403, { message: "Forbidden." });
    return;
  }

  const fileStats = await stat(requestedPath).catch(() => null);

  if (!fileStats?.isFile()) {
    sendJson(response, 404, { message: "Not found." });
    return;
  }

  response.writeHead(200, {
    "Content-Type": getContentType(requestedPath),
    "Content-Length": fileStats.size,
    "Cache-Control": "no-store",
  });

  if (headOnly) {
    response.end();
    return;
  }

  createReadStream(requestedPath).pipe(response);
}

function loadDotEnv() {
  const envPath = join(root, ".env");

  if (!existsSync(envPath)) {
    return;
  }

  const contents = readFileSync(envPath, "utf8");

  for (const line of contents.split(/\r?\n/)) {
    const trimmed = line.trim();

    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) {
      continue;
    }

    const [key, ...valueParts] = trimmed.split("=");
    const value = valueParts.join("=").trim().replace(/^["']|["']$/g, "");

    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}

function loadLegacyConfig() {
  const legacyPath = join(root, "config.js");

  if (!existsSync(legacyPath)) {
    return;
  }

  const contents = readFileSync(legacyPath, "utf8");
  const mappings = {
    COPERNICUS_CLIENT_ID: /clientId\s*:\s*["']([^"']+)["']/,
    COPERNICUS_CLIENT_SECRET: /clientSecret\s*:\s*["']([^"']+)["']/,
    COPERNICUS_TOKEN_URL: /tokenUrl\s*:\s*["']([^"']+)["']/,
    COPERNICUS_PROCESS_URL: /processUrl\s*:\s*["']([^"']+)["']/,
    COPERNICUS_BBOX: /bbox\s*:\s*\[([^\]]+)\]/,
    COPERNICUS_IMAGE_WIDTH: /width\s*:\s*(\d+)/,
    COPERNICUS_IMAGE_HEIGHT: /height\s*:\s*(\d+)/,
  };

  for (const [envKey, pattern] of Object.entries(mappings)) {
    const match = contents.match(pattern);

    if (match?.[1] && !process.env[envKey]) {
      process.env[envKey] = match[1];
    }
  }
}

function getPublicConfig() {
  return {
    bbox: parseBbox(process.env.COPERNICUS_BBOX) || [25.43, 42.31, 25.86, 42.57],
    width: Number(process.env.COPERNICUS_IMAGE_WIDTH || 768),
    height: Number(process.env.COPERNICUS_IMAGE_HEIGHT || 480),
  };
}

function parseBbox(value) {
  if (!value) {
    return null;
  }

  if (Array.isArray(value)) {
    const bbox = value
      .map((part) => Number(part))
      .filter((part) => Number.isFinite(part));

    return bbox.length === 4 ? bbox : null;
  }

  const bbox = value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((part) => Number.isFinite(part));

  return bbox.length === 4 ? bbox : null;
}

function buildRecentDays(count) {
  const dates = [];

  for (let offset = 0; offset < count; offset += 1) {
    const date = new Date();
    date.setUTCHours(0, 0, 0, 0);
    date.setUTCDate(date.getUTCDate() - offset);
    dates.push(toIsoDate(date));
  }

  return dates;
}

function toIsoDate(date) {
  return date.toISOString().slice(0, 10);
}

function bboxCenter(bbox) {
  const [west, south, east, north] = bbox;
  return {
    lon: (west + east) / 2,
    lat: (south + north) / 2,
  };
}

function average(values) {
  const filtered = values.filter((value) => Number.isFinite(value));

  if (filtered.length === 0) {
    return 0;
  }

  return filtered.reduce((sum, value) => sum + value, 0) / filtered.length;
}

function normalizeSource(source) {
  const value = String(source || "era5").toLowerCase();

  if (value === "sentinel-2" || value === "s2") {
    return "sentinel2";
  }

  if (value === "sentinel-5p" || value === "s5p") {
    return "sentinel5p";
  }

  if (value === "sentinel-3-olci" || value === "sentinel3" || value === "s3olci") {
    return "sentinel3olci";
  }

  if (value === "sentinel-3-slstr" || value === "sentinel3slstr" || value === "s3slstr" || value === "slstr") {
    return "sentinel3slstr";
  }

  if (value === "era-5" || value === "era_5") {
    return "era5";
  }

  return value;
}

function getStatFieldMeta(source, field) {
  if (source === "sentinel5p") {
    return S5P_FIELDS[field] || null;
  }

  if (source === "sentinel3olci") {
    return S3_FIELDS[field] || null;
  }

  if (source === "sentinel3slstr") {
    return SLSTR_FIELDS[field] || null;
  }

  if (source === "sentinel2") {
    if (field === "cloud_probability") {
      return {
        band: "CLP",
        min: 0,
        max: 255,
        label: "Cloud probability",
        unit: "fraction",
        normalize: (value) => value / 255,
      };
    }

    if (field === "cloud_mask") {
      return {
        band: "CLM",
        min: 0,
        max: 1,
        label: "Cloud mask",
        unit: "fraction",
        normalize: (value) => value,
      };
    }
  }

  return null;
}

function getStatisticsDataType(source) {
  if (source === "sentinel5p") {
    return "sentinel-5p-l2";
  }

  if (source === "sentinel3olci") {
    return "sentinel-3-olci";
  }

  if (source === "sentinel3slstr") {
    return "sentinel-3-slstr";
  }

  if (source === "sentinel2") {
    return "sentinel-2-l2a";
  }

  throw new Error(`Statistics are not available for source ${source}.`);
}

function getStatisticsDataFilter(source) {
  if (source === "sentinel5p") {
    return {
      mosaickingOrder: "mostRecent",
      timeliness: "OFFL",
    };
  }

  if (source === "sentinel2") {
    return {
      mosaickingOrder: "leastCC",
    };
  }

  return {
    mosaickingOrder: "mostRecent",
  };
}

function getStatisticsResolution(source, field, days) {
  if (source === "sentinel5p") {
    return 0.01;
  }

  if (source === "sentinel3olci") {
    return 0.003;
  }

  if (source === "sentinel3slstr") {
    return 0.005;
  }

  if (source === "sentinel2") {
    return field === "cloud_mask" ? 0.001 : 0.0005;
  }

  return 0.01;
}

function clampNumber(value, min, max, fallback) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return fallback;
  }

  return Math.min(max, Math.max(min, number));
}

function escapeXml(value) {
  return String(value).replace(/[<>&"']/g, (character) => {
    const replacements = {
      "<": "&lt;",
      ">": "&gt;",
      "&": "&amp;",
      '"': "&quot;",
      "'": "&apos;",
    };
    return replacements[character];
  });
}

function sendJson(response, status, payload) {
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  response.end(JSON.stringify(payload));
}

function getContentType(filePath) {
  const types = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
  };

  return types[extname(filePath).toLowerCase()] || "application/octet-stream";
}
