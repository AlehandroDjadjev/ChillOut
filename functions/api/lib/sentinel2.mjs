/**
 * Sentinel-2 (Copernicus Data Space) proxy helper.
 *
 * Ported from the local dev server (server.mjs). The Lambda holds the OAuth client
 * credentials as env vars and fetches a rendered PNG from the CDSE Process API, so the
 * browser never sees the secret and is not blocked by CDSE CORS.
 *
 * Env: COPERNICUS_CLIENT_ID, COPERNICUS_CLIENT_SECRET
 *      (optional overrides) COPERNICUS_TOKEN_URL, COPERNICUS_PROCESS_URL
 */

const TOKEN_URL =
  process.env.COPERNICUS_TOKEN_URL ||
  'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token';
const PROCESS_URL = process.env.COPERNICUS_PROCESS_URL || 'https://sh.dataspace.copernicus.eu/process/v1';

let tokenCache = null;

function clampNumber(value, min, max, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

async function getAccessToken() {
  const now = Math.floor(Date.now() / 1000);
  if (tokenCache && tokenCache.expiresAt - 60 > now) {
    return tokenCache.accessToken;
  }

  const clientId = process.env.COPERNICUS_CLIENT_ID;
  const clientSecret = process.env.COPERNICUS_CLIENT_SECRET;
  if (!clientId || !clientSecret) {
    throw new Error('Missing COPERNICUS_CLIENT_ID or COPERNICUS_CLIENT_SECRET on the function.');
  }

  const body = new URLSearchParams({
    grant_type: 'client_credentials',
    client_id: clientId,
    client_secret: clientSecret,
  });

  const response = await fetch(TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Copernicus token request failed: ${text}`);
  }

  const token = JSON.parse(text);
  tokenCache = {
    accessToken: token.access_token,
    expiresAt: now + Number(token.expires_in || 3600),
  };
  return tokenCache.accessToken;
}

function buildEvalscript(mode) {
  if (mode === 'natural_color') {
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

  // Default: binary cloud mask (CLM/SCL → white on dark).
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

function shiftIsoDate(date, days) {
  const d = new Date(`${date}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function buildProcessRequest({ bbox, date, mode, width, height, lookbackDays = 12, lookaheadDays = 3 }) {
  // Sentinel-2 revisits every ~5 days, so a single day usually has no acquisition.
  // Query a window around the requested date and take the most recent scene in it.
  const from = `${shiftIsoDate(date, -lookbackDays)}T00:00:00Z`;
  const to = `${shiftIsoDate(date, lookaheadDays)}T23:59:59Z`;
  return {
    input: {
      bounds: { bbox },
      data: [
        {
          type: 'sentinel-2-l2a',
          dataFilter: {
            timeRange: { from, to },
            mosaickingOrder: 'mostRecent',
          },
          processing: { upsampling: 'NEAREST', downsampling: 'NEAREST' },
        },
      ],
    },
    output: {
      width: clampNumber(width, 32, 2048, 512),
      height: clampNumber(height, 32, 2048, 512),
      responses: [{ identifier: 'default', format: { type: 'image/png' } }],
    },
    evalscript: buildEvalscript(mode),
  };
}

/**
 * Fetch a rendered Sentinel-2 PNG for the given bbox + date and return it as a data URL.
 * @returns {Promise<string>} data:image/png;base64,...
 */
export async function fetchSentinel2Png({ bbox, date, mode = 'cloud_mask', width, height }) {
  if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every((n) => Number.isFinite(Number(n)))) {
    throw new Error('bbox must be [minLon, minLat, maxLon, maxLat].');
  }
  if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(String(date))) {
    throw new Error('date must be YYYY-MM-DD.');
  }

  const accessToken = await getAccessToken();
  const requestBody = buildProcessRequest({ bbox: bbox.map(Number), date, mode, width, height });

  const response = await fetch(PROCESS_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
      Accept: 'image/png',
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`Sentinel-2 request failed (${response.status}): ${detail.slice(0, 400)}`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  if (!buffer.length) {
    throw new Error('Sentinel-2 returned an empty image (no scene for this date/area).');
  }
  return `data:image/png;base64,${buffer.toString('base64')}`;
}
