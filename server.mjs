import { createServer } from "node:http";
import { stat } from "node:fs/promises";
import { createReadStream, existsSync, readFileSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";

const root = process.cwd();
const port = Number(process.env.PORT || 5173);
const maxBodyBytes = 2_000_000;

const defaults = {
  tokenUrl:
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
  processUrl: "https://sh.dataspace.copernicus.eu/process/v1",
  catalogUrl: "https://sh.dataspace.copernicus.eu/catalog/v1/search",
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

    if (request.method === "POST" && url.pathname === "/api/process") {
      await handleProcess(request, response);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/latest-scenes") {
      await handleLatestScenes(request, response);
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
  console.log(`Stara Zagora Sentinel viewer running at http://localhost:${port}`);
});

async function handleProcess(request, response) {
  if (!process.env.COPERNICUS_CLIENT_ID || !process.env.COPERNICUS_CLIENT_SECRET) {
    sendJson(response, 500, {
      message:
        "Missing COPERNICUS_CLIENT_ID or COPERNICUS_CLIENT_SECRET. Copy .env.example to .env and fill in your Copernicus OAuth credentials.",
    });
    return;
  }

  const requestBody = await readJsonBody(request);
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

async function handleLatestScenes(request, response) {
  if (!process.env.COPERNICUS_CLIENT_ID || !process.env.COPERNICUS_CLIENT_SECRET) {
    sendJson(response, 500, {
      message:
        "Missing COPERNICUS_CLIENT_ID or COPERNICUS_CLIENT_SECRET. Copy .env.example to .env and fill in your Copernicus OAuth credentials.",
    });
    return;
  }

  const requestBody = await readJsonBody(request);
  const limit = clampNumber(requestBody.limit, 1, 20, 7);
  const maxCloudCoverage = clampNumber(requestBody.maxCloudCoverage, 0, 100, 100);
  const scenes = await findLatestScenes({ limit, maxCloudCoverage });

  sendJson(response, 200, { scenes });
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

async function findLatestScenes({ limit, maxCloudCoverage }) {
  const accessToken = await getAccessToken();
  const end = new Date();
  const byDate = new Map();
  const searchWindowsInMonths = [3, 6, 12, 24, 48];

  for (const months of searchWindowsInMonths) {
    const start = new Date(end);
    start.setUTCMonth(start.getUTCMonth() - months);

    const catalog = await searchCatalog(accessToken, {
      from: start.toISOString(),
      to: end.toISOString(),
    });

    for (const feature of catalog.features || []) {
      const datetime = feature.properties?.datetime;
      const cloudCover = Number(feature.properties?.["eo:cloud_cover"]);

      if (!datetime || !Number.isFinite(cloudCover) || cloudCover > maxCloudCoverage) {
        continue;
      }

      const date = datetime.slice(0, 10);
      const existing = byDate.get(date);

      if (!existing || cloudCover < existing.cloudCover) {
        byDate.set(date, {
          id: feature.id,
          date,
          datetime,
          cloudCover,
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

async function searchCatalog(accessToken, { from, to }) {
  const catalogBody = {
    bbox: getPublicConfig().bbox,
    datetime: `${from}/${to}`,
    collections: ["sentinel-2-l2a"],
    limit: 100,
    fields: {
      include: ["id", "properties.datetime", "properties.eo:cloud_cover"],
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

  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
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

function clampNumber(value, min, max, fallback) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return fallback;
  }

  return Math.min(max, Math.max(min, number));
}

function parseBbox(value) {
  if (!value) {
    return null;
  }

  const bbox = value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((part) => Number.isFinite(part));

  return bbox.length === 4 ? bbox : null;
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
