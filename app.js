const DEFAULT_CONFIG = {
  processUrl: "/api/process",
  latestScenesUrl: "/api/latest-scenes",
  bbox: [25.43, 42.31, 25.86, 42.57],
  width: 768,
  height: 480,
};

const latestImageCount = 7;
let CONFIG = {
  ...DEFAULT_CONFIG,
};

let moments = [];

const elements = {
  maxCloud: document.querySelector("#max-cloud"),
  renderMode: document.querySelector("#render-mode"),
  form: document.querySelector("#settings-form"),
  loadAll: document.querySelector("#load-all"),
  globalStatus: document.querySelector("#global-status"),
  moments: document.querySelector("#moments"),
  bboxLabel: document.querySelector("#bbox-label"),
};

async function init() {
  CONFIG = {
    ...CONFIG,
    ...(await loadPublicConfig()),
  };
  elements.bboxLabel.textContent = CONFIG.bbox.join(", ");

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    loadAllImages();
  });

  renderMoments();
}

async function loadPublicConfig() {
  try {
    const response = await fetch("/api/config");

    if (!response.ok) {
      return {};
    }

    return response.json();
  } catch {
    return {};
  }
}

function renderMoments() {
  elements.moments.innerHTML = "";

  if (moments.length === 0) {
    const empty = document.createElement("article");
    empty.className = "moment-card";
    empty.innerHTML = `
      <div class="image-stage">
        <p class="placeholder">The latest ${latestImageCount} Sentinel-2 images will appear here.</p>
      </div>
      <div class="card-footer">
        <span>Waiting</span>
        <strong>Latest scenes</strong>
      </div>
    `;
    elements.moments.appendChild(empty);
    return;
  }

  moments.forEach((moment, index) => {
    const card = document.createElement("article");
    card.className = "moment-card";
    card.dataset.index = String(index);
    card.innerHTML = `
      <div class="moment-controls">
        <label>
          Acquisition date
          <input data-field="date" type="date" value="${moment.date}" readonly />
        </label>
        <label>
          Tile cloud %
          <input data-field="cloudCover" type="text" value="${formatCloudCover(moment.cloudCover)}" readonly />
        </label>
      </div>
      <div class="image-stage" data-role="stage">
        <p class="placeholder">Imagery will appear here.</p>
      </div>
      <div class="card-footer">
        <span data-role="card-status">Waiting</span>
        <strong>Range ${index + 1}</strong>
      </div>
    `;

    elements.moments.appendChild(card);
  });
}

async function loadAllImages() {
  clearStatus();

  elements.loadAll.disabled = true;
  setStatus(`Finding the latest ${latestImageCount} Sentinel-2 images...`);

  try {
    moments = await fetchLatestMoments();
    renderMoments();

    if (moments.length === 0) {
      setStatus("No Sentinel-2 images were found for the current bbox/cloud settings.", true);
      return;
    }

    setStatus(`Found ${moments.length} images. Loading Sentinel-2 pixels...`);

    for (const [index, moment] of moments.entries()) {
      await loadMomentImage(index, moment);
    }

    setStatus(`Done. Showing the latest ${moments.length} Sentinel-2 L2A images found.`);
  } catch (error) {
    setStatus(getFriendlyError(error), true);
  } finally {
    elements.loadAll.disabled = false;
  }
}

async function loadMomentImage(index, moment) {
  const card = document.querySelector(`.moment-card[data-index="${index}"]`);
  const stage = card.querySelector('[data-role="stage"]');
  const status = card.querySelector('[data-role="card-status"]');

  status.textContent = "Loading...";
  stage.innerHTML = '<p class="placeholder">Fetching satellite pixels...</p>';

  try {
    const response = await fetch(CONFIG.processUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(buildProcessRequest(moment)),
    });

    if (!response.ok) {
      throw new Error(await readApiError(response));
    }

    const imageBlob = await response.blob();
    const imageUrl = URL.createObjectURL(imageBlob);
    stage.innerHTML = "";

    const image = document.createElement("img");
    image.alt = `Sentinel-2 cloud imagery around Stara Zagora acquired on ${moment.date}`;
    image.src = imageUrl;
    stage.appendChild(image);

    status.textContent = moment.datetime
      ? new Date(moment.datetime).toLocaleString()
      : moment.date;
  } catch (error) {
    status.textContent = "Failed";
    stage.innerHTML = `<p class="placeholder">${escapeHtml(getFriendlyError(error))}</p>`;
  }
}

async function fetchLatestMoments() {
  const response = await fetch(CONFIG.latestScenesUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      limit: latestImageCount,
      maxCloudCoverage: Number(elements.maxCloud.value || 100),
    }),
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  const payload = await response.json();
  return payload.scenes || [];
}

function buildProcessRequest(moment) {
  const date = moment.date;

  return {
    input: {
      bounds: {
        bbox: CONFIG.bbox,
      },
      data: [
        {
          type: "sentinel-2-l2a",
          dataFilter: {
            timeRange: {
              from: `${date}T00:00:00Z`,
              to: `${date}T23:59:59Z`,
            },
            maxCloudCoverage: Number(elements.maxCloud.value || 100),
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
      width: CONFIG.width,
      height: CONFIG.height,
      responses: [
        {
          identifier: "default",
          format: {
            type: "image/png",
          },
        },
      ],
    },
    evalscript: getEvalscript(elements.renderMode.value),
  };
}

function formatCloudCover(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}%` : "unknown";
}

function getEvalscript(mode) {
  if (mode === "mask") {
    return `//VERSION=3
function setup() {
  return {
    input: ["CLM", "CLP", "SCL", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function isCloud(sample) {
  return sample.CLM === 1 || [8, 9, 10].includes(sample.SCL) || sample.CLP > 107;
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [0, 0, 0, 0];
  if (isCloud(sample)) return [1, 1, 1, 1];
  return [0.02, 0.16, 0.22, 0.92];
}`;
  }

  if (mode === "probability") {
    return `//VERSION=3
function setup() {
  return {
    input: ["CLM", "CLP", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [0, 0, 0, 0];
  var probability = sample.CLM === 255 ? 0 : sample.CLP / 255;
  return [probability, probability * 0.82 + 0.08, 1 - probability * 0.72, 1];
}`;
  }

  return `//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "CLM", "CLP", "SCL", "dataMask"],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function isCloud(sample) {
  return sample.CLM === 1 || [8, 9, 10].includes(sample.SCL) || sample.CLP > 107;
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [0, 0, 0, 0];

  var red = Math.min(1, 2.6 * sample.B04);
  var green = Math.min(1, 2.6 * sample.B03);
  var blue = Math.min(1, 2.6 * sample.B02);

  if (!isCloud(sample)) {
    return [red, green, blue, 1];
  }

  var probability = Math.max(0.58, sample.CLP / 255);
  return [
    Math.min(1, red * 0.2 + probability * 0.86),
    Math.min(1, green * 0.25 + probability * 0.9),
    Math.min(1, blue * 0.35 + probability),
    1
  ];
}`;
}

async function readApiError(response) {
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();

  if (contentType.includes("application/json")) {
    try {
      const json = JSON.parse(text);
      return json.error_description || json.message || JSON.stringify(json);
    } catch {
      return text;
    }
  }

  return text || `${response.status} ${response.statusText}`;
}

function getFriendlyError(error) {
  const message = error instanceof Error ? error.message : String(error);

  if (message.toLowerCase().includes("failed to fetch")) {
    return "The browser could not reach the local proxy. Start it with node server.mjs and open http://localhost:5173.";
  }

  return message;
}

function clearStatus() {
  elements.globalStatus.classList.remove("error");
}

function setStatus(message, isError = false) {
  elements.globalStatus.textContent = message;
  elements.globalStatus.classList.toggle("error", isError);
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => {
    const replacements = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return replacements[character];
  });
}

init();
