const SOURCE_OPTIONS = {
  sentinel2: {
    label: "Sentinel-2",
    fields: [
      { value: "natural_color", label: "Image" },
      { value: "cloud_mask", label: "Mask" },
      { value: "cloud_probability", label: "Cloud probability" },
    ],
  },
  sentinel3olci: {
    label: "Sentinel-3 OLCI",
    fields: [
      { value: "cloud_pressure_proxy", label: "Cloud pressure proxy" },
      { value: "humidity", label: "Humidity" },
      { value: "sea_level_pressure", label: "Sea level pressure" },
      { value: "water_vapour", label: "Water vapour" },
    ],
  },
  era5: {
    label: "ERA5",
    fields: [
      { value: "shortwave_radiation", label: "Radiation" },
      { value: "wind_speed_10m", label: "Wind speed" },
      { value: "wind_direction_10m", label: "Wind direction" },
      { value: "relative_humidity_2m", label: "Humidity" },
      { value: "cloud_cover", label: "Cloud cover" },
      { value: "cloud_cover_low", label: "Cloud cover low" },
      { value: "cloud_cover_mid", label: "Cloud cover mid" },
      { value: "cloud_cover_high", label: "Cloud cover high" },
    ],
  },
  sentinel5p: {
    label: "Sentinel-5P",
    fields: [
      { value: "cloud_fraction", label: "Cloud fraction" },
      { value: "cloud_optical_thickness", label: "Cloud thickness" },
      { value: "cloud_top_height", label: "Cloud top height" },
      { value: "cloud_top_pressure", label: "Cloud top pressure" },
      { value: "no2", label: "NO2" },
      { value: "o3", label: "Ozone" },
      { value: "co", label: "CO" },
      { value: "ch4", label: "Methane" },
      { value: "so2", label: "SO2" },
      { value: "aerosol_index", label: "Aerosol index" },
    ],
  },
};

const DEFAULT_CONFIG = {
  processUrl: "/api/process",
  latestScenesUrl: "/api/latest-scenes",
  bbox: [25.43, 42.31, 25.86, 42.57],
  width: 768,
  height: 480,
};

const latestItemCount = 7;
let config = {
  ...DEFAULT_CONFIG,
};
let items = [];

const elements = {
  source: document.querySelector("#source"),
  field: document.querySelector("#field"),
  form: document.querySelector("#settings-form"),
  loadAll: document.querySelector("#load-all"),
  globalStatus: document.querySelector("#global-status"),
  moments: document.querySelector("#moments"),
  bboxLabel: document.querySelector("#bbox-label"),
};

async function init() {
  config = {
    ...config,
    ...(await loadPublicConfig()),
  };

  elements.bboxLabel.textContent = config.bbox.join(", ");

  populateSourceOptions();
  populateFieldOptions(elements.source.value);

  elements.source.addEventListener("change", () => {
    populateFieldOptions(elements.source.value);
    items = [];
    renderItems();
    updateActionLabel();
  });

  elements.field.addEventListener("change", () => {
    items = [];
    renderItems();
    updateActionLabel();
  });

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    loadLatestItems();
  });

  updateActionLabel();
  renderItems();
}

function populateSourceOptions() {
  elements.source.innerHTML = Object.entries(SOURCE_OPTIONS)
    .map(
      ([value, meta]) =>
        `<option value="${value}">${escapeHtml(meta.label)}</option>`
    )
    .join("");

  elements.source.value = "sentinel2";
}

function populateFieldOptions(source) {
  const fields = SOURCE_OPTIONS[source]?.fields || [];
  elements.field.innerHTML = fields
    .map(
      (field) => `<option value="${field.value}">${escapeHtml(field.label)}</option>`
    )
    .join("");

  elements.field.value = fields[0]?.value || "";
}

function updateActionLabel() {
  const sourceLabel = SOURCE_OPTIONS[elements.source.value]?.label || "selected source";
  const fieldLabel =
    SOURCE_OPTIONS[elements.source.value]?.fields.find(
      (entry) => entry.value === elements.field.value
    )?.label || "field";
  elements.loadAll.textContent = `Load latest ${latestItemCount} ${sourceLabel} ${fieldLabel.toLowerCase()} samples`;
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

function renderItems() {
  elements.moments.innerHTML = "";

  if (items.length === 0) {
    const emptyCard = document.createElement("article");
    emptyCard.className = "moment-card";
    emptyCard.innerHTML = `
      <div class="image-stage">
        <p class="placeholder">The latest ${latestItemCount} ${escapeHtml(
          SOURCE_OPTIONS[elements.source.value]?.label || "selected"
        )} samples will appear here.</p>
      </div>
      <div class="card-footer">
        <span>Waiting</span>
        <strong>${escapeHtml(
          SOURCE_OPTIONS[elements.source.value]?.label || "Selected source"
        )}</strong>
      </div>
    `;
    elements.moments.appendChild(emptyCard);
    return;
  }

  items.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "moment-card";
    card.dataset.index = String(index);
    card.innerHTML = `
      <div class="moment-controls">
        <label>
          Date
          <input data-field="date" type="text" value="${escapeHtml(item.date || "")}" readonly />
        </label>
        <label>
          Info
          <input data-field="detail" type="text" value="${escapeHtml(item.detail || "")}" readonly />
        </label>
      </div>
      <div class="image-stage" data-role="stage">
        <p class="placeholder">Loading sample...</p>
      </div>
      <div class="card-footer">
        <span data-role="card-status">Waiting</span>
        <strong>${escapeHtml(item.label || item.date || `Sample ${index + 1}`)}</strong>
      </div>
    `;
    elements.moments.appendChild(card);
  });
}

async function loadLatestItems() {
  clearStatus();

  const source = elements.source.value;
  const field = elements.field.value;
  elements.loadAll.disabled = true;
  setStatus(`Finding the latest ${latestItemCount} ${SOURCE_OPTIONS[source].label} samples...`);

  try {
    items = await fetchLatestItems(source);
    renderItems();

    if (items.length === 0) {
      setStatus(`No recent ${SOURCE_OPTIONS[source].label} samples were found.`, true);
      return;
    }

    setStatus(`Rendering ${items.length} ${SOURCE_OPTIONS[source].label} samples...`);

    for (const [index, item] of items.entries()) {
      await loadItem(source, field, index, item);
    }

    setStatus(`Done. Showing the latest ${items.length} ${SOURCE_OPTIONS[source].label} ${getFieldLabel(source, field).toLowerCase()} samples.`);
  } catch (error) {
    setStatus(getFriendlyError(error), true);
  } finally {
    elements.loadAll.disabled = false;
  }
}

async function fetchLatestItems(source) {
  const response = await fetch(config.latestScenesUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source,
      limit: latestItemCount,
    }),
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  const payload = await response.json();
  return payload.scenes || [];
}

async function loadItem(source, field, index, item) {
  const card = document.querySelector(`.moment-card[data-index="${index}"]`);
  const stage = card.querySelector('[data-role="stage"]');
  const status = card.querySelector('[data-role="card-status"]');

  status.textContent = "Loading...";
  stage.innerHTML = '<p class="placeholder">Fetching data...</p>';

  try {
    const response = await fetch(config.processUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        source,
        field,
        item,
        bbox: config.bbox,
        width: config.width,
        height: config.height,
      }),
    });

    if (!response.ok) {
      throw new Error(await readApiError(response));
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    stage.innerHTML = "";

    const image = document.createElement("img");
    image.alt = `${SOURCE_OPTIONS[source].label} ${getFieldLabel(source, field)} for ${item.label || item.date}`;
    image.src = url;
    stage.appendChild(image);

    status.textContent = item.detail || item.label || item.date || "Ready";
  } catch (error) {
    status.textContent = "Failed";
    stage.innerHTML = `<p class="placeholder">${escapeHtml(getFriendlyError(error))}</p>`;
  }
}

function getFieldLabel(source, field) {
  return (
    SOURCE_OPTIONS[source]?.fields.find((entry) => entry.value === field)?.label ||
    field
  );
}

async function readApiError(response) {
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();

  if (contentType.includes("application/json")) {
    try {
      const json = JSON.parse(text);
      return json.message || json.error_description || JSON.stringify(json);
    } catch {
      return text;
    }
  }

  return text || `${response.status} ${response.statusText}`;
}

function getFriendlyError(error) {
  const message = error instanceof Error ? error.message : String(error);

  if (message.toLowerCase().includes("failed to fetch")) {
    return "The local server could not be reached. Make sure `node server.mjs` is running.";
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
  return String(value).replace(/[&<>"']/g, (character) => {
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
