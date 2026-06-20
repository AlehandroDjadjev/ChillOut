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
  statisticsUrl: "/api/statistics",
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
  view: document.querySelector("#view"),
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
  syncViewAvailability();

  elements.source.addEventListener("change", () => {
    populateFieldOptions(elements.source.value);
    items = [];
    renderItems();
    updateActionLabel();
  });

  elements.field.addEventListener("change", () => {
    syncViewAvailability();
    items = [];
    renderItems();
    updateActionLabel();
  });

  elements.view.addEventListener("change", () => {
    syncViewAvailability();
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
  syncViewAvailability();
}

function syncViewAvailability() {
  const canChart = supportsStatistics(elements.source.value, elements.field.value);
  const chartOption = elements.view.querySelector('option[value="chart"]');

  if (chartOption) {
    chartOption.disabled = !canChart;
  }

  if (!canChart && elements.view.value === "chart") {
    elements.view.value = "scenes";
  }
}

function updateActionLabel() {
  const sourceLabel = SOURCE_OPTIONS[elements.source.value]?.label || "selected source";
  const fieldLabel =
    SOURCE_OPTIONS[elements.source.value]?.fields.find(
      (entry) => entry.value === elements.field.value
      )?.label || "field";
  elements.loadAll.textContent =
    elements.view.value === "chart"
      ? `Load latest ${latestItemCount} days of ${sourceLabel} ${fieldLabel.toLowerCase()}`
      : `Load latest ${latestItemCount} ${sourceLabel} ${fieldLabel.toLowerCase()} samples`;
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

  if (elements.view.value === "chart") {
    renderChart();
    return;
  }

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
  if (elements.view.value === "chart") {
    setStatus(`Finding the latest ${latestItemCount} days of ${SOURCE_OPTIONS[source].label} ${getFieldLabel(source, field).toLowerCase()}...`);
  } else {
    setStatus(`Finding the latest ${latestItemCount} ${SOURCE_OPTIONS[source].label} samples...`);
  }

  try {
    if (elements.view.value === "chart") {
      items = await fetchStatistics(source, field);
      renderItems();

      if (items.length === 0) {
        setStatus(`No recent ${SOURCE_OPTIONS[source].label} measurements were found.`, true);
        return;
      }

      setStatus(`Rendering ${items.length} ${SOURCE_OPTIONS[source].label} measurements...`);
      setStatus(
        `Done. Showing the latest ${items.length} days of ${SOURCE_OPTIONS[source].label} ${getFieldLabel(
          source,
          field
        ).toLowerCase()}.`
      );
      return;
    }

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

async function fetchStatistics(source, field) {
  const response = await fetch(config.statisticsUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source,
      field,
      days: latestItemCount,
    }),
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  const payload = await response.json();
  return payload.points || [];
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

function renderChart() {
  const source = elements.source.value;
  const field = elements.field.value;
  const sourceLabel = SOURCE_OPTIONS[source]?.label || "Selected source";
  const fieldLabel = getFieldLabel(source, field);

  if (items.length === 0) {
    const emptyCard = document.createElement("article");
    emptyCard.className = "moment-card moment-card--chart";
    emptyCard.innerHTML = `
      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <strong>${escapeHtml(sourceLabel)} ${escapeHtml(fieldLabel)}</strong>
            <span>No chartable values yet.</span>
          </div>
        </div>
        <div class="chart-figure">
          <p class="placeholder">Choose a source and field that expose numeric Copernicus bands.</p>
        </div>
      </div>
    `;
    elements.moments.appendChild(emptyCard);
    return;
  }

  const values = items
    .map((point) => (typeof point.mean === "number" && Number.isFinite(point.mean) ? point.mean : null))
    .filter((value) => value !== null);

  const numericValues = values.length > 0 ? values : [0];
  const minValue = Math.min(...numericValues);
  const maxValue = Math.max(...numericValues);
  const span = maxValue - minValue || 1;
  const width = 1100;
  const height = 360;
  const padding = { top: 28, right: 20, bottom: 74, left: 40 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const barWidth = innerWidth / items.length;

  const bars = items
    .map((point, index) => {
      const value = typeof point.mean === "number" && Number.isFinite(point.mean) ? point.mean : null;
      const ratio = value === null ? 0 : (value - minValue) / span;
      const barHeight = value === null ? 6 : Math.max(6, ratio * innerHeight);
      const x = padding.left + index * barWidth + 10;
      const y = padding.top + innerHeight - barHeight;
      const barLabel = escapeHtml(point.date || `Day ${index + 1}`);
      const displayValue = value === null ? "n/a" : formatChartValue(value);
      const fill = value === null ? "rgba(15, 122, 138, 0.36)" : "#0f7a8a";
      return `
        <g>
          <rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${Math.max(18, barWidth - 20).toFixed(2)}" height="${barHeight.toFixed(2)}" rx="6" fill="${fill}" />
          <text x="${(x + (barWidth - 20) / 2).toFixed(2)}" y="${Math.max(24, y - 10).toFixed(2)}" text-anchor="middle" font-size="12" fill="var(--ink)" font-weight="700">${escapeHtml(displayValue)}</text>
          <text x="${(x + (barWidth - 20) / 2).toFixed(2)}" y="${(height - 28).toFixed(2)}" text-anchor="middle" font-size="12" fill="var(--muted)">${barLabel}</text>
        </g>
      `;
    })
    .join("");

  const points = items
    .map((point) => {
      const value = typeof point.mean === "number" && Number.isFinite(point.mean) ? point.mean : null;
      return `
        <div class="chart-legend-item">
          <span>${escapeHtml(point.date || "Day")}</span>
          <strong>${value === null ? "n/a" : escapeHtml(formatChartValue(value))}</strong>
        </div>
      `;
    })
    .join("");

  const unitLabel = payloadUnitLabel(elements.view.value, source, field);

  const card = document.createElement("article");
  card.className = "moment-card moment-card--chart";
  card.innerHTML = `
    <div class="chart-card">
      <div class="chart-header">
        <div class="chart-title">
          <strong>${escapeHtml(sourceLabel)} ${escapeHtml(fieldLabel)}</strong>
          <span>${escapeHtml(unitLabel)} over the latest ${items.length} days</span>
        </div>
        <div class="chart-title">
          <strong>${escapeHtml(formatChartValue(minValue))} to ${escapeHtml(formatChartValue(maxValue))}</strong>
          <span>Daily mean values from Copernicus statistics</span>
        </div>
      </div>
      <svg class="chart-figure" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(
    `${sourceLabel} ${fieldLabel} chart`
  )}">
        <rect x="0" y="0" width="${width}" height="${height}" rx="8" fill="transparent"></rect>
        <line x1="${padding.left}" y1="${padding.top + innerHeight}" x2="${width - padding.right}" y2="${padding.top + innerHeight}" stroke="rgba(23,32,38,0.22)" />
        ${bars}
      </svg>
      <div class="chart-legend">
        ${points}
      </div>
    </div>
  `;
  elements.moments.appendChild(card);
}

function formatChartValue(value) {
  const magnitude = Math.abs(value);

  if (!Number.isFinite(value)) {
    return "n/a";
  }

  if (magnitude >= 1000) {
    return value.toFixed(0);
  }

  if (magnitude >= 100) {
    return value.toFixed(1);
  }

  if (magnitude >= 1) {
    return value.toFixed(2);
  }

  if (magnitude >= 0.01) {
    return value.toFixed(3);
  }

  return value.toExponential(2);
}

function payloadUnitLabel(view, source, field) {
  if (view !== "chart") {
    return "";
  }

  if (source === "era5") {
    return ERA5_FIELDS[field]?.unit || "units";
  }

  if (source === "sentinel3olci") {
    return (
      {
        cloud_pressure_proxy: "reflectance",
        humidity: "%",
        sea_level_pressure: "hPa",
        water_vapour: "kg/m^2",
      }[field] || "units"
    );
  }

  if (source === "sentinel5p") {
    return (
      {
        cloud_fraction: "fraction",
        cloud_optical_thickness: "a.u.",
        cloud_top_height: "m",
        cloud_top_pressure: "Pa",
        no2: "mol/m^2",
        o3: "mol/m^2",
        co: "mol/m^2",
        ch4: "ppb",
        so2: "mol/m^2",
        aerosol_index: "index",
      }[field] || "units"
    );
  }

  if (source === "sentinel2") {
    return (
      {
        cloud_probability: "fraction",
        cloud_mask: "fraction",
      }[field] || "units"
    );
  }

  return "units";
}

function supportsStatistics(source, field) {
  if (source === "era5") {
    return true;
  }

  if (source === "sentinel3olci") {
    return true;
  }

  if (source === "sentinel5p") {
    return true;
  }

  if (source === "sentinel2") {
    return field === "cloud_probability" || field === "cloud_mask";
  }

  return false;
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
