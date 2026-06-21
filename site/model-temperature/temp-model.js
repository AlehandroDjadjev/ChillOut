(function () {
  "use strict";

  var C = window.ChillOutContracts;
  var sample = null;
  var maskDataUrl = "";

  var el = {
    form: document.getElementById("sampleForm"),
    place: document.getElementById("placeInput"),
    preset: document.getElementById("presetSelect"),
    date: document.getElementById("dateInput"),
    endpoint: document.getElementById("endpointInput"),
    fetchBtn: document.getElementById("fetchBtn"),
    predictBtn: document.getElementById("predictBtn"),
    status: document.getElementById("status"),
    table: document.querySelector("#featureTable tbody"),
    canvas: document.getElementById("maskCanvas"),
    caption: document.getElementById("maskCaption"),
    payload: document.getElementById("payloadJson"),
    metrics: document.getElementById("metrics"),
    modePill: document.getElementById("modePill")
  };

  function init() {
    el.date.value = C.todayIso();
    el.endpoint.value = C.getConfig().temperatureEndpoint;
    el.preset.innerHTML = C.CITY_PRESETS.map(function (city, index) {
      return "<option value=\"" + index + "\">" + C.escapeHtml(city.name + ", " + city.country) + "</option>";
    }).join("");
    el.preset.addEventListener("change", function () {
      var city = C.CITY_PRESETS[Number(el.preset.value)];
      el.place.value = city.name + ", " + city.country;
    });
    el.endpoint.addEventListener("change", function () {
      C.setEndpoint("temperature", el.endpoint.value.trim());
      setStatus(el.endpoint.value.trim() ? "Endpoint saved for future model calls." : "Endpoint cleared. Preview mode is active.", "ok");
    });
    el.fetchBtn.addEventListener("click", function () { loadSample(false); });
    el.form.addEventListener("submit", function (event) {
      event.preventDefault();
      runPrediction();
    });
    renderFeatureTable();
    renderPayload();
  }

  async function resolvePlace() {
    var preset = C.CITY_PRESETS.find(function (city) {
      return (city.name + ", " + city.country).toLowerCase() === el.place.value.trim().toLowerCase();
    });
    if (preset) return preset;
    return C.geocodePlace(el.place.value);
  }

  async function loadSample(silent) {
    setBusy(true);
    if (!silent) setStatus("Fetching weather and building model features...");
    try {
      var place = await resolvePlace();
      sample = await C.fetchWeatherSample(place, el.date.value);
      maskDataUrl = C.drawMask(el.canvas, sample);
      el.caption.textContent = sample.city + ", " + sample.country + " - " + sample.date;
      renderFeatureTable();
      renderPayload();
      setStatus("Sample ready: " + sample.sample_id, "ok");
    } catch (error) {
      setStatus(error.message || String(error), "error");
    } finally {
      setBusy(false);
    }
  }

  async function runPrediction() {
    if (!sample) {
      await loadSample(true);
      if (!sample) return;
    }
    setBusy(true);
    setStatus("Sending payload through the model client...");
    try {
      C.setEndpoint("temperature", el.endpoint.value.trim());
      var result = await C.runTemperatureModel(sample, maskDataUrl);
      renderResult(result);
      renderPayload(result);
      setStatus(result.mode === "endpoint" ? "Hosted model response received." : "Preview inference complete. Add an endpoint when the model is hosted.", "ok");
    } catch (error) {
      setStatus(error.message || String(error), "error");
    } finally {
      setBusy(false);
    }
  }

  function renderFeatureTable() {
    if (!sample) {
      el.table.innerHTML = C.FEATURE_NAMES.map(function (name) {
        return "<tr><th>" + name + "</th><td>-</td></tr>";
      }).join("");
      return;
    }
    el.table.innerHTML = C.FEATURE_NAMES.map(function (name, index) {
      return "<tr><th>" + C.escapeHtml(name) + "</th><td>" +
        C.escapeHtml(sample.feature_vector[index]) + "</td></tr>";
    }).join("");
  }

  function renderResult(result) {
    el.modePill.textContent = result.mode || "result";
    var prediction = Number(result.prediction_c);
    var baseline = Number(result.baseline_observed_c || (sample && sample.observed_temperature_c));
    var delta = Number.isFinite(prediction) && Number.isFinite(baseline)
      ? C.round(prediction - baseline, 2)
      : null;
    var cloud = Number(result.cloud_cooling_component_c);
    var confidence = Number(result.confidence);
    el.metrics.innerHTML = [
      metric(Number.isFinite(prediction) ? prediction.toFixed(2) + " C" : "see JSON", "Predicted temperature"),
      metric(delta !== null ? (delta > 0 ? "+" : "") + delta.toFixed(2) + " C" : (Number.isFinite(cloud) ? cloud.toFixed(2) + " C" : "-"), "Delta / cloud contribution"),
      metric(Number.isFinite(confidence) ? Math.round(confidence * 100) + "%" : (result.mode === "endpoint" ? "endpoint" : "preview"), "Confidence / source")
    ].join("");
  }

  function metric(value, label) {
    return "<div class=\"lab-metric\"><b>" + C.escapeHtml(value) + "</b><span>" + C.escapeHtml(label) + "</span></div>";
  }

  function renderPayload(result) {
    var payload = sample ? {
      sample: sample,
      cloud_mask_data_url: maskDataUrl ? "[data-url omitted from preview block]" : null,
      feature_names: C.FEATURE_NAMES,
      latest_result: result || null
    } : {};
    el.payload.textContent = JSON.stringify(payload, null, 2);
  }

  function setBusy(isBusy) {
    el.fetchBtn.disabled = isBusy;
    el.predictBtn.disabled = isBusy;
  }

  function setStatus(message, kind) {
    el.status.textContent = message || "";
    el.status.classList.toggle("is-error", kind === "error");
    el.status.classList.toggle("is-ok", kind === "ok");
  }

  init();
})();
