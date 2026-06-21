(function () {
  "use strict";

  var C = window.ChillOutContracts;
  var sample = null;
  var originalMaskDataUrl = "";
  var uploadedMaskDataUrl = "";

  var el = {
    form: document.getElementById("plannerForm"),
    place: document.getElementById("placeInput"),
    preset: document.getElementById("presetSelect"),
    date: document.getElementById("dateInput"),
    target: document.getElementById("targetInput"),
    endpoint: document.getElementById("endpointInput"),
    image: document.getElementById("imageInput"),
    sampleBtn: document.getElementById("sampleBtn"),
    runBtn: document.getElementById("runBtn"),
    status: document.getElementById("status"),
    table: document.querySelector("#featureTable tbody"),
    originalCanvas: document.getElementById("originalCanvas"),
    originalCaption: document.getElementById("originalCaption"),
    generatedImage: document.getElementById("generatedImage"),
    generatedCaption: document.getElementById("generatedCaption"),
    metrics: document.getElementById("metrics"),
    payload: document.getElementById("payloadJson"),
    modePill: document.getElementById("modePill")
  };

  function init() {
    el.date.value = C.todayIso();
    el.endpoint.value = C.getConfig().plannerEndpoint;
    el.preset.innerHTML = C.CITY_PRESETS.map(function (city, index) {
      return "<option value=\"" + index + "\">" + C.escapeHtml(city.name + ", " + city.country) + "</option>";
    }).join("");
    el.preset.addEventListener("change", function () {
      var city = C.CITY_PRESETS[Number(el.preset.value)];
      el.place.value = city.name + ", " + city.country;
    });
    el.endpoint.addEventListener("change", function () {
      C.setEndpoint("planner", el.endpoint.value.trim());
      setStatus(el.endpoint.value.trim() ? "Endpoint saved for future RL calls." : "Endpoint cleared. Preview mode is active.", "ok");
    });
    el.image.addEventListener("change", handleUpload);
    el.sampleBtn.addEventListener("click", function () { buildSample(false); });
    el.form.addEventListener("submit", function (event) {
      event.preventDefault();
      runPlanner();
    });
    drawBlank();
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

  async function buildSample(silent) {
    setBusy(true);
    if (!silent) setStatus("Fetching weather, building feature planes, and preparing the mask...");
    try {
      var place = await resolvePlace();
      sample = await C.fetchWeatherSample(place, el.date.value);
      if (!uploadedMaskDataUrl) {
        originalMaskDataUrl = C.drawMask(el.originalCanvas, sample);
      } else {
        originalMaskDataUrl = await C.loadImageToCanvas(uploadedMaskDataUrl, el.originalCanvas);
      }
      el.target.value = Number.isFinite(Number(el.target.value))
        ? el.target.value
        : C.round(sample.observed_temperature_c - 2, 1);
      el.originalCaption.textContent = sample.sample_id + (uploadedMaskDataUrl ? " - uploaded mask" : " - generated mask");
      renderFeatureTable();
      renderPayload();
      setStatus("Planner input ready: " + sample.sample_id, "ok");
    } catch (error) {
      setStatus(error.message || String(error), "error");
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload() {
    var file = el.image.files && el.image.files[0];
    if (!file) {
      uploadedMaskDataUrl = "";
      return;
    }
    try {
      uploadedMaskDataUrl = await C.fileToDataUrl(file);
      if (sample) {
        originalMaskDataUrl = await C.loadImageToCanvas(uploadedMaskDataUrl, el.originalCanvas);
        el.originalCaption.textContent = sample.sample_id + " - uploaded mask";
        renderPayload();
      }
      setStatus("Image loaded. It will be resized to 256 x 256 for the planner contract.", "ok");
    } catch (error) {
      setStatus(error.message || String(error), "error");
    }
  }

  async function runPlanner() {
    if (!sample) {
      await buildSample(true);
      if (!sample) return;
    }
    setBusy(true);
    setStatus("Sending image and feature vector through the planner client...");
    try {
      C.setEndpoint("planner", el.endpoint.value.trim());
      var target = Number(el.target.value);
      if (!Number.isFinite(target)) throw new Error("Enter a valid target temperature.");
      var result = await C.runPlannerModel(sample, originalMaskDataUrl, target);
      renderResult(result);
      renderPayload(result);
      setStatus(result.mode === "endpoint" ? "Hosted planner response received." : "Preview planner complete. Add an endpoint when the policy is hosted.", "ok");
    } catch (error) {
      setStatus(error.message || String(error), "error");
    } finally {
      setBusy(false);
    }
  }

  function renderResult(result) {
    var output = result.output || result;
    el.modePill.textContent = result.mode || "result";
    if (output.generated_mask_data_url) {
      el.generatedImage.src = output.generated_mask_data_url;
      el.generatedCaption.textContent = output.generated_mask_path || "Generated mask";
    }
    var actions = output.actions || [];
    var expected = Number(output.expected_temperature_c);
    var reward = Number(output.reward_proxy);
    el.metrics.innerHTML = [
      metric(String(actions.length), "Actions emitted"),
      metric(Number.isFinite(expected) ? expected.toFixed(2) + " C" : "see JSON", "Expected temperature"),
      metric(Number.isFinite(reward) ? reward.toFixed(3) : (result.mode === "endpoint" ? "endpoint" : "preview"), "Reward proxy")
    ].join("");
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

  function renderPayload(result) {
    var payload = sample ? {
      sample_id: sample.sample_id,
      raw_features: sample.inputs,
      feature_vector: sample.feature_vector,
      target_temperature_c: Number(el.target.value),
      original_mask_data_url: originalMaskDataUrl ? "[data-url omitted from preview block]" : null,
      latest_result: result ? compactResult(result) : null
    } : {};
    el.payload.textContent = JSON.stringify(payload, null, 2);
  }

  function compactResult(result) {
    var clone = JSON.parse(JSON.stringify(result));
    if (clone.input) clone.input.original_mask_data_url = "[data-url omitted]";
    if (clone.output) {
      if (clone.output.original_mask_data_url) clone.output.original_mask_data_url = "[data-url omitted]";
      if (clone.output.generated_mask_data_url) clone.output.generated_mask_data_url = "[data-url omitted]";
    }
    if (clone.generated_mask_data_url) clone.generated_mask_data_url = "[data-url omitted]";
    return clone;
  }

  function metric(value, label) {
    return "<div class=\"lab-metric\"><b>" + C.escapeHtml(value) + "</b><span>" + C.escapeHtml(label) + "</span></div>";
  }

  function drawBlank() {
    var ctx = el.originalCanvas.getContext("2d");
    ctx.fillStyle = "#02090b";
    ctx.fillRect(0, 0, el.originalCanvas.width, el.originalCanvas.height);
    el.generatedImage.src = el.originalCanvas.toDataURL("image/png");
  }

  function setBusy(isBusy) {
    el.sampleBtn.disabled = isBusy;
    el.runBtn.disabled = isBusy;
  }

  function setStatus(message, kind) {
    el.status.textContent = message || "";
    el.status.classList.toggle("is-error", kind === "error");
    el.status.classList.toggle("is-ok", kind === "ok");
  }

  init();
})();
