(function () {
  "use strict";

  var C = window.ChillOutContracts;
  var sample = null;
  var originalMaskDataUrl = "";
  var polling = false;

  var el = {
    form: document.getElementById("inferForm"),
    place: document.getElementById("placeInput"),
    preset: document.getElementById("presetSelect"),
    date: document.getElementById("dateInput"),
    target: document.getElementById("targetInput"),
    checkpoint: document.getElementById("checkpointInput"),
    stats: document.getElementById("statsInput"),
    sampleBtn: document.getElementById("sampleBtn"),
    runBtn: document.getElementById("runBtn"),
    status: document.getElementById("status"),
    table: document.querySelector("#featureTable tbody"),
    originalCanvas: document.getElementById("originalCanvas"),
    originalCaption: document.getElementById("originalCaption"),
    generatedImage: document.getElementById("generatedImage"),
    generatedCaption: document.getElementById("generatedCaption"),
    propertyMaps: document.getElementById("propertyMaps"),
    metrics: document.getElementById("metrics"),
    actions: document.getElementById("actionsJson"),
    payload: document.getElementById("payloadJson"),
    modePill: document.getElementById("modePill")
  };

  function init() {
    el.date.value = C.todayIso();
    el.preset.innerHTML = C.CITY_PRESETS.map(function (city, index) {
      return "<option value=\"" + index + "\">" + C.escapeHtml(city.name + ", " + city.country) + "</option>";
    }).join("");
    el.preset.addEventListener("change", function () {
      var city = C.CITY_PRESETS[Number(el.preset.value)];
      el.place.value = city.name + ", " + city.country;
    });
    el.sampleBtn.addEventListener("click", function () { buildSample(false); });
    el.form.addEventListener("submit", function (event) {
      event.preventDefault();
      runInference();
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
    if (!silent) setStatus("Fetching weather and the latest Sentinel-2 cloud mask...");
    try {
      var place = await resolvePlace();
      sample = await C.fetchWeatherSample(place, el.date.value);
      var sentinelUrl = await C.fetchSentinelImage(sample, "cloud_mask");
      originalMaskDataUrl = await C.loadImageToCanvas(sentinelUrl, el.originalCanvas);
      el.originalCaption.textContent = "Sentinel-2 cloud mask - " + sample.date;
      renderFeatureTable();
      renderPayload();
      setStatus("Mask ready for " + sample.sample_id + ". Upload a checkpoint or run the AI fallback.", "ok");
    } catch (error) {
      setStatus("Could not build the mask: " + (error.message || String(error)), "error");
    } finally {
      setBusy(false);
    }
  }

  async function runInference() {
    var checkpointFile = el.checkpoint.files && el.checkpoint.files[0];
    var target = Number(el.target.value);
    if (!Number.isFinite(target)) {
      setStatus("Enter a valid target temperature.", "error");
      return;
    }
    if (!sample || !originalMaskDataUrl) {
      await buildSample(true);
      if (!sample || !originalMaskDataUrl) return;
    }

    setBusy(true);
    el.modePill.textContent = "uploading";
    try {
      var jobUuid = (window.crypto && crypto.randomUUID)
        ? crypto.randomUUID()
        : "job-" + Date.now() + "-" + Math.random().toString(16).slice(2);

      var checkpointKey = null;
      if (checkpointFile) {
        setStatus("Uploading checkpoint to storage...");
        var ckptUp = await C.fetchUploadUrl(jobUuid, "checkpoint.pt", "application/octet-stream");
        await C.uploadFile(ckptUp.uploadUrl, checkpointFile);
        checkpointKey = ckptUp.key;
      } else {
        setStatus("No checkpoint uploaded. Running AI image fallback...");
        el.modePill.textContent = "AI fallback";
      }

      var statsKey = null;
      var statsFile = el.stats.files && el.stats.files[0];
      if (checkpointFile && statsFile) {
        setStatus("Uploading stats.json...");
        var statsUp = await C.fetchUploadUrl(jobUuid, "stats.json", "application/json");
        await C.uploadFile(statsUp.uploadUrl, statsFile);
        statsKey = statsUp.key;
      }

      setStatus("Queueing inference job...");
      el.modePill.textContent = "queued";
      var created = await C.createInference({
        checkpoint_key: checkpointKey,
        stats_key: statsKey,
        mask_data_url: originalMaskDataUrl,
        raw_features: sample.feature_vector,
        target_temperature_c: target,
        place: sample.sample_id,
        date: sample.date
      });
      renderPayload(created);
      if (created.status === "completed" && created.result) {
        renderResult(created.result);
        el.modePill.textContent = created.mode || "fallback";
        setStatus("AI fallback image complete.", "ok");
        return;
      }
      await pollResult(created.id);
    } catch (error) {
      el.modePill.textContent = "error";
      setStatus(error.message || String(error), "error");
    } finally {
      setBusy(false);
    }
  }

  function pollResult(id) {
    polling = true;
    var started = Date.now();
    var TIMEOUT_MS = 5 * 60 * 1000;
    return new Promise(function (resolve) {
      function tick() {
        C.getInference(id).then(function (res) {
          el.modePill.textContent = res.status || "running";
          if (res.status === "completed") {
            polling = false;
            renderResult(res.result);
            setStatus("Inference complete.", "ok");
            resolve();
            return;
          }
          if (res.status === "failed") {
            polling = false;
            el.modePill.textContent = "failed";
            setStatus("Inference failed: " + (res.error || "unknown error"), "error");
            resolve();
            return;
          }
          if (Date.now() - started > TIMEOUT_MS) {
            polling = false;
            setStatus("Timed out waiting for the worker. Is the RL worker daemon running?", "error");
            resolve();
            return;
          }
          setStatus("Worker is " + (res.status || "running") + "... (checkpoint loads + inference run on the VM)");
          setTimeout(tick, 3000);
        }).catch(function (error) {
          polling = false;
          setStatus("Polling error: " + (error.message || String(error)), "error");
          resolve();
        });
      }
      tick();
    });
  }

  function renderResult(result) {
    if (!result) {
      setStatus("Worker returned an empty result.", "error");
      return;
    }
    if (result.generated_mask_data_url) {
      el.generatedImage.src = result.generated_mask_data_url;
      el.generatedCaption.textContent = result.fallback ? "AI fallback overlay" : "Generated mask";
    }
    var maps = result.property_maps || {};
    el.propertyMaps.innerHTML = Object.keys(maps).map(function (name) {
      return "<figure class=\"lab-frame\"><img src=\"" + maps[name] + "\" alt=\"" +
        C.escapeHtml(name) + "\" /><figcaption>" + C.escapeHtml(name) + "</figcaption></figure>";
    }).join("");

    var actions = result.actions || [];
    var normalizationLabel = result.fallback
      ? "OpenAI fallback"
      : (result.normalization === "uploaded_stats" ? "uploaded" : "default");
    el.metrics.innerHTML = [
      metric(String(actions.length), "Actions emitted"),
      metric(Number.isFinite(Number(result.target_temperature_c)) ? Number(result.target_temperature_c).toFixed(1) + " C" : "-", "Target temperature"),
      metric(normalizationLabel, "Mode")
    ].join("");
    el.actions.textContent = JSON.stringify(actions, null, 2);
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

  function renderPayload(created) {
    var payload = sample ? {
      action: "createInference",
      place: sample.sample_id,
      date: sample.date,
      target_temperature_c: Number(el.target.value),
      raw_features: sample.feature_vector,
      mask_data_url: originalMaskDataUrl ? "[data-url omitted]" : null,
      checkpoint_key: (el.checkpoint.files && el.checkpoint.files[0]) ? "media/inference/<uuid>/checkpoint.pt" : null,
      fallback_without_checkpoint: !(el.checkpoint.files && el.checkpoint.files[0]),
      job: created ? { id: created.id, status: created.status } : null
    } : {};
    el.payload.textContent = JSON.stringify(payload, null, 2);
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
