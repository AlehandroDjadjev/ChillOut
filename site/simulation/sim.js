/* ChillOut Simulation Studio — scenario editor + result viewer.
   Emits the exact scenario contract (see infra/scenario.example.json), validates against
   the backend, queues a WRF job, and polls its status. No simulation runs in the browser. */
(function () {
  "use strict";

  var API_BASE = "https://7mfql7bgxtukoux34v3g26nbie0axyyc.lambda-url.eu-central-1.on.aws/";
  var KM_PER_DEG_LAT = 111.0;
  var $ = function (id) { return document.getElementById(id); };

  // ---- map ----------------------------------------------------------------
  var map = L.map("map", { zoomControl: true, attributionControl: false })
    .setView([42.42, 25.62], 8);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
  }).addTo(map);

  var domainRect = null;
  var drawnItems = new L.FeatureGroup().addTo(map);

  var drawControl = new L.Control.Draw({
    draw: {
      polygon: { shapeOptions: { color: "#ff8a5c", weight: 1.5, fillOpacity: 0.12 } },
      rectangle: { shapeOptions: { color: "#ff8a5c", weight: 1.5, fillOpacity: 0.12 } },
      polyline: false, circle: false, marker: false, circlemarker: false,
    },
    edit: { featureGroup: drawnItems, edit: true, remove: true },
  });
  map.addControl(drawControl);

  // Accumulate every drawn shape — targets can be many polygons (clouds have complex shapes).
  map.on(L.Draw.Event.CREATED, function (e) { drawnItems.addLayer(e.layer); sync(); });
  map.on(L.Draw.Event.DELETED, sync);
  map.on(L.Draw.Event.EDITED, sync);
  $("clearPoly").addEventListener("click", function () { drawnItems.clearLayers(); sync(); });

  function domainBounds(lat, lon, sizeKm) {
    var half = sizeKm / 2;
    var dLat = half / KM_PER_DEG_LAT;
    var cos = Math.cos((lat * Math.PI) / 180) || 1e-6;
    var dLon = half / (KM_PER_DEG_LAT * Math.abs(cos));
    return [[lat - dLat, lon - dLon], [lat + dLat, lon + dLon]];
  }

  function drawDomain() {
    var lat = parseFloat($("centerLat").value);
    var lon = parseFloat($("centerLon").value);
    var size = parseFloat($("innerSize").value);
    if (isNaN(lat) || isNaN(lon) || isNaN(size)) return;
    var b = domainBounds(lat, lon, size);
    if (domainRect) map.removeLayer(domainRect);
    domainRect = L.rectangle(b, { color: "#e2553a", weight: 1.5, dashArray: "6 5", fill: false }).addTo(map);
  }

  // ---- scenario build -----------------------------------------------------
  // GeoJSON MultiPolygon coordinates: one [ring] per drawn shape. Each shape contributes
  // its outer ring (getLatLngs()[0]); the domain rect lives on `map`, not drawnItems, so it
  // is excluded. Empty (no shapes) → whole-domain fallback downstream.
  function polygonCoords() {
    var polys = [];
    drawnItems.getLayers().forEach(function (layer) {
      if (!layer.getLatLngs) return;
      var latlngs = layer.getLatLngs()[0];
      if (!latlngs) return;
      var ring = latlngs.map(function (p) { return [p.lng, p.lat]; });
      if (ring.length < 3) return;
      ring.push(ring[0].slice()); // close ring
      polys.push([ring]);
    });
    return polys;
  }

  function num(id) { return parseFloat($(id).value); }

  function buildScenario() {
    var start = $("startUtc").value; // local datetime-local string
    var startUtc = start ? new Date(start).toISOString().replace(/\.\d{3}Z$/, "Z") : "";
    return {
      simulation_id: "studio_" + Date.now(),
      region: {
        target_polygon: { type: "MultiPolygon", coordinates: polygonCoords() },
        domain_center: { lat: num("centerLat"), lon: num("centerLon") },
      },
      time: {
        start_utc: startUtc,
        duration_hours: num("durationHours"),
        output_interval_minutes: num("outInterval"),
      },
      domain: {
        max_domains: parseInt($("maxDomains").value, 10),
        outer_resolution_m: num("outerRes"),
        inner_resolution_m: num("innerRes"),
        inner_domain_size_km: num("innerSize"),
      },
      input_data: { source: $("inputSource").value, forecast_cycle: $("forecastCycle").value },
      objective: {
        target_variable: $("targetVar").value,
        desired_delta_c: num("desiredDelta"),
        target_time_window_hours: [num("winStart"), num("winEnd")],
        max_precipitation_mm: num("maxPrecip"),
      },
      bad_scenario: { type: "baseline" },
      good_scenario: {
        type: "cloud_modification",
        cloud: {
          cloud_type: $("cloudType").value,
          base_height_m_agl: num("baseH"),
          top_height_m_agl: num("topH"),
          cloud_fraction: num("cloudFrac"),
          liquid_water_mixing_ratio_kg_kg: num("lwc"),
          ice_mixing_ratio_kg_kg: num("iwc"),
          start_offset_hours: num("cloudStart"),
          duration_hours: num("cloudDur"),
        },
      },
    };
  }

  // ---- live validation (server is authoritative) --------------------------
  var validateTimer = null;
  function renderErrors(errors) {
    var ul = $("errors");
    ul.innerHTML = "";
    if (!errors || !errors.length) { ul.hidden = true; $("submitBtn").disabled = false; return; }
    errors.forEach(function (e) {
      var li = document.createElement("li");
      li.innerHTML = "<b>" + (e.field || "scenario") + "</b> — " + e.reason;
      ul.appendChild(li);
    });
    ul.hidden = false;
    $("submitBtn").disabled = true;
  }

  function validate(scenario) {
    clearTimeout(validateTimer);
    validateTimer = setTimeout(function () {
      api("validateScenario", { scenario: scenario })
        .then(function (r) { renderErrors(r.errors); })
        .catch(function () { /* offline: let submit surface it */ });
    }, 300);
  }

  function sync() {
    $("durOut").textContent = $("durationHours").value;
    $("sizeOut").textContent = $("innerSize").value;
    $("fracOut").textContent = $("cloudFrac").value;
    drawDomain();
    var s = buildScenario();
    if (!$("jsonView").hidden) $("jsonView").textContent = JSON.stringify(s, null, 2);
    validate(s);
  }

  // bind all inputs
  Array.prototype.forEach.call(document.querySelectorAll("input, select"), function (el) {
    el.addEventListener("input", sync);
    el.addEventListener("change", sync);
  });

  $("toggleJson").addEventListener("click", function () {
    var v = $("jsonView");
    v.hidden = !v.hidden;
    this.textContent = v.hidden ? "View scenario JSON" : "Hide scenario JSON";
    if (!v.hidden) v.textContent = JSON.stringify(buildScenario(), null, 2);
  });

  // ---- API ----------------------------------------------------------------
  function api(action, payload) {
    return fetch(API_BASE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.assign({ action: action }, payload || {})),
    }).then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) { if (!res.ok && res.body && res.body.errors) return res.body; if (!res.ok) throw new Error((res.body && res.body.error) || "request failed"); return res.body; });
  }

  function toast(msg) {
    var t = document.createElement("div");
    t.className = "toast"; t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(function () { t.remove(); }, 3200);
  }

  // ---- result viewer ------------------------------------------------------
  var STATUSES = ["created", "validating", "preprocessing", "initializing",
    "running_baseline", "running_candidate", "postprocessing", "completed"];
  var STATUS_MSG = {
    created: ["Queued", "Waiting for the WRF worker to pick up this job."],
    validating: ["Validating scenario", "Checking the scenario contract against the model's limits."],
    preprocessing: ["Fetching GFS + preprocessing", "Downloading the latest GFS cycle and running geogrid / ungrib / metgrid."],
    initializing: ["Initializing the model", "real.exe is building initial and boundary conditions for the domain."],
    running_baseline: ["Running baseline model", "WRF is integrating the unmodified atmosphere — this is the heavy step."],
    running_candidate: ["Injecting cloud + candidate run", "Seeding the target column, then re-running WRF to measure the effect."],
    postprocessing: ["Computing the delta", "Comparing baseline vs. candidate over your target window."],
  };
  var pollTimer = null;
  var timerTick = null;
  var runStartMs = 0;

  function fmtElapsed(ms) {
    var s = Math.floor(ms / 1000);
    var m = Math.floor(s / 60);
    s = s % 60;
    return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
  }

  function stopTimer() { if (timerTick) { clearInterval(timerTick); timerTick = null; } }

  function startTimer() {
    stopTimer();
    timerTick = setInterval(function () {
      var el = $("workTimer");
      if (el) el.textContent = fmtElapsed(Date.now() - runStartMs);
      else stopTimer();
    }, 1000);
  }

  function renderPipeline(status) {
    var ol = $("pipeline"); ol.innerHTML = "";
    var failed = status === "failed";
    var idx = STATUSES.indexOf(status);
    STATUSES.forEach(function (st, i) {
      var li = document.createElement("li");
      li.textContent = st.replace(/_/g, " ");
      if (idx >= 0 && i < idx) li.classList.add("is-done");
      if (i === idx) li.classList.add(status === "completed" ? "is-done" : "is-active");
      ol.appendChild(li);
    });
    if (failed) {
      var li = document.createElement("li");
      li.textContent = "failed"; li.classList.add("is-failed");
      ol.appendChild(li);
    }
  }

  // One result panel: #resultStatus carries the live spinner / error; #player is the inline
  // animation (always laid out, no hidden tab); #resultStats holds metrics + summary. On
  // completion all three share one view — no tab switching.
  function renderResult(data) {
    renderPipeline(data.status);
    var status = $("resultStatus");
    var stats = $("resultStats");

    if (data.status === "failed") {
      stopTimer();
      if (window.WrfPlayer) window.WrfPlayer.reset();
      status.hidden = false;
      status.className = "result__error";
      status.innerHTML = "<b>This run did not complete.</b><br>" +
        "Worker error: " + (data.error || "unknown");
      stats.innerHTML = "";
      return;
    }

    if (data.status === "completed" && data.result) {
      stopTimer();
      var r = data.result;
      status.hidden = true;
      var html = '<div class="metrics">';
      (r.metrics || []).forEach(function (m) {
        html += '<div class="metric"><b>' + m.value + '</b><span>' + m.label + '</span></div>';
      });
      html += "</div>";
      if (r.summary) html += '<p class="result__summary">' + r.summary + "</p>";
      stats.innerHTML = html;
      if (window.WrfPlayer) {
        if (r.animation) window.WrfPlayer.load(r.animation);
        else window.WrfPlayer.reset();
      }
      return;
    }

    // working / created
    if (window.WrfPlayer) window.WrfPlayer.reset();
    stats.innerHTML = "";
    var msg = STATUS_MSG[data.status] || ["Working", "The WRF worker is processing this job."];
    status.hidden = false;
    status.className = "placeholder";
    status.innerHTML =
      '<div class="working">' +
        '<span class="spinner spinner--lg"></span>' +
        '<div class="working__msg">' + msg[0] + '</div>' +
        '<div class="working__sub">' + msg[1] + '</div>' +
        '<div class="working-bar"></div>' +
        '<div class="working__timer">elapsed <span id="workTimer">' +
          fmtElapsed(Date.now() - runStartMs) + '</span></div>' +
      "</div>";
    startTimer();
  }

  function poll(id) {
    clearTimeout(pollTimer);
    api("getResults", { id: id }).then(function (data) {
      renderResult(data);
      if (data.status !== "completed" && data.status !== "failed") {
        pollTimer = setTimeout(function () { poll(id); }, 4000);
      }
    }).catch(function () { pollTimer = setTimeout(function () { poll(id); }, 6000); });
  }

  $("submitBtn").addEventListener("click", function () {
    var scenario = buildScenario();
    var btn = this;
    btn.disabled = true;
    btn.classList.add("is-busy");
    api("createSimulation", { scenario: scenario }).then(function (r) {
      if (r.valid === false) { renderErrors(r.errors); toast("Fix validation errors first"); return; }
      runStartMs = Date.now();
      $("result").hidden = false;
      $("resId").textContent = r.id;
      $("result").scrollIntoView({ behavior: "smooth" });
      renderResult({ status: r.status });
      poll(r.id);
      toast("Simulation queued");
    }).catch(function (e) { toast("Error: " + e.message); })
      .finally(function () { btn.disabled = false; btn.classList.remove("is-busy"); });
  });

  $("newRun").addEventListener("click", function () {
    clearTimeout(pollTimer);
    stopTimer();
    if (window.WrfPlayer) window.WrfPlayer.reset();
    $("result").hidden = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // init
  drawDomain();
  sync();
})();
