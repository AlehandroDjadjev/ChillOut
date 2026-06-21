(function () {
  "use strict";

  var FEATURE_NAMES = [
    "s5p_cloud_fraction",
    "s5p_cloud_optical_thickness",
    "s5p_cloud_top_height_m",
    "s5p_cloud_top_pressure_pa",
    "s5p_aerosol_index",
    "s3_humidity_pct",
    "s3_sea_level_pressure_hpa",
    "s3_water_vapour_kg_m2",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "precipitation_sum",
    "cloud_cover_mean",
    "surface_pressure_mean"
  ];

  var CITY_PRESETS = [
    { name: "Plovdiv", country: "Bulgaria", lat: 42.1354, lon: 24.7453 },
    { name: "Sofia", country: "Bulgaria", lat: 42.6977, lon: 23.3219 },
    { name: "Varna", country: "Bulgaria", lat: 43.2141, lon: 27.9147 },
    { name: "Burgas", country: "Bulgaria", lat: 42.5048, lon: 27.4626 },
    { name: "Belgrade", country: "Serbia", lat: 44.7866, lon: 20.4489 },
    { name: "Skopje", country: "North Macedonia", lat: 41.9973, lon: 21.428 },
    { name: "Kyiv", country: "Ukraine", lat: 50.4501, lon: 30.5234 },
    { name: "Phoenix", country: "United States", lat: 33.4484, lon: -112.074 },
    { name: "Madrid", country: "Spain", lat: 40.4168, lon: -3.7038 }
  ];

  var DEFAULT_API_BASE = "https://7mfql7bgxtukoux34v3g26nbie0axyyc.lambda-url.eu-central-1.on.aws/";

  function getConfig() {
    var globalConfig = window.ChillOutModelConfig || {};
    return {
      temperatureEndpoint:
        globalConfig.temperatureEndpoint ||
        localStorage.getItem("chillout.temperatureEndpoint") ||
        "",
      plannerEndpoint:
        globalConfig.plannerEndpoint ||
        localStorage.getItem("chillout.plannerEndpoint") ||
        "",
      apiBase:
        globalConfig.apiBase ||
        localStorage.getItem("chillout.apiBase") ||
        DEFAULT_API_BASE
    };
  }

  function setEndpoint(kind, url) {
    var key = kind === "planner" ? "chillout.plannerEndpoint" : "chillout.temperatureEndpoint";
    if (url) localStorage.setItem(key, url);
    else localStorage.removeItem(key);
  }

  function todayIso() {
    return new Date().toISOString().slice(0, 10);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function round(value, digits) {
    var p = Math.pow(10, digits || 2);
    return Math.round(value * p) / p;
  }

  function mean(values) {
    var filtered = values.filter(function (v) { return Number.isFinite(v); });
    if (!filtered.length) return 0;
    return filtered.reduce(function (sum, v) { return sum + v; }, 0) / filtered.length;
  }

  function sum(values) {
    return values.filter(function (v) { return Number.isFinite(v); })
      .reduce(function (total, v) { return total + v; }, 0);
  }

  function circularMeanDegrees(values) {
    var radians = values.filter(function (v) { return Number.isFinite(v); })
      .map(function (v) { return (v * Math.PI) / 180; });
    if (!radians.length) return 0;
    var sin = mean(radians.map(Math.sin));
    var cos = mean(radians.map(Math.cos));
    var deg = (Math.atan2(sin, cos) * 180) / Math.PI;
    return (deg + 360) % 360;
  }

  function slug(value) {
    return String(value || "sample")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  }

  function seededRandom(seedText) {
    var seed = 2166136261;
    for (var i = 0; i < seedText.length; i += 1) {
      seed ^= seedText.charCodeAt(i);
      seed = Math.imul(seed, 16777619);
    }
    return function () {
      seed += 0x6D2B79F5;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  async function geocodePlace(query) {
    var q = String(query || "").trim();
    if (!q) throw new Error("Enter a place name first.");
    var url = "https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name=" +
      encodeURIComponent(q);
    var response = await fetch(url);
    if (!response.ok) throw new Error("Could not geocode that place.");
    var payload = await response.json();
    var result = payload.results && payload.results[0];
    if (!result) throw new Error("No matching place found.");
    return {
      name: result.name,
      country: result.country || "",
      lat: result.latitude,
      lon: result.longitude
    };
  }

  // Derive the 8 satellite-ish fields from coarse weather inputs. Used as a preview when the real
  // Copernicus stats are unavailable, and to fill any single field a satellite pass missed.
  function deriveSatelliteInputs(w) {
    var cloudFraction = clamp((w.cloudCover || 0) / 100, 0, 1);
    var humidity = Number.isFinite(w.humidity) ? w.humidity : clamp(40 + cloudFraction * 45, 0, 100);
    var precipitation = w.precipitation || 0;
    var windSpeed = w.windSpeed || 0;
    var temperature = Number.isFinite(w.temperature) ? w.temperature : 15;
    var opticalThickness = clamp(2 + cloudFraction * 34 + precipitation * 1.6 + humidity * 0.04, 0, 100);
    var cloudTopHeight = clamp(900 + cloudFraction * 5200 + windSpeed * 70 + precipitation * 120, 500, 12000);
    var cloudTopPressure = clamp(101325 * Math.pow(1 - (2.25577e-5 * cloudTopHeight), 5.2559), 18000, 100000);
    var aerosolIndex = clamp(-1.2 + (windSpeed / 35) + ((100 - humidity) / 180) - precipitation * 0.04, -2, 5);
    var seaLevelPressure = (Number.isFinite(w.surfacePressure) ? w.surfacePressure : 1010) + 14;
    var waterVapour = clamp(4 + humidity * 0.28 + temperature * 0.18, 0, 70);
    return {
      s5p_cloud_fraction: round(cloudFraction, 4),
      s5p_cloud_optical_thickness: round(opticalThickness, 3),
      s5p_cloud_top_height_m: round(cloudTopHeight, 2),
      s5p_cloud_top_pressure_pa: round(cloudTopPressure, 2),
      s5p_aerosol_index: round(aerosolIndex, 4),
      s3_humidity_pct: round(humidity, 2),
      s3_sea_level_pressure_hpa: round(seaLevelPressure, 2),
      s3_water_vapour_kg_m2: round(waterVapour, 2)
    };
  }

  // Call the backend buildSample action: real Copernicus Sentinel-5P/3 stats + Open-Meteo weather
  // for one city + date. Any satellite field a pass missed (even after window widening) is filled
  // from deriveSatelliteInputs so the page still renders, and flagged in source_notes.
  async function fetchRealSample(place, date) {
    var apiBase = getConfig().apiBase;
    if (!apiBase) throw new Error("No API base configured for sample fetch.");
    var result = await postJson(apiBase, {
      action: "buildSample",
      place: place.name,
      lat: place.lat,
      lon: place.lon,
      date: date || todayIso()
    });
    var sample = result && result.sample;
    if (!sample || !Array.isArray(sample.feature_vector)) {
      throw new Error("buildSample returned no sample.");
    }

    var missing = (sample.source_notes && sample.source_notes.missing_features) || [];
    if (missing.length) {
      var proxy = deriveSatelliteInputs({
        cloudCover: sample.inputs.cloud_cover_mean,
        humidity: sample.inputs.s3_humidity_pct,
        precipitation: sample.inputs.precipitation_sum,
        windSpeed: sample.inputs.wind_speed_10m_mean,
        temperature: sample.observed_temperature_c,
        surfacePressure: sample.inputs.surface_pressure_mean
      });
      missing.forEach(function (name) {
        if (sample.inputs[name] === null || sample.inputs[name] === undefined) {
          if (proxy[name] !== undefined) sample.inputs[name] = proxy[name];
        }
      });
    }
    // Recompute the vector after any fills and ensure no nulls leak into the model contract.
    sample.country = place.country || sample.country || "";
    sample.feature_vector = FEATURE_NAMES.map(function (name) {
      var value = sample.inputs[name];
      return Number.isFinite(Number(value)) ? Number(value) : 0;
    });
    sample.mode = "live";
    return sample;
  }

  async function fetchWeatherSample(place, date) {
    try {
      return await fetchRealSample(place, date);
    } catch (realError) {
      var preview = await fetchPreviewSample(place, date);
      preview.source_notes = preview.source_notes || {};
      preview.source_notes.fallback_reason = "Live satellite fetch failed: " + (realError.message || realError);
      return preview;
    }
  }

  async function fetchPreviewSample(place, date) {
    var selectedDate = date || todayIso();
    var hourly = [
      "temperature_2m",
      "relative_humidity_2m",
      "cloud_cover",
      "surface_pressure",
      "precipitation",
      "shortwave_radiation",
      "wind_speed_10m",
      "wind_direction_10m"
    ].join(",");
    var url = "https://api.open-meteo.com/v1/forecast?latitude=" + encodeURIComponent(place.lat) +
      "&longitude=" + encodeURIComponent(place.lon) +
      "&hourly=" + hourly +
      "&past_days=7&forecast_days=7&timezone=UTC";
    var response = await fetch(url);
    if (!response.ok) throw new Error("Weather fetch failed.");
    var payload = await response.json();
    var h = payload.hourly || {};
    var times = h.time || [];
    var indexes = times.reduce(function (acc, value, index) {
      if (String(value).slice(0, 10) === selectedDate) acc.push(index);
      return acc;
    }, []);

    if (!indexes.length) {
      selectedDate = times[0] ? String(times[0]).slice(0, 10) : selectedDate;
      indexes = times.reduce(function (acc, value, index) {
        if (String(value).slice(0, 10) === selectedDate) acc.push(index);
        return acc;
      }, []);
    }

    if (!indexes.length) throw new Error("No hourly weather values were returned.");

    function pick(name) {
      var list = h[name] || [];
      return indexes.map(function (index) { return Number(list[index]); });
    }

    var temperature = mean(pick("temperature_2m"));
    var humidity = mean(pick("relative_humidity_2m"));
    var cloudCover = mean(pick("cloud_cover"));
    var surfacePressure = mean(pick("surface_pressure"));
    var precipitation = sum(pick("precipitation"));
    var shortwaveMj = sum(pick("shortwave_radiation")) * 3600 / 1000000;
    var windSpeed = mean(pick("wind_speed_10m"));
    var windDirection = circularMeanDegrees(pick("wind_direction_10m"));
    var cloudFraction = clamp(cloudCover / 100, 0, 1);
    var opticalThickness = clamp(2 + cloudFraction * 34 + precipitation * 1.6 + humidity * 0.04, 0, 100);
    var cloudTopHeight = clamp(900 + cloudFraction * 5200 + windSpeed * 70 + precipitation * 120, 500, 12000);
    var cloudTopPressure = clamp(101325 * Math.pow(1 - (2.25577e-5 * cloudTopHeight), 5.2559), 18000, 100000);
    var aerosolIndex = clamp(-1.2 + (windSpeed / 35) + ((100 - humidity) / 180) - precipitation * 0.04, -2, 5);
    var seaLevelPressure = surfacePressure + 14;
    var waterVapour = clamp(4 + humidity * 0.28 + temperature * 0.18, 0, 70);

    var inputs = {
      s5p_cloud_fraction: round(cloudFraction, 4),
      s5p_cloud_optical_thickness: round(opticalThickness, 3),
      s5p_cloud_top_height_m: round(cloudTopHeight, 2),
      s5p_cloud_top_pressure_pa: round(cloudTopPressure, 2),
      s5p_aerosol_index: round(aerosolIndex, 4),
      s3_humidity_pct: round(humidity, 2),
      s3_sea_level_pressure_hpa: round(seaLevelPressure, 2),
      s3_water_vapour_kg_m2: round(waterVapour, 2),
      wind_speed_10m_mean: round(windSpeed, 2),
      wind_direction_10m_dominant: round(windDirection, 1),
      shortwave_radiation_sum: round(shortwaveMj, 2),
      precipitation_sum: round(precipitation, 2),
      cloud_cover_mean: round(cloudCover, 2),
      surface_pressure_mean: round(surfacePressure, 2)
    };

    var bbox = makeBbox(place.lat, place.lon, 20);
    return {
      sample_id: slug(place.name) + "_" + selectedDate,
      city: place.name,
      country: place.country,
      lat: place.lat,
      lon: place.lon,
      bbox: bbox,
      date: selectedDate,
      anchor: selectedDate + "T00:00:00Z",
      mask_path: "browser_generated_cloud_mask.png",
      observed_temperature_c: round(temperature, 2),
      target_temperature_c: null,
      inputs: inputs,
      feature_vector: FEATURE_NAMES.map(function (name) { return inputs[name]; }),
      source_notes: {
        mask: "Browser-generated cloud mask preview, sized like the RL prototype input.",
        satellite_stats: "Derived preview fields from Open-Meteo cloud, humidity, wind, and pressure values until Sentinel endpoints are wired here.",
        weather: "Open-Meteo hourly forecast/archive values aggregated to one day.",
        radiation_note: "Hourly shortwave radiation is summed into MJ/m2 to mirror the training feature scale."
      }
    };
  }

  function makeBbox(lat, lon, radiusKm) {
    var dLat = radiusKm / 111;
    var dLon = radiusKm / (111 * Math.max(0.2, Math.cos((lat * Math.PI) / 180)));
    return [round(lon - dLon, 6), round(lat - dLat, 6), round(lon + dLon, 6), round(lat + dLat, 6)];
  }

  function drawMask(canvas, sample) {
    var size = canvas.width || 256;
    canvas.width = size;
    canvas.height = size;
    var ctx = canvas.getContext("2d");
    var rand = seededRandom(sample.sample_id);
    var cloudFraction = sample.inputs.s5p_cloud_fraction;
    var wind = ((sample.inputs.wind_direction_10m_dominant - 90) * Math.PI) / 180;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = "#02090b";
    ctx.fillRect(0, 0, size, size);

    var blobs = Math.max(4, Math.round(5 + cloudFraction * 18));
    for (var i = 0; i < blobs; i += 1) {
      var x = rand() * size;
      var y = rand() * size;
      var rx = size * (0.05 + rand() * 0.16) * (0.7 + cloudFraction);
      var ry = rx * (0.38 + rand() * 0.46);
      var alpha = 0.18 + cloudFraction * 0.56 + rand() * 0.18;
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(wind + (rand() - 0.5) * 0.7);
      var gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
      gradient.addColorStop(0, "rgba(255,255,255," + alpha + ")");
      gradient.addColorStop(0.58, "rgba(215,235,238," + (alpha * 0.65) + ")");
      gradient.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    return canvas.toDataURL("image/png");
  }

  async function fetchSentinelImage(sample, mode) {
    var apiBase = getConfig().apiBase;
    if (!apiBase) throw new Error("No API base configured for Sentinel-2 fetch.");
    var result = await postJson(apiBase, {
      action: "sentinel2",
      bbox: sample.bbox,
      date: sample.date,
      mode: mode || "cloud_mask"
    });
    if (!result || !result.image_data_url) {
      throw new Error((result && (result.error || result.message)) || "Sentinel-2 fetch returned no image.");
    }
    return result.image_data_url;
  }

  async function fetchUploadUrl(jobUuid, filename, contentType) {
    var apiBase = getConfig().apiBase;
    if (!apiBase) throw new Error("No API base configured for upload.");
    var result = await postJson(apiBase, {
      action: "getUploadUrl",
      jobUuid: jobUuid,
      filename: filename,
      contentType: contentType || "application/octet-stream"
    });
    if (!result || !result.uploadUrl) throw new Error("Upload URL request returned nothing.");
    return result;
  }

  async function uploadFile(uploadUrl, file) {
    var response = await fetch(uploadUrl, {
      method: "PUT",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file
    });
    if (!response.ok) throw new Error("File upload to storage failed (" + response.status + ").");
  }

  async function createInference(payload) {
    var apiBase = getConfig().apiBase;
    if (!apiBase) throw new Error("No API base configured for inference.");
    return postJson(apiBase, Object.assign({ action: "createInference" }, payload));
  }

  async function getInference(id) {
    var apiBase = getConfig().apiBase;
    if (!apiBase) throw new Error("No API base configured for inference.");
    return postJson(apiBase, { action: "getInference", id: id });
  }

  async function postJson(url, body) {
    var response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    var json = await response.json().catch(function () { return null; });
    if (!response.ok) {
      throw new Error((json && (json.error || json.message)) || "Model endpoint failed.");
    }
    return json;
  }

  async function runTemperatureModel(sample, maskDataUrl) {
    var endpoint = getConfig().temperatureEndpoint;
    var payload = {
      sample: sample,
      cloud_mask_data_url: maskDataUrl,
      feature_names: FEATURE_NAMES
    };
    if (endpoint) {
      var realResult = await postJson(endpoint, payload);
      return Object.assign({ mode: "endpoint", endpoint: endpoint }, realResult);
    }

    var i = sample.inputs;
    var predicted = sample.observed_temperature_c -
      i.s5p_cloud_fraction * 2.2 -
      (i.s5p_cloud_optical_thickness / 100) * 1.4 +
      (i.shortwave_radiation_sum - 18) * 0.10 -
      i.precipitation_sum * 0.03 -
      (i.wind_speed_10m_mean - 8) * 0.04;

    return {
      mode: "frontend-preview",
      prediction_c: round(predicted, 2),
      baseline_observed_c: sample.observed_temperature_c,
      cloud_cooling_component_c: round(-i.s5p_cloud_fraction * 2.2 - (i.s5p_cloud_optical_thickness / 100) * 1.4, 2),
      radiation_component_c: round((i.shortwave_radiation_sum - 18) * 0.10, 2),
      confidence: round(clamp(0.58 + i.cloud_cover_mean / 300 - Math.abs(i.precipitation_sum) / 80, 0.45, 0.86), 2),
      note: "Preview heuristic only. Set chillout.temperatureEndpoint or window.ChillOutModelConfig.temperatureEndpoint to call the hosted model."
    };
  }

  function loadImageToCanvas(sourceUrl, canvas) {
    return new Promise(function (resolve, reject) {
      var image = new Image();
      image.onload = function () {
        canvas.width = 256;
        canvas.height = 256;
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/png"));
      };
      image.onerror = function () { reject(new Error("Could not read the image.")); };
      image.src = sourceUrl;
    });
  }

  function fileToDataUrl(file) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () { resolve(String(reader.result)); };
      reader.onerror = function () { reject(new Error("Could not read the selected file.")); };
      reader.readAsDataURL(file);
    });
  }

  async function runPlannerModel(sample, originalMaskDataUrl, targetTemperatureC) {
    var endpoint = getConfig().plannerEndpoint;
    var payload = {
      sample_id: sample.sample_id,
      original_mask_data_url: originalMaskDataUrl,
      raw_features: sample.inputs,
      feature_vector: sample.feature_vector,
      target_temperature_c: targetTemperatureC,
      image_shape: [256, 256],
      action_schema: ["op", "x01", "y01", "radius_px", "mass_proxy", "humidity_pct", "optical_thickness", "cloud_top_height_m", "cloud_type"]
    };
    if (endpoint) {
      var realResult = await postJson(endpoint, payload);
      return Object.assign({ mode: "endpoint", endpoint: endpoint, input: payload }, realResult);
    }
    return plannerPreview(payload, sample);
  }

  async function plannerPreview(payload, sample) {
    var canvas = document.createElement("canvas");
    var original = await loadImageToCanvas(payload.original_mask_data_url, canvas);
    var ctx = canvas.getContext("2d");
    var rand = seededRandom(payload.sample_id + "_planner");
    var delta = payload.target_temperature_c - sample.observed_temperature_c;
    var wantsCooling = delta < 0;
    var actions = [];
    var actionCount = Math.min(3, Math.max(1, Math.round(Math.abs(delta) / 1.8) + 1));

    for (var i = 0; i < actionCount; i += 1) {
      var x01 = 0.18 + rand() * 0.64;
      var y01 = 0.18 + rand() * 0.64;
      var radius = 10 + rand() * 18 + Math.abs(delta) * 2;
      var op = wantsCooling ? (i === 0 ? "create" : "modify") : (i === 0 ? "remove" : "modify");
      actions.push({
        op: op,
        x01: round(x01, 3),
        y01: round(y01, 3),
        radius_px: round(radius, 1),
        mass_proxy: round(clamp(0.38 + Math.abs(delta) / 10 + rand() * 0.25, 0, 1), 3),
        humidity_pct: round(clamp(sample.inputs.s3_humidity_pct + (wantsCooling ? 8 : -8), 0, 100), 1),
        optical_thickness: round(clamp(sample.inputs.s5p_cloud_optical_thickness + (wantsCooling ? 8 : -6), 0, 100), 1),
        cloud_top_height_m: round(clamp(sample.inputs.s5p_cloud_top_height_m + (wantsCooling ? 450 : -300), 500, 12000), 1),
        cloud_type: wantsCooling ? 0.28 : 0.64
      });

      ctx.save();
      ctx.globalCompositeOperation = op === "remove" ? "destination-out" : "screen";
      var x = x01 * canvas.width;
      var y = y01 * canvas.height;
      var gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
      gradient.addColorStop(0, op === "remove" ? "rgba(0,0,0,0.85)" : "rgba(255,255,255,0.78)");
      gradient.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    var generated = canvas.toDataURL("image/png");
    var expected = sample.observed_temperature_c + delta * 0.55;

    return {
      mode: "frontend-preview",
      input: payload,
      output: {
        original_mask_data_url: original,
        generated_mask_data_url: generated,
        generated_mask_path: payload.sample_id + "_generated.png",
        actions: actions,
        property_maps_summary: {
          humidity_norm_mean: round(clamp((sample.inputs.s3_humidity_pct + (wantsCooling ? 6 : -6)) / 100, 0, 1), 3),
          thickness_norm_mean: round(clamp((sample.inputs.s5p_cloud_optical_thickness + (wantsCooling ? 8 : -5)) / 100, 0, 1), 3),
          height_norm_mean: round(clamp((sample.inputs.s5p_cloud_top_height_m + (wantsCooling ? 450 : -300)) / 12000, 0, 1), 3),
          mass_proxy_mean: round(clamp(0.28 + Math.abs(delta) / 10, 0, 1), 3),
          cloud_type_mean: wantsCooling ? 0.28 : 0.64
        },
        expected_temperature_c: round(expected, 2),
        reward_proxy: round(1 / (1 + Math.abs(payload.target_temperature_c - expected)), 3)
      },
      note: "Preview heuristic only. Set chillout.plannerEndpoint or window.ChillOutModelConfig.plannerEndpoint to call the hosted RL policy."
    };
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (character) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[character];
    });
  }

  window.ChillOutContracts = {
    FEATURE_NAMES: FEATURE_NAMES,
    CITY_PRESETS: CITY_PRESETS,
    getConfig: getConfig,
    setEndpoint: setEndpoint,
    todayIso: todayIso,
    round: round,
    escapeHtml: escapeHtml,
    geocodePlace: geocodePlace,
    fetchWeatherSample: fetchWeatherSample,
    fetchRealSample: fetchRealSample,
    fetchSentinelImage: fetchSentinelImage,
    fetchUploadUrl: fetchUploadUrl,
    uploadFile: uploadFile,
    createInference: createInference,
    getInference: getInference,
    drawMask: drawMask,
    runTemperatureModel: runTemperatureModel,
    runPlannerModel: runPlannerModel,
    fileToDataUrl: fileToDataUrl,
    loadImageToCanvas: loadImageToCanvas
  };
})();
