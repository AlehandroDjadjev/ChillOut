(function () {
  "use strict";

  var meta = null;
  var el = {
    form: document.getElementById("cloudModelForm"),
    split: document.getElementById("splitSelect"),
    sampleIndex: document.getElementById("sampleIndex"),
    inputMode: document.getElementById("inputMode"),
    targetDeltaC: document.getElementById("targetDeltaC"),
    wm2PerC: document.getElementById("wm2PerC"),
    targetLoss: document.getElementById("targetLoss"),
    runOracle: document.getElementById("runOracle"),
    runBtn: document.getElementById("runBtn"),
    status: document.getElementById("status"),
    servicePill: document.getElementById("servicePill"),
    resultPill: document.getElementById("resultPill"),
    metaMetrics: document.getElementById("metaMetrics"),
    metrics: document.getElementById("metrics"),
    imageGrid: document.getElementById("imageGrid"),
    templateTable: document.getElementById("templateTable"),
    modeTable: document.getElementById("modeTable"),
    resultJson: document.getElementById("resultJson")
  };

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (ch) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[ch];
    });
  }

  function fmt(value, digits) {
    var n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits == null ? 2 : digits) : "-";
  }

  function metric(value, label) {
    return "<div class=\"lab-metric\"><b>" + escapeHtml(value) + "</b><span>" + escapeHtml(label) + "</span></div>";
  }

  function figure(src, label) {
    if (!src) return "";
    return "<figure class=\"lab-frame\"><img src=\"" + src + "\" alt=\"" + escapeHtml(label) + "\" /><figcaption>" + escapeHtml(label) + "</figcaption></figure>";
  }

  async function postJson(url, body) {
    var res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body || {})
    });
    var json = await res.json().catch(function () { return null; });
    if (!res.ok) {
      throw new Error((json && (json.detail || json.error || json.message)) || "Cloud model request failed.");
    }
    return json;
  }

  async function fetchJson(url) {
    var res = await fetch(url);
    var json = await res.json().catch(function () { return null; });
    if (!res.ok) {
      throw new Error((json && (json.detail || json.error || json.message)) || "Cloud model service failed.");
    }
    return json;
  }

  function setStatus(message, kind) {
    el.status.textContent = message || "";
    el.status.classList.toggle("is-error", kind === "error");
    el.status.classList.toggle("is-ok", kind === "ok");
  }

  function updateMaxIndex() {
    if (!meta || !meta.splits) return;
    var count = Number(meta.splits[el.split.value] || 0);
    el.sampleIndex.max = Math.max(0, count - 1);
  }

  function renderMeta(payload) {
    meta = payload;
    el.servicePill.textContent = "ready";
    updateMaxIndex();
    el.metaMetrics.innerHTML = [
      metric(payload.device || "-", "Device"),
      metric(String(payload.codebook_size || "-"), "Codebook templates"),
      metric(String((payload.splits && payload.splits.val) || "-"), "Validation samples")
    ].join("");
    setStatus("Cloud model service is ready.", "ok");
  }

  function renderTables(data) {
    var templates = (((data || {}).selector || {}).top_templates || []).slice(0, 8);
    el.templateTable.innerHTML = "<thead><tr><th>Template</th><th>Prob</th><th>Location</th><th>Coverage</th></tr></thead><tbody>" +
      templates.map(function (row) {
        return "<tr><td>" + escapeHtml(row.template_index) + "</td><td>" + fmt(row.prob, 3) + "</td><td>" +
          escapeHtml(row.location || "") + "</td><td>" + fmt(row.coverage, 3) + "</td></tr>";
      }).join("") + "</tbody>";

    var modes = (((data || {}).selector || {}).mode_probs || []);
    el.modeTable.innerHTML = "<thead><tr><th>Mode</th><th>Prob</th></tr></thead><tbody>" +
      modes.map(function (row) {
        return "<tr><td>" + escapeHtml(row.mode) + "</td><td>" + fmt(row.prob, 3) + "</td></tr>";
      }).join("") + "</tbody>";
  }

  function renderResult(data) {
    var verification = data.verification || {};
    var target = data.target || {};
    var chosen = data.chosen || {};
    var images = data.images || {};

    el.resultPill.textContent = chosen.mode || "complete";
    el.metrics.innerHTML = [
      metric(fmt(target.target_delta_c_proxy, 2) + " C", "Target proxy"),
      metric(fmt(verification.verified_temperature_change_c, 2) + " C", "Verified proxy"),
      metric(fmt(verification.temperature_abs_error_c, 2) + " C", "Proxy error")
    ].join("");

    el.imageGrid.innerHTML = [
      figure(images.original, "Original sample"),
      figure(images.model_input, "Model input"),
      figure(images.generated, "Generated cloud scene"),
      figure(images.original_mask, "Original cloud mask"),
      figure(images.generated_mask, "Generated cloud mask"),
      figure(images.added_delta, "Delta: added red, removed blue"),
      figure(images.original_strip, "Original sequence"),
      figure(images.generated_strip, "Generated sequence"),
      figure(images.channels, "Generated channels")
    ].join("");

    renderTables(data);
    el.resultJson.textContent = JSON.stringify({
      sample: data.sample,
      target: data.target,
      verification: data.verification,
      chosen: data.chosen,
      selector: data.selector
    }, null, 2);
  }

  async function loadMeta() {
    el.servicePill.textContent = "loading";
    renderMeta(await fetchJson("/api/cloud-model/meta"));
  }

  async function runGenerate(event) {
    event.preventDefault();
    el.runBtn.disabled = true;
    el.resultPill.textContent = "running";
    setStatus("Running cloud model inference...");
    try {
      var targetLossRaw = el.targetLoss.value.trim();
      var payload = {
        split: el.split.value,
        sample_index: Number(el.sampleIndex.value),
        target_delta_c: Number(el.targetDeltaC.value),
        wm2_per_c: Number(el.wm2PerC.value),
        target_loss_wm2: targetLossRaw === "" ? null : Number(targetLossRaw),
        input_mode: el.inputMode.value,
        run_oracle: el.runOracle.checked
      };
      renderResult(await postJson("/api/cloud-model/generate", payload));
      setStatus("Cloud model inference complete.", "ok");
    } catch (error) {
      el.resultPill.textContent = "error";
      setStatus(error.message || String(error), "error");
    } finally {
      el.runBtn.disabled = false;
    }
  }

  el.form.addEventListener("submit", runGenerate);
  el.split.addEventListener("change", updateMaxIndex);
  loadMeta().catch(function (error) {
    el.servicePill.textContent = "offline";
    setStatus(error.message || String(error), "error");
  });
})();
