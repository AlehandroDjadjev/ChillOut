#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from cloud_model_testapp_inference import CloudTemplateInference, InferenceConfig


APP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cloud Template Selector V13 Test App</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101316;
      --panel: #181d21;
      --panel2: #20262b;
      --line: #333b42;
      --text: #edf2f7;
      --muted: #a9b4bf;
      --accent: #43b88f;
      --accent2: #70a7ff;
      --warn: #ffcf70;
      --bad: #ff7a7a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      width: min(1420px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 20px 0 32px;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 720;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      text-align: right;
      line-height: 1.45;
    }
    .layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 16px;
      align-items: start;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    aside {
      position: sticky;
      top: 14px;
      padding: 14px;
    }
    label {
      display: grid;
      gap: 7px;
      margin-bottom: 13px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    input, select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #11161a;
      color: var(--text);
      padding: 8px 10px;
      font: inherit;
      outline: none;
    }
    input:focus, select:focus { border-color: var(--accent2); }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .check {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 10px 0 16px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
      text-transform: none;
      letter-spacing: 0;
    }
    .check input { width: 18px; min-height: 18px; }
    button {
      width: 100%;
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #06110d;
      font-weight: 780;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .hint {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .content {
      display: grid;
      gap: 16px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 14px;
    }
    .metric {
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 11px;
      min-height: 82px;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 9px;
    }
    .metric .value {
      font-size: 24px;
      font-weight: 780;
      line-height: 1.05;
    }
    .metric .small {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
      line-height: 1.35;
    }
    .images {
      padding: 14px;
    }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    figure {
      margin: 0;
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    figure img {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      background: #11161a;
    }
    figcaption {
      padding: 9px 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .wide {
      margin-top: 12px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .wide img {
      width: 100%;
      aspect-ratio: auto;
      object-fit: contain;
    }
    .tables {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .table-wrap {
      padding: 14px;
      overflow: auto;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      letter-spacing: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      color: var(--text);
    }
    th, td {
      text-align: left;
      padding: 8px 7px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      white-space: nowrap;
    }
    th { color: var(--muted); font-weight: 700; }
    .empty {
      padding: 42px 18px;
      color: var(--muted);
      text-align: center;
      line-height: 1.5;
    }
    .error {
      margin-top: 12px;
      color: var(--bad);
      font-size: 13px;
      line-height: 1.4;
      min-height: 18px;
    }
    @media (max-width: 980px) {
      .layout, .tables, .wide { grid-template-columns: 1fr; }
      aside { position: static; }
      .metrics, .image-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      header { display: block; }
      .status { text-align: left; margin-top: 8px; }
    }
    @media (max-width: 560px) {
      main { width: min(100vw - 20px, 1420px); }
      .metrics, .image-grid, .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Cloud Template Selector V13 Test App</h1>
      <div class="sub">Top model generates the cloud state; bottom V8 model verifies radiation and temperature proxy.</div>
    </div>
    <div class="status" id="status">Loading models...</div>
  </header>

  <div class="layout">
    <aside>
      <label>Split
        <select id="split">
          <option value="val">val</option>
          <option value="train">train</option>
          <option value="test">test</option>
        </select>
      </label>
      <div class="row">
        <label>Dataset index
          <input id="sampleIndex" type="number" min="0" step="1" value="50" />
        </label>
        <label>Input
          <select id="inputMode">
            <option value="full">full</option>
            <option value="dropout">dropout</option>
          </select>
        </label>
      </div>
      <div class="row">
        <label>Target temp delta C
          <input id="targetDeltaC" type="number" step="0.1" value="-2.0" />
        </label>
        <label>W/m2 per C
          <input id="wm2PerC" type="number" min="1" step="1" value="80" />
        </label>
      </div>
      <label>Absolute target loss W/m2
        <input id="targetLoss" type="number" step="1" placeholder="blank = use temp delta" />
      </label>
      <label class="check">
        Run codebook oracle
        <input id="runOracle" type="checkbox" checked />
      </label>
      <button id="runBtn">Generate and verify</button>
      <div class="hint">
        Negative temperature delta asks for more cloud-loss. Positive delta asks for less cloud-loss. The Celsius value is only a practical proxy for comparing outputs.
      </div>
      <div class="error" id="error"></div>
    </aside>

    <div class="content">
      <section id="resultPane">
        <div class="empty">Choose an index and generate. The first run may take a moment while the checkpoints settle on the device.</div>
      </section>
      <div class="tables">
        <section class="table-wrap">
          <h2>Selected Templates</h2>
          <table id="templateTable"><tbody></tbody></table>
        </section>
        <section class="table-wrap">
          <h2>Mode Probabilities</h2>
          <table id="modeTable"><tbody></tbody></table>
        </section>
      </div>
    </div>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
const fmt = (v, digits = 2) => v === null || v === undefined || Number.isNaN(Number(v)) ? "n/a" : Number(v).toFixed(digits);
let meta = null;

function metric(label, value, small = "") {
  return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div>${small ? `<div class="small">${small}</div>` : ""}</div>`;
}

function imageFigure(src, label) {
  return `<figure><img src="${src}" alt="${label}" /><figcaption>${label}</figcaption></figure>`;
}

function renderTables(data) {
  const tRows = data.selector.top_templates.map(row => `
    <tr>
      <td>${row.template_index}</td>
      <td>${fmt(row.prob, 3)}</td>
      <td>${row.location || ""}</td>
      <td>${fmt(row.coverage, 3)}</td>
    </tr>`).join("");
  $("templateTable").innerHTML = `<thead><tr><th>Index</th><th>Prob</th><th>Location</th><th>Coverage</th></tr></thead><tbody>${tRows}</tbody>`;

  const mRows = data.selector.mode_probs.map(row => `
    <tr><td>${row.mode}</td><td>${fmt(row.prob, 3)}</td></tr>`).join("");
  $("modeTable").innerHTML = `<thead><tr><th>Mode</th><th>Prob</th></tr></thead><tbody>${mRows}</tbody>`;
}

function renderResult(data) {
  const v = data.verification;
  const s = data.sample;
  const c = data.chosen;
  $("resultPane").innerHTML = `
    <div class="metrics">
      ${metric("Target proxy", `${fmt(data.target.target_delta_c_proxy, 2)} C`, `${fmt(data.target.target_loss_wm2, 1)} W/m2 loss`)}
      ${metric("Verified proxy", `${fmt(v.verified_temperature_change_c, 2)} C`, `${fmt(v.bottom_generated_loss_wm2, 1)} W/m2 by bottom model`)}
      ${metric("Error", `${fmt(v.target_abs_error_wm2, 1)} W/m2`, `${fmt(v.temperature_abs_error_c, 2)} C proxy`)}
      ${metric("Sample", `#${s.sample_index}`, `${s.location || "unknown"} | ${s.anchor || ""}`)}
      ${metric("Original bottom", `${fmt(v.bottom_original_loss_wm2, 1)}`, `${fmt(v.original_temperature_change_proxy_c, 2)} C proxy`)}
      ${metric("Input bottom", `${fmt(v.bottom_input_loss_wm2, 1)}`, `${fmt(v.input_temperature_change_proxy_c, 2)} C proxy`)}
      ${metric("Actual dataset", `${fmt(v.actual_dataset_loss_wm2, 1)}`, `clear sky ${fmt(v.clear_sky_wm2, 1)} W/m2`)}
      ${metric("Chosen action", c.mode, `${c.source} template ${c.template_index}`)}
    </div>
    <div class="images">
      <div class="image-grid">
        ${imageFigure(data.images.original, "Original sample")}
        ${imageFigure(data.images.model_input, "Model input")}
        ${imageFigure(data.images.selected_template, "Selected cloud template")}
        ${imageFigure(data.images.generated, "Generated and verified")}
      </div>
      <div class="wide">
        ${imageFigure(data.images.original_strip, "Original sequence")}
        ${imageFigure(data.images.generated_strip, "Generated sequence")}
      </div>
      <div class="wide">
        ${imageFigure(data.images.channels, "Generated channels")}
        <figure>
          <figcaption>
            Chosen template: ${c.template_location || "unknown"} / ${c.template_sample_id || "no id"}<br>
            Chosen error: ${fmt(c.abs_error_wm2, 2)} W/m2 | coverage ${fmt(c.coverage, 3)} | L1 ${fmt(c.image_l1_vs_input, 4)}<br>
            Dataset future temp: ${fmt(s.dataset_future_temperature_c, 2)} C | current ${fmt(s.current_temperature_c, 2)} C
          </figcaption>
        </figure>
      </div>
    </div>
  `;
  renderTables(data);
}

async function loadMeta() {
  const res = await fetch("/api/meta");
  if (!res.ok) throw new Error(await res.text());
  meta = await res.json();
  $("status").innerHTML = `Device: ${meta.device}<br>val ${meta.splits.val} | train ${meta.splits.train} | test ${meta.splits.test} | codebook ${meta.codebook_size}`;
  $("sampleIndex").max = meta.splits[$("split").value] - 1;
}

async function runGenerate() {
  $("runBtn").disabled = true;
  $("runBtn").textContent = "Generating...";
  $("error").textContent = "";
  try {
    const targetLossRaw = $("targetLoss").value.trim();
    const payload = {
      split: $("split").value,
      sample_index: Number($("sampleIndex").value),
      target_delta_c: Number($("targetDeltaC").value),
      wm2_per_c: Number($("wm2PerC").value),
      target_loss_wm2: targetLossRaw === "" ? null : Number(targetLossRaw),
      input_mode: $("inputMode").value,
      run_oracle: $("runOracle").checked
    };
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const detail = await res.json().catch(async () => ({detail: await res.text()}));
      throw new Error(detail.detail || "generation failed");
    }
    renderResult(await res.json());
  } catch (err) {
    $("error").textContent = err.message || String(err);
  } finally {
    $("runBtn").disabled = false;
    $("runBtn").textContent = "Generate and verify";
  }
}

$("split").addEventListener("change", () => {
  if (meta) $("sampleIndex").max = meta.splits[$("split").value] - 1;
});
$("runBtn").addEventListener("click", runGenerate);
loadMeta().catch(err => {
  $("status").textContent = "Model load failed";
  $("error").textContent = err.message || String(err);
});
</script>
</body>
</html>
"""


class GenerateRequest(BaseModel):
    split: str = Field(default="val")
    sample_index: int = Field(default=50, ge=0)
    target_loss_wm2: Optional[float] = None
    target_delta_c: float = -2.0
    wm2_per_c: float = Field(default=80.0, gt=0)
    input_mode: str = Field(default="full")
    run_oracle: bool = True
    oracle_batch_size: int = Field(default=32, ge=1, le=256)
    penalty_l1: float = 25.0
    penalty_coverage: float = 80.0
    seed: int = 123


def create_app(engine: CloudTemplateInference) -> FastAPI:
    app = FastAPI(title="Cloud Template Selector V13 Test App")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return APP_HTML

    @app.get("/api/meta")
    def meta():
        return engine.meta()

    @app.post("/api/generate")
    def generate(req: GenerateRequest):
        try:
            return engine.generate(
                split=req.split,
                sample_index=req.sample_index,
                target_loss_wm2=req.target_loss_wm2,
                target_delta_c=req.target_delta_c,
                wm2_per_c=req.wm2_per_c,
                input_mode=req.input_mode,
                run_oracle=req.run_oracle,
                oracle_batch_size=req.oracle_batch_size,
                penalty_l1=req.penalty_l1,
                penalty_coverage=req.penalty_coverage,
                seed=req.seed,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset_cloudforce_radiation_v6_big_clean")
    parser.add_argument("--selector-checkpoint", default="runs/cloud_template_selector_v13/best_selector.pt")
    parser.add_argument("--reward-checkpoint", default="runs/cloud_radiation_v8_clean_direct1/best.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    engine = CloudTemplateInference(
        InferenceConfig(
            data_root=Path(args.data_root),
            selector_checkpoint=Path(args.selector_checkpoint),
            reward_checkpoint=Path(args.reward_checkpoint),
            force_cpu=bool(args.force_cpu),
        )
    )
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
