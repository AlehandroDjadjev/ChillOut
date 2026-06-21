# Cloud-RL before/after demo — bundle real model outputs into a new dark page

## Context
The user uploaded a standalone "Cloud RL before / after test" page and asked to **integrate
the same thing, using the models we made, to demonstrate visually what's up**, and to make
**target values (target temperature + radiation) user-settable**.

The uploaded page can't work as-is on the deployed site because:
- It fetches `/dataset_out/*` and `/cloud_rl/*`, which are **gitignored** and not served by
  CloudFront (OpenKBS deploys **only `site/`**).
- "Before" imagery comes from `POST /api/process` (Sentinel-2), which lives only in the
  un-deployed `server.mjs`; the deployed Lambda (`functions/api`) is Node and has no such route
  and cannot run PyTorch.
- Its metrics are **heuristic JS arithmetic** (lines 177–186), not real model output.

Decisions confirmed with the user:
1. **Data strategy: precompute & bundle real model outputs.** Run the trained models offline,
   emit JSON + mask PNGs, and bundle them into `site/` so the page is fully static.
2. **Placement: a new dedicated dark-themed page** (`site/lab/`), linked from the nav — kept
   separate from the WRF simulation studio.
3. **User-settable target: target temperature** (the RL agent's real objective), with
   **radiation surfaced as a real predicted before/after outcome** from the same v6 reward model.

Local facts established: `dataset_out/` is tiny (75 records, masks ~420 KB) and trivially
bundleable; the RL pipeline is `cloud_rl/evaluate.py` → `CloudActorCritic` policy + the
`cloudforced_radiation_v6` reward model (`first_model_reward.py`), which exposes
`predicted_temperature_c` / `original_predicted_temperature_c`. Target temperature is injected
by writing `batch["target_temp"]`, `target_temp_norm`, and the last `obs_map` channel — exactly
what `targeting.augment_target_temperature` does with a random offset.

## Prerequisite (blocking the precompute step)
**No trained checkpoints and no `predictions.json` exist in this workspace** (`cloud_rl/runs/`
is empty; only `dataset.last/dataset.pt`, a serialized dataset, is present). To produce REAL
model outputs the user must supply, in the workspace:
- the **RL agent checkpoint** (`CloudActorCritic`, the `--checkpoint` arg), and
- the **v6 reward checkpoint** (temperature/radiation predictor, the `--reward-checkpoint` arg).

If they're available, the precompute script below runs locally and bundles its outputs. The
frontend is built against the bundled-JSON contract and can be developed in parallel using a
small hand-written fixture, then swapped to the real precomputed data.

## Part A — Precompute script (new: `cloud_rl/export_demo.py`)
A thin variant of `evaluate.py` that sweeps user-selectable target temperatures per sample.

- Args: `--checkpoint`, `--reward-checkpoint`, `--config`, `--data-root dataset_out`,
  `--split all`, `--out-dir site/lab/data`, `--targets-around 5 --target-step 1.0`
  (sweep observed temp ±5 °C in 1 °C steps → 11 targets/sample).
- Load policy + v6 reward model exactly as `evaluate.py` does (lines 41–80).
- For each sample, for each target T in the sweep: set `target_temp`/`target_temp_norm`/
  `obs_map[:, -1:]` to T (reuse the injection math in `targeting.py:39–41`, but with an explicit
  T instead of a random offset), run `model.deterministic(...)`, `rasterize_actions(...)`, then
  `reward_fn(...)` to get `original_predicted_temperature_c`, `predicted_temperature_c`, and the
  radiation/cloud property maps.
- Save per (sample, target): `generated_<sid>_<T>.png` (generated cloud mask) +
  `cloud_prob_<sid>_<T>.png`. Save the original mask once per sample: `original_<sid>.png`.
- Emit `site/lab/data/predictions.json`: an array of
  `{ sample_id, city, date, bbox, observed_temperature_c, base_radiation, base_cloud_cover,
     targets: [ { target_temperature_c, predicted_before_c, predicted_after_c,
                  predicted_radiation_before, predicted_radiation_after,
                  cloud_cover_after, masks: { original, generated, cloud_prob } } ] }`.
- Also emit `site/lab/data/index.json`: `{ cities:[…], byCity:{ city:[{sample_id,date}] } }`
  so the frontend doesn't parse the whole dataset.
- **Radiation:** read it from the v6 reward model's outputs (it is radiation-forced). If a given
  build of the reward model does not expose a radiation head, fall back to deriving the
  shortwave shift from the predicted cloud-cover change using the dataset's own
  `shortwave_radiation_sum`↔`cloud_cover_mean` relationship, and label that field
  `radiation_estimated: true` so the UI can mark it as derived. (Resolve which at implementation
  time by reading `first_model_reward.py` forward outputs around lines 616–727.)

Copy the per-sample original masks the script needs from `dataset_out/masks/<city>/<date>.png`.
Net bundle: `predictions.json`, `index.json`, and a few hundred small PNGs — comfortably static.

## Part B — New page (`site/lab/index.html`, `site/lab/lab.css`, `site/lab/lab.js`)
Mirror the structure/markup conventions of `site/simulation/` (shared `../css/styles.css`
tokens + a page-scoped CSS file), styled to the **dark teal-navy + yellow** theme already live.

- **Header/nav:** same nav as `simulation/index.html` with a "Lab" / "Before-after" tag;
  add a nav link to it from `index.html` and `simulation/index.html`.
- **Controls panel:** Destination `<select>` + Date `<select>` (populated from `index.json`),
  and a **Target temperature slider** (range = observed ±5 °C, step 1 °C) with a live readout.
  A read-only **Target radiation** field shows the predicted radiation outcome for the selected
  plan (and updates as the slider moves); if `radiation_estimated`, badge it "derived".
- **Before/after compare:** two framed canvases. Reuse the uploaded page's overlay technique
  (`maskToCloudAlpha` + separable `blur` + `screen` composite, lines 132–175) but:
  - render over a **bundled neutral terrain base** instead of live Sentinel-2 — bundle one
    small basemap thumbnail per city under `site/lab/data/base/<city>.jpg` (6 files), OR draw a
    dark canvas terrain gradient if we prefer zero external assets (decide at build time;
    default: 6 bundled thumbnails for a real-place feel);
  - **Before** = original mask over base; **After** = generated mask (for the selected target)
    over base. Re-tint cloud RGB cooler/luminous to match the dark theme (as done in
    `js/clouds.js`).
- **Metric cards:** Temperature (predicted before → after), Radiation (before → after),
  Cloud cover, plus temperature-error-to-target — **all read from `predictions.json`**, no
  heuristic arithmetic. Show the RL `actions` count ("plan: N cloud edits").
- **lab.js:** load `index.json` + `predictions.json`, populate selectors, on change/slider pick
  the nearest precomputed target entry, draw before/after, fill metric cards. Pure static fetch
  of bundled files (no API calls). Keep DOM-id + small-helper style consistent with `sim.js`.

## Part C — Wiring & cache-bust
- Add a nav link to `/lab/` in `site/index.html` and `site/simulation/index.html`.
- `site/lab/index.html`: dark `theme-color`, dark/yellow favicon, fonts, `?v=11`-style cache
  param on its own assets (lab.css/lab.js start at `?v=1`).
- No changes to `functions/`, `worker/`, `openkbs.json`, or the deployed Lambda.

## Critical files
- Read/extend: `cloud_rl/evaluate.py`, `cloud_rl/cloud_rl/targeting.py`,
  `cloud_rl/cloud_rl/first_model_reward.py` (radiation/temperature outputs),
  `cloud_rl/cloud_rl/{models,actions,dataset}.py`, `cloud_rl/configs/default.yaml`.
- New: `cloud_rl/export_demo.py`; `site/lab/index.html`, `site/lab/lab.css`, `site/lab/lab.js`;
  `site/lab/data/*` (generated).
- Edit: `site/index.html`, `site/simulation/index.html` (nav links).
- Reuse from the uploaded demo: overlay math (`maskToCloudAlpha`, `blur`, screen composite).

## Verification
1. **Precompute (needs checkpoints):** run `python -m cloud_rl.export_demo --checkpoint … 
   --reward-checkpoint … --data-root dataset_out --split all --out-dir site/lab/data`; confirm
   `predictions.json` + masks land in `site/lab/data/` and values look sane (after-temp moves
   toward target; more cloud → lower radiation).
2. **Local UI:** open `site/lab/index.html` via agent-browser; verify dark theme, city/date
   selectors, target slider drives a visible cloud change in the After canvas, metric cards
   update from real JSON, radiation field updates (badged if derived), no console/network errors
   (all data served from `site/lab/data/`).
3. **Deploy:** commit (`feat: cloud-RL before/after demo page with real model outputs`), push via
   SSH to `origin main`, `openkbs site deploy`; hard-refresh live `/lab/` and confirm it loads
   the bundled data on CloudFront.

## Out of scope
- No live/per-request inference, no Python endpoint, no new infra (static precompute only).
- No changes to the WRF simulation studio behavior or the Lambda API.
- Radiation is shown as a **predicted outcome** of the temperature-driven plan, not separately
  optimized; full "set any target and re-optimize" is not in scope.
