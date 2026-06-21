# Run uploaded RL weights on the Sentinel-2 mask (upload → backend inference → render)

## Context
The Cloud RL showcase (`site/cloud-rl/`) now fetches a **real Sentinel-2 cloud mask**
for a place/date (done, deployed v4). The next step the user asked for: a page where you
**upload a checkpoint of the latest RL weights**, click **Build/Run**, and the backend runs
that model on the Sentinel-2 mask, returning the generated mask + action data to the frontend.
**The weights and inference must run server-side**, not in the browser.

Hard constraint discovered: **PyTorch cannot run in the OpenKBS Lambda** (Node 24, 512MB, 30s).
The repo already uses a proven pattern for heavy compute: the Lambda enqueues a job in Postgres,
and a **Python daemon on a worker VM** (`worker/run_simulation.py`, `worker/db.py`) polls and runs
it. We mirror that pattern for RL inference. The user confirmed the **worker-VM daemon** approach.

Decisions (user picked the daemon; the other two use the recommended defaults — flagged so they
can be redirected):
- **Inference host:** new Python torch daemon on a VM/local machine with torch (reuse `cloud_rl/.venv`).
- **Normalization stats:** `stats.json` does **not** exist in the repo but `dataset.py`/`rewards.py`
  require `feature_mean/std` + `target_temp_mean/std`. The page will let the user **upload
  `stats.json` alongside the `.pt`** so normalization matches the weights.
- **Output scope:** run the **policy only** (`model.deterministic` + `rasterize_actions`) → generated
  cloud mask, action tokens, property maps. (Temperature prediction needs a 2nd reward checkpoint —
  out of scope for now.)

Checkpoints are large (often >6MB, exceeding the Lambda ~6MB payload limit), so the `.pt` (and
`stats.json`) are uploaded **directly to S3** via a presigned URL; only small JSON (mask data URL +
14-feature vector + target temp + S3 keys) flows through the Lambda.

## Architecture / data flow
```
Browser (site/rl-inference/)
  1. request presigned upload URL(s) from Lambda  ── action: getUploadUrl
  2. PUT checkpoint.pt (+ optional stats.json) → S3 (media/inference/<uuid>/...)
  3. fetch Sentinel-2 mask + weather feature vector (reuse existing helpers)
  4. enqueue job ──────────────────────────────── action: createInference
  5. poll ─────────────────────────────────────── action: getInference  (every ~3s)
        │
Lambda (functions/api) — never runs torch; only S3 + Postgres
        │
Postgres  inference_jobs  (status: queued → running → completed/failed)
        │
Python torch daemon on VM (worker/rl_inference_worker.py)
  claim job → openkbs storage download checkpoint/stats → build single-sample batch
  → model.deterministic → rasterize_actions → encode masks to base64 PNG
  → write result JSONB → mark completed
        │
Browser renders generated mask + actions + property maps
```

## Plan

### 1. Backend — new Postgres table + Lambda actions
A **separate table** (not `simulations`) so the WRF worker never claims RL jobs.

New `functions/api/lib/inference_db.mjs` (mirror `lib/db.mjs` style, reuse its `pg` pool pattern):
- `ensureInferenceSchema()` — create `inference_jobs`:
  `id uuid pk default gen_random_uuid(), status text default 'queued',
   checkpoint_key text, stats_key text, mask_data_url text, raw_features jsonb,
   target_temp double precision, place text, date text, result jsonb, error text,
   created_at timestamptz default now(), updated_at timestamptz default now()`
  plus index on `(status, created_at)`.
- `createInferenceJob({checkpoint_key, stats_key, mask_data_url, raw_features, target_temp, place, date})` → row.
- `getInferenceJob(id)` → row.

Edit `functions/api/index.mjs` — add three actions to the dispatch + to the `available` list:
- `getUploadUrl`: body `{ filename, contentType, jobUuid }`. Generate key
  `media/inference/<jobUuid>/<filename>`; call the Project API
  `POST https://project.openkbs.com/projects/${OPENKBS_PROJECT_ID}/storage/upload-url`
  (Bearer `OPENKBS_API_KEY`) — per the skill's "S3 presigned upload" pattern (no AWS SDK import).
  Return `{ uploadUrl, key }`. Frontend generates one `jobUuid` (crypto.randomUUID) and reuses it
  for both files so they share a folder.
- `createInference`: validate inputs, `await ensureInferenceSchema()`, insert via
  `createInferenceJob(...)`, return `{ id, status }`.
- `getInference`: `{ id }` → `getInferenceJob` → `{ id, status, result, error }`.

### 2. Worker — new RL inference daemon (mirrors worker/run_simulation.py)
New files under `worker/`:
- `rl_db.py` — copy `worker/db.py`'s connection/retry skeleton (`_run`, reconnect-on-disconnect),
  but operate on `inference_jobs`: `claim_next_job()` (status `queued`→`running`,
  `FOR UPDATE SKIP LOCKED`), `mark_completed(id, result)`, `mark_failed(id, err)`,
  `reclaim_stale_jobs(minutes)`.
- `rl_inference_worker.py` — daemon loop identical in shape to `run_simulation.py:main()`:
  - `process_job(job)`:
    1. `openkbs storage download media/inference/<...>/checkpoint.pt` (and `stats.json` if present)
       to a temp dir (subprocess; the VM already has the CLI — same as `run_worker.sh`).
    2. `ckpt = torch.load(path, map_location="cpu")`; `cfg = ckpt["cfg"]`.
    3. Build a **single-sample batch** from `job.mask_data_url` (decode PNG → 256×256 mask tensor)
       and `job.raw_features` (14-vector), using `stats.json` for normalization and
       `job.target_temp` → `target_temp_norm`. Fake the lookback window by repeating the mask/
       features `lookback` times and zeroing trend features (as established from `dataset.py`).
       **Reuse the obs_map assembly from `cloud_rl/cloud_rl/dataset.py`** — factor the channel-stacking
       in `CloudFolderDataset.__getitem__` into a reusable `assemble_obs_map(...)` and call it here,
       so the 34-channel layout exactly matches training (this is the main correctness risk).
    4. Build model exactly as `cloud_rl/evaluate.py:64-71`
       (`CloudActorCritic(obs_channels=1+feature_dim+1, feature_dim, max_actions, hidden_dim)`,
       `load_state_dict(ckpt["model"])`, `eval()`).
    5. `sampled = model.deterministic(obs_map, features, target_temp_norm)`;
       `rast = rasterize_actions(original_mask, sampled["op"], sampled["params"])`
       (same calls as `evaluate.py:92-93`).
    6. Encode `original_mask`, `rast["generated_mask"]`, and the 5 `property_maps` to base64 PNG
       data URLs (reuse `evaluate.py:save_mask` logic, but to an in-memory buffer).
       Build `actions = actions_to_jsonable(sampled["op"][0], sampled["params"][0])`.
    7. `rl_db.mark_completed(id, {generated_mask_data_url, property_maps:{...}, actions, feature_dim})`.
  - reuse the same try/except discipline: torch/load errors → `mark_failed` with a clear message,
    daemon stays alive.
- `run_rl_worker.sh` — like `worker/run_worker.sh` but simpler (no WRF env): resolve `DATABASE_URL`
  via `openkbs postgres connection`, activate `cloud_rl/.venv`, `exec python3 rl_inference_worker.py`.
  Needs `psycopg2` + the `cloud_rl` package importable (run from repo root or set `PYTHONPATH`).

### 3. Frontend — new page site/rl-inference/
New `site/rl-inference/index.html` + `rl-inference.js` (reuse `../model-lab/model-lab.css`,
`model-contracts.js`, `../css/styles.css`), styled to match the existing lab pages. UI:
- Place + date + target-temp inputs (reuse the cloud-rl form pattern).
- **File inputs:** checkpoint `.pt` (required) and `stats.json` (optional).
- "Build / Run inference" button. Status line + before/after mask frames + actions/metrics panel
  + JSON payload panel (same visual language as `site/cloud-rl/index.html`).

`rl-inference.js` flow (reuse `model-contracts.js` helpers `geocodePlace`, `makeBbox`,
`fetchWeatherSample`, `fetchSentinelImage`, `loadImageToCanvas`, and `DEFAULT_API_BASE`):
1. On Run: `const jobUuid = crypto.randomUUID()`.
2. `fetchWeatherSample(place,date)` → `sample` (gives `sample.bbox`, `sample.date`, `raw_features`).
3. `fetchSentinelImage(sample,'cloud_mask')` → mask data URL; draw into the "original" canvas.
4. For checkpoint (+ stats): `getUploadUrl` → `fetch(uploadUrl,{method:'PUT',body:file})`.
5. `createInference` with `{ checkpoint_key, stats_key, mask_data_url, raw_features, target_temp, place, date }` → `id`.
6. Poll `getInference` every ~3s until `completed`/`failed` (mirror `site/simulation/sim.js:poll`).
7. On completed: render `generated_mask_data_url` into the "generated" frame, list `actions`,
   show property maps + JSON. On failed: surface `error`.

Add `fetchUploadUrl`, `createInference`, `getInference` thin wrappers to `model-contracts.js`
(reuse the existing `api`/POST helper + `DEFAULT_API_BASE`), and export them.
Add a nav link to the new page from `site/cloud-rl/index.html`.

## Critical files
- `functions/api/index.mjs` — add `getUploadUrl`, `createInference`, `getInference` actions.
- `functions/api/lib/inference_db.mjs` — NEW (mirror `lib/db.mjs`).
- `functions/api/lib/db.mjs` — reference for `pg` pool + `ensureSchema` style (read-only).
- `worker/rl_db.py`, `worker/rl_inference_worker.py`, `worker/run_rl_worker.sh` — NEW (mirror
  `worker/db.py`, `worker/run_simulation.py`, `worker/run_worker.sh`).
- `cloud_rl/cloud_rl/dataset.py` — factor out `assemble_obs_map(...)`; load `stats.json` (lines ~327-346).
- `cloud_rl/evaluate.py` — reference inference flow (load ckpt, build model, deterministic,
  rasterize_actions, save_mask) — lines 41-93, 21-24.
- `cloud_rl/cloud_rl/actions.py` — `rasterize_actions`, `actions_to_jsonable`.
- `site/rl-inference/index.html`, `site/rl-inference/rl-inference.js` — NEW page.
- `site/model-lab/model-contracts.js` — add `fetchUploadUrl`/`createInference`/`getInference` wrappers.
- `site/cloud-rl/index.html` — add nav link to the new page.

## Verification
1. `openkbs fn deploy api` → confirm `inference_jobs` is created (call `createInference` with dummy
   keys, then `getInference` returns `queued`).
2. Start the daemon locally: `bash worker/run_rl_worker.sh` (uses `cloud_rl/.venv` torch). Confirm
   it logs "starting" and connects to the queue.
3. End-to-end with agent-browser on the live `/rl-inference/` page: upload a real `.pt` (+ `stats.json`),
   pick Plovdiv + a recent date, click Run. Expect: original frame = Sentinel-2 mask, then after the
   daemon finishes (~seconds), generated frame = model output, actions list + property maps populated.
4. Failure paths: upload a corrupt/empty `.pt` → job ends `failed` with a clear error shown on the page;
   omit `stats.json` → page warns (and either blocks or uses identity normalization, per final choice).
5. Commit, then `openkbs site deploy` (auto-tags next version).

## Out of scope
- Temperature prediction (needs a 2nd reward/temperature checkpoint).
- True-color/mask toggle on the Sentinel-2 fetch (separate easy follow-up).
- Auto-provisioning the worker VM; the user runs the daemon (matches the existing WRF worker model).
- Rotating the leaked Copernicus secret (recommended separately).
