# Fetch real Sentinel-2 imagery for the Cloud RL demo page

## Context
The Cloud RL showcase page (`site/cloud-rl/`) currently builds its "before" cloud image
with `drawMask()` in `site/model-lab/model-contracts.js` — seeded random gaussian blobs,
not real data. The user wants the page to fetch the **latest Sentinel-2 satellite imagery
around the specified place + date** and use that as the real observation image.

Sentinel-2 comes from Copernicus Data Space (CDSE) via an OAuth `client_credentials` flow
and the Process API. This **cannot run in the browser** (the client secret must stay private,
and CDSE blocks browser CORS), so it needs a server-side proxy. The repo already has a fully
working reference implementation in `server.mjs` (local dev server) that we will port into the
deployed Lambda. The user has confirmed they have Copernicus credentials.

Image render: the **cloud mask** (CLM/SCL, binary white-on-dark) — matches the page's
"Before / after cloud masks" framing and what the RL model consumes. (A true-color/toggle
variant is a trivial follow-up since both evalscripts already exist in `server.mjs`.)

## Plan

### 1. Add a Sentinel-2 proxy to the deployed Lambda
New file `functions/api/lib/sentinel2.mjs` — port these from `server.mjs`:
- `getAccessToken()` (server.mjs:986) — CDSE OAuth, with in-module token cache.
- `buildSentinel2ProcessRequest(bbox, date, mode, width, height)` (server.mjs:691) —
  `sentinel-2-l2a`, `mosaickingOrder: "leastCC"`, day-long `timeRange` from the date.
- `buildSentinel2Evalscript(mode)` (server.mjs:843) — reuse the `cloud_mask` branch
  (CLM/SCL → white) as default; keep `natural_color` available for the future toggle.
- `fetchSentinel2Png({ bbox, date, mode, width, height })` — calls the Process API
  (`https://sh.dataspace.copernicus.eu/process/v1`, defaults mirror server.mjs:11-14),
  returns a `data:image/png;base64,...` string.

Edit `functions/api/index.mjs`:
- Add `case 'sentinel2':` to the dispatch (after the existing cases, ~line 86). It reads
  `{ bbox, date, mode, width, height }`, calls `fetchSentinel2Png(...)`, and returns
  `json(200, { image_data_url, mode, date })`. On missing creds / no scene, return a clear
  error so the frontend can fall back gracefully.
- Add the `'sentinel2'` action to the `available` list in the default case.

### 2. Credentials (Lambda env vars)
Set `COPERNICUS_CLIENT_ID` and `COPERNICUS_CLIENT_SECRET` via the studio's secure secret
input (never in chat / never read `.env`), which writes `functions/api/.env`. The CLI injects
them on `openkbs fn deploy api`. `getAccessToken()` reads `process.env.COPERNICUS_CLIENT_ID/SECRET`.

### 3. Frontend: call the proxy and render the real image
`site/model-lab/model-contracts.js`:
- Extend `getConfig()` (line 33) with an `apiBase` (default the deployed `api` Lambda URL,
  fetched via `openkbs fn list` at implementation time; overridable via
  `window.ChillOutModelConfig.apiBase` / localStorage).
- Add `async fetchSentinelImage(sample, mode)` — POST `{ action:'sentinel2', bbox: sample.bbox,
  date: sample.date, mode }` to `apiBase`, return the `image_data_url`. Reuse the existing
  `sample.bbox` already produced by `makeBbox()` in `fetchWeatherSample()` (line 200).
- Export `fetchSentinelImage` in the public object (~line 434).

`site/cloud-rl/cloud-rl.js`:
- In `buildSample()` (line 63), when no mask is uploaded, replace the `C.drawMask(...)` call
  with `await C.fetchSentinelImage(sample, "cloud_mask")` → `C.loadImageToCanvas(url, originalCanvas)`.
  If the fetch fails (no creds, no cloud-free scene, error), fall back to `C.drawMask(...)`
  and surface a status note, so the page never hard-breaks.
- Update `originalCaption` to indicate "Sentinel-2 cloud mask · <date>" on success.

`site/cloud-rl/index.html`: minor copy tweak only — update the lead/caption text that says the
mask is "browser-generated" to reflect real Sentinel-2 fetch (lines ~38-42, 109).

## Critical files
- `server.mjs` (reference, read-only) — getAccessToken:986, buildSentinel2ProcessRequest:691,
  buildSentinel2Evalscript:843, proxyCopernicusProcess:386, defaults:11.
- `functions/api/index.mjs` — add `sentinel2` action.
- `functions/api/lib/sentinel2.mjs` — NEW proxy helper.
- `site/model-lab/model-contracts.js` — `getConfig`, new `fetchSentinelImage`, `makeBbox`(reuse).
- `site/cloud-rl/cloud-rl.js` — `buildSample` wiring + fallback.

## Verification
1. `openkbs fn deploy api` (with the two Copernicus env vars set). Confirm success.
2. Smoke-test the action:
   `openkbs fn invoke api -d '{"action":"sentinel2","bbox":[24.6,42.0,24.9,42.27],"date":"2026-06-15","mode":"cloud_mask"}'`
   → expect `{ image_data_url: "data:image/png;base64,..." }`.
3. Point `apiBase` at the Lambda URL, `openkbs site deploy`.
4. With agent-browser: open the live `/cloud-rl/` page, pick Plovdiv + a recent clear date,
   click "Build sample" → the original frame shows a real Sentinel-2 cloud mask (not blobs).
   Verify the fallback path by choosing a date with no scene (should degrade with a note).

## Out of scope
- The real-model precompute work (policy.pt/reward.pt → masks+JSON) — separate paused thread.
- True-color / mask toggle UI — easy follow-up; both evalscripts already exist.
- Temperature page changes.
