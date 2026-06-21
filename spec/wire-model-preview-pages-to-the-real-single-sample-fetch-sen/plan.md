# Wire model preview pages to the real single-sample fetch (Sentinel-2 mask + Copernicus stats + Open-Meteo)

## Context
The model preview pages (`site/cloud-rl/`, `site/model-temperature/`, `site/rl-inference/`) currently
build their 14-feature input with `fetchWeatherSample` in `site/model-lab/model-contracts.js`. That
function only calls the Open-Meteo **forecast** API and then **heuristically fakes** the satellite
fields (`s5p_*`, `s3_*`) — its own `source_notes` admit this ("Derived preview fields ... until
Sentinel endpoints are wired here"). The user wants the preview pages to instead use the **same real
fetch the dataset builder uses**, for a **single chosen city + date**, so the numbers match training.

The dataset builder (`build_temperature_dataset.py`) gathers three things per (city, date):
1. **Sentinel-2 cloud mask** → server `/api/process` (`download_satellite_mask`). The Lambda already
   ports this as the `sentinel2` action (`functions/api/lib/sentinel2.mjs`). ✓ reuse as-is.
2. **5 `s5p_*` + 3 `s3_*` scientific features** → server `/api/statistics` (`fetch_satellite_series`),
   which calls the **Copernicus Statistics API** (OAuth, client secret). This is **not** in the Lambda
   yet and must move server-side (the browser can't hold the secret / is CORS-blocked).
3. **6 weather features + observed temp** → **Open-Meteo archive** API (`fetch_open_meteo_daily`),
   daily aggregation. (The "ERA5" data is Open-Meteo archive, not the Copernicus ERA5 stats endpoint.)

Decisions confirmed with the user:
- **Missing satellite data → widen the date window.** Single dates often have no Sentinel-5P/3 pass;
  query a window around the date and take the nearest available daily aggregate (mirrors how the
  Sentinel-2 mask already uses a ±day window in `sentinel2.mjs`).
- **Weather source: archive, fall back to forecast.** Use Open-Meteo archive (matches training); if the
  date is too recent for the archive (~5-day lag), fall back to the forecast API so the page still works.

## Architecture / data flow
```
Browser (preview pages)
  C.fetchRealSample(place, date)  ──► Lambda action "buildSample"
        │                                   │
        │   (mask, separately)              ├─ 8× Copernicus Statistics API (s5p_*, s3_*) — OAuth, windowed
        │   C.fetchSentinelImage ──► "sentinel2" action (existing)
        │                                   └─ Open-Meteo archive (6 weather feats + temp), forecast fallback
        ▼
  connected sample { inputs, feature_vector(14), observed_temperature_c, bbox, source_notes }
```

## Plan

### 1. Backend — shared Copernicus auth + statistics helper
- **`functions/api/lib/sentinel2.mjs`** — export the currently-private `getAccessToken` (add `export`)
  so the stats helper can reuse the same cached OAuth token (avoid duplicate token logic/cache).
- **NEW `functions/api/lib/sentinel_stats.mjs`** — port the statistics path from `server.mjs`:
  - field metadata `S5P_FIELDS` (cloud_fraction, cloud_optical_thickness, cloud_top_height,
    cloud_top_pressure, aerosol_index) and `S3_FIELDS` (humidity, sea_level_pressure, water_vapour)
    — copy the relevant entries from `server.mjs:20-86`.
  - `buildStatisticsRequest` / `buildStatisticsEvalscript` / `getStatistics{DataType,DataFilter,Resolution}`
    (`server.mjs:731-796, 1278-1335`), and the request/parse loop from `fetchCopernicusStatisticsSeries`
    (`server.mjs:467-543`) hitting `https://sh.dataspace.copernicus.eu/statistics/v1`.
  - Export `fetchSatelliteStat({ source, field, date, bbox, windowDays })` → query
    `[date-windowDays, date+windowDays]` with `aggregationInterval P1D`, return the **mean of the
    nearest non-empty day** to `date` (widen-window behavior), or `null` if the whole window is empty.

### 2. Backend — new `buildSample` Lambda action (`functions/api/index.mjs`)
Body `{ place, lat, lon, date, bbox?, windowDays? }`. Steps:
1. Resolve `bbox` (reuse a 20 km box like `bbox_from_center`/frontend `makeBbox` if not supplied).
2. **Satellite stats** — `Promise.all` over the 8 (source, field) pairs calling
   `fetchSatelliteStat(...)` (token is cached after the first). Map results to the 8 feature names.
3. **Weather** — fetch Open-Meteo **archive** daily for `date` (the 6 weather fields + `temperature_2m_mean`);
   if the date is missing/too recent, fetch the **forecast** API and aggregate that day. Mirror the
   daily-aggregation in `aggregate_hourly_weather`/`fetch_open_meteo_daily` for the 6 features:
   wind_speed_10m_mean, wind_direction_10m_dominant (circular mean), shortwave_radiation_sum,
   precipitation_sum, cloud_cover_mean, surface_pressure_mean.
4. Assemble `inputs` keyed by the 14 `FEATURE_NAMES` + `feature_vector` in order; return the same sample
   shape `fetchWeatherSample` returns today (sample_id, city, lat/lon, bbox, date, observed_temperature_c,
   inputs, feature_vector) plus `source_notes` recording which fields were real vs. fallback and the
   window offset used. Add `buildSample` to the action dispatch + `available` list.

### 3. Frontend — replace the heuristic with the real fetch (`site/model-lab/model-contracts.js`)
- Add `async fetchRealSample(place, date)` → POST `{ action: "buildSample", ... }` to `apiBase`, returns
  the connected sample; export it on `window.ChillOutContracts`.
- Rework `fetchWeatherSample(place, date)` to **call `fetchRealSample` first**; only if it throws/returns
  incomplete data, fall back to the existing Open-Meteo-forecast+heuristic body (kept as the safety net,
  with `source_notes` clearly marking preview mode). This way **all three pages get real data with no
  per-page change** (they all call `fetchWeatherSample`). Bump the `?v=` query on the script tags.

### 4. Wire + verify the pages
The pages already consume `fetchWeatherSample` + `fetchSentinelImage`, so no structural change is needed
in `site/cloud-rl/cloud-rl.js`, `site/model-temperature/temp-model.js`, or `site/rl-inference/`. Confirm
each page's status line surfaces the new `source_notes` (real vs. fallback) so the user can see when the
data is live satellite vs. estimated.

## Critical files
- `functions/api/lib/sentinel2.mjs` — export `getAccessToken` (shared OAuth token cache).
- `functions/api/lib/sentinel_stats.mjs` — NEW; ported Copernicus Statistics helper + `fetchSatelliteStat`.
- `functions/api/index.mjs` — NEW `buildSample` action (stats Promise.all + Open-Meteo archive/forecast).
- `site/model-lab/model-contracts.js` — add `fetchRealSample`; make `fetchWeatherSample` prefer it.
- `server.mjs` — read-only reference for the statistics logic being ported (lines 20-117, 338-543,
  731-796, 1238-1335).
- `build_temperature_dataset.py` — read-only reference for SATELLITE_FEATURES/WEATHER_FIELDS mapping and
  daily aggregation (lines 86-128, 423-543).

## Verification
1. `openkbs fn deploy api`; smoke-test the action:
   `openkbs fn invoke api -d '{"action":"buildSample","place":"Plovdiv","lat":42.1354,"lon":24.7453,"date":"<~10 days ago>"}'`
   → expect a 14-length `feature_vector` with real (non-round-number) `s5p_*`/`s3_*` values and
   `source_notes` showing the window offset used.
2. Bump script `?v=`, `openkbs site deploy`.
3. agent-browser on the live `/cloud-rl/` page: pick Plovdiv + a date ~10 days ago, click **Preview mask**
   → original frame shows the Sentinel-2 mask, the feature table shows real satellite stats (not the old
   heuristic values), and the status line reports "live satellite" sourcing.
4. Edge cases: a very recent date (archive lag → forecast fallback path) still fills weather; a date/place
   with no satellite pass in the window → `source_notes` flags the affected fields and the page still renders.
5. Confirm `model-temperature` and `rl-inference` pages also now show the real feature vector.
6. Commit (`feat:`-prefixed), then redeploy so the version tag matches.

## Out of scope
- The RL-inference upload/worker pipeline (already implemented in a prior step; unchanged here).
- Porting the Sentinel-3 SLSTR / extra Open-Meteo fields not in the 14-feature model contract.
- Rotating the Copernicus secret (recommended separately).
