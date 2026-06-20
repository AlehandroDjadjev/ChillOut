# ChillOut — Make simulations actually run, show progress, re-theme (warm light)

## Context

The WRF foundation is proven: `infra/run_test_case.sh` now produces a real `wrfout`
(7 time levels, 184 fields) over Stara Zagora through the real chain
geogrid→ungrib→metgrid→real→wrf. The web app (static `/simulation` editor + Node Lambda
API + Postgres queue) is deployed and a submission creates a job row. **The gap:** the
Python worker stages are `NotImplementedError` stubs, so a submitted scenario never
completes — and the UI gives no live feedback while it waits. The user also finds the
ice-cyan theme cold.

This round delivers three things:
1. **It actually works.** Wire the worker to the proven WRF chain and run it live on this
   VM, so a submitted scenario runs baseline + cloud-modified WRF and returns real results.
2. **Visible progress.** Loading/spinner animations + an animated status pipeline so the
   user always knows something is happening.
3. **Warm-light re-theme** across the site (cream background, terracotta primary, green
   success), replacing the ice-cyan dark theme.

**"Works on all machines" framing:** WRF cannot run in a browser or on every laptop (heavy
Fortran/MPI). The *web app* already works for any user on any device (it's on AWS). WRF runs
centrally on a **worker VM** — we wire+run it here. Any user, any machine, hits the same
site and gets results from the central worker. The `infra/` scripts already reproduce the
worker on any Debian/Ubuntu VM; we add a launch script + docs so it's repeatable.

**Decisions:** Results this round = real **metrics + summary** (PNG delta maps are an
immediate fast-follow). First worker runs are **single-domain** with a tunable safety budget
(grid size + run hours) so jobs complete in minutes; the actual config used is recorded in
the result for honesty. GFS is auto-selected from the latest available NOMADS cycle (real
data is only retained ~10 days), and the cycle used is reported in the summary.

---

## A. Worker — wire the proven chain (the core of "it works")

Mirror `infra/run_test_case.sh` (the proven sequence) into Python.

- **`worker/wrf_runner.py`** — replace the three stubs:
  - `run_wps(run_dir, scenario)`: fetch GFS subset (port the NOMADS auto-cycle loop from
    `run_test_case.sh` lines 64–106, box from scenario domain center/size), write
    `namelist.wps`, symlink WPS exes + `Vtable.GFS`, run `geogrid.exe` → `link_grib.csh` →
    `ungrib.exe` → `metgrid.exe`, asserting each output (`geo_em.d01.nc`, `FILE:*`,
    `met_em.d01.*.nc` with ≥2 time levels). Return detected `num_metgrid_levels`.
  - `run_real(run_dir, nmet)`: symlink `real.exe`/`wrf.exe` + run-dir TBL/RRTMG/DATA files
    (per script lines 217–223), patch `num_metgrid_levels` into `namelist.input`, run
    `mpirun -np N ./real.exe` (fallback serial), assert `wrfinput_d01` + `wrfbdy_d01`.
  - `run_wrf(run_dir, log_path)`: `mpirun -np N ./wrf.exe`, assert `wrfout_d01_*` with
    ≥2 time levels, return its path.
  - Keep `assert_binaries()`, `_run()`, `WrfUnavailable`, `WrfRunError` as-is.
  - Rename `wrfout` after baseline (e.g. `wrf_baseline.nc`) before injecting + second run so
    the candidate run doesn't clobber it.

- **`worker/wrf_case_builder.py`** — bring `build_namelists()` up to the **proven** namelist
  (currently missing groups). Add `&dynamics`, `&bdy_control`, `&namelist_quilt`,
  `sfcp_to_sfcp=.true.`, `interval_seconds=10800`, `geog_data_res='lowres'`, the safe
  date-window handling (model window ≥ one 3h interval — the bug fixed in the shell script),
  and a `__NMET__` placeholder for `num_metgrid_levels` that `run_real` fills. Single domain;
  honor scenario center/resolution/size/duration but clamp via env budget
  (`WORKER_MAX_CELLS`, `WORKER_MAX_HOURS`) and record actual values.

- **`worker/cloud_injector.py`** — real `inject_cloud(wrfinput_path, scenario)` with
  `netCDF4`+`numpy`:
  - Open `wrfinput_d01`; read `XLAT`,`XLONG`,`PH`,`PHB`,`HGT`,`QCLOUD`,`QICE`,`QVAPOR`.
  - Column mask via `matplotlib.path.Path(polygon).contains_points(...)` (matplotlib is
    already a dep — no shapely). If `target_polygon` is empty, fall back to a disk around
    `domain_center` and note it in the result.
  - Height AGL per level = `(PH+PHB)/9.81 - HGT`; select levels in `[base, top]_m_agl`.
  - Set `QCLOUD = liquid_water_mixing_ratio * cloud_fraction`,
    `QICE = ice_mixing_ratio * cloud_fraction` in masked cells/levels; bump `QVAPOR` toward
    saturation. Write back in-place.

- **`worker/postprocessor.py`** — real `build_result(baseline, candidate, scenario)`:
  open both wrfout, compute Δ of `objective.target_variable` (T2/RAINNC/SWDOWN/TSK) over the
  target polygon/disk and `target_time_window_hours`; return
  `{ metrics:[{label,value}], summary, maps:[] }` (maps empty this round). Summary states
  the GFS cycle used, actual domain/run config, and honest caveats ("simulated to produce …
  with these assumptions").

- **`worker/config.py`** — confirm `SIMULATION_STORAGE_PATH`, `MPI_RANKS`, add
  `WORKER_MAX_CELLS`/`WORKER_MAX_HOURS` budget knobs. `worker/requirements.txt` already has
  psycopg2/netCDF4/numpy/matplotlib.

- **`worker/run_worker.sh`** (new) — source `/opt/wrf/wrf_env.sh`, export `DATABASE_URL`
  from `openkbs postgres connection`, `exec python3 run_simulation.py`. One-command,
  repeatable launch on any worker VM.

- **Run it live here:** start the daemon on this VM (nohup/background) so queued jobs
  process automatically during testing.

## B. Frontend — loading & progress feedback (`site/simulation/`)

- **`sim.css` + `css/animations.css`**: add a reusable `.spinner` (CSS `@keyframes` spin),
  an active-stage **pulse/shimmer**, a `.working-bar` with an indeterminate sweep, and
  done/active/failed pipeline styles.
- **`sim.js`** (functions at lines 185–251):
  - Submit button → inline spinner + "Submitting…" disabled state (already toggles
    `disabled`; add spinner + label swap).
  - `renderPipeline`: animate the current stage (pulsing dot), green checks for done, red
    for failed.
  - `renderResult` (running branch): replace static placeholder with a per-status message
    ("Fetching GFS…", "Running baseline model…", "Injecting cloud + candidate run…") + a
    spinner + a **live elapsed timer** (mm:ss since submit).
  - `poll`: keep 4s cadence; drive the elapsed timer; stop on completed/failed.
  - Tabs (Maps/Time-series/Diagnostics) get clear empty/loading states ("maps coming soon").

## C. Warm-light re-theme (`site/css/styles.css` `:root`, `sim.css`)

Replace the dark ice-cyan tokens with the approved warm-light palette and update both the
landing page and the simulation studio:

```
--bg #fbf4ec  --bg-2 #f5ebdf  --surface #fffaf3  --surface-2 #f3e7d8
--ink #2a211c  --ink-soft #6b5b4f  --ink-mute #9c8b7d
--primary (was --cool) #e2553a (terracotta)   --primary-deep #b8402a
--success #3a9d5d (green)   --accent #ffb152 (amber)
--line rgba(42,33,28,.10)  --glass-bg rgba(255,255,255,.55)
```

- Rename/repoint `--cool*` usages to `--primary*`; map status "completed" to `--success`
  (green), "failed" to a warm red, active stages to terracotta/amber.
- Invert glass cards + grain overlay for a light background; keep Fraunces/Schibsted fonts.
- Update `site/index.html` hero/nav/buttons + the "Launch Simulator" CTA for the light theme.

## D. Portability + persistence

- Document worker setup for any Debian/Ubuntu VM in `worker/README.md` (install → check →
  `run_worker.sh`) and note `infra/` already reproduces WRF on aarch64 **and** x86_64.
- **Commit** `infra/`, `worker/`, `spec/`, and the frontend changes so the project can be
  cloned onto another machine (currently all untracked).

---

## Critical files
- Worker: `worker/wrf_runner.py`, `worker/wrf_case_builder.py`, `worker/cloud_injector.py`,
  `worker/postprocessor.py`, `worker/config.py`, new `worker/run_worker.sh`.
- Reference (proven, do not break): `infra/run_test_case.sh`.
- API contract (unchanged): `functions/api/lib/schema.mjs`, `functions/api/lib/db.mjs`,
  `functions/api/index.mjs` (result jsonb = `{metrics,summary,maps}`; worker writes Postgres
  directly via `worker/db.py`).
- Frontend: `site/simulation/sim.js`, `site/simulation/sim.css`, `site/css/styles.css`,
  `site/css/animations.css`, `site/index.html`.

## Verification (end-to-end, no faking)
1. Start the worker daemon on this VM (`worker/run_worker.sh`).
2. Submit a real scenario via the API (curl `createSimulation`) and watch the row move
   `created→…→completed` (`getResults` shows real `metrics`), or `failed` with an honest
   error. Confirm two distinct wrfout files (baseline vs candidate) and a non-zero Δ.
3. `agent-browser`: load `/simulation`, draw a polygon, submit; confirm the spinner +
   animated pipeline + elapsed timer, then real metrics + summary on completion.
4. Confirm the warm-light theme on both `/simulation` and the landing page in a browser.
5. Deploy site (`openkbs site deploy`); commit everything before each deploy so versions tag.
6. Fast-follow (next round): matplotlib PNG delta maps → S3 → Maps tab.
