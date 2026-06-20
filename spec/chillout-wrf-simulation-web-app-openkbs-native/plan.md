# ChillOut — WRF Simulation Web App (OpenKBS-native)

## Context

ChillOut is evolving from a marketing landing page into a real product: a web app where a
user composes a cloud-modification scenario, the backend runs **WRF** (Weather Research &
Forecasting model) as the *actual* simulation engine, and the frontend visualizes the
result. No faked results — a baseline run and a cloud-modified run are compared.

The hard constraint that shapes everything: **WRF cannot run in a Lambda** (30s timeout,
512MB, no MPI/Fortran). It must run on a real VM/EC2. This dev VM (Debian 12, **aarch64**,
16 cores, 30GB RAM, 169GB free, sudo) is suitable, and network access to WRF/WPS sources,
NOMADS GFS, and WPS geog data is confirmed working.

**Decisions (this round):**
- **Architecture: OpenKBS-native** — static-site frontend + Node Lambda API + a Python WRF
  worker daemon on the VM; Postgres for the job queue/status, S3/CloudFront for outputs.
- **Scope: WRF foundation first** — install + check + ONE minimal run, get it green, then
  build API/frontend in later rounds. (Matches the spec's stated priority.)
- **First successful run: tiny real-data over Stara Zagora** through the real product chain
  geogrid→ungrib→metgrid→real→wrf. Idealized `em_les` is the documented fallback if GRIB/GFS
  blocks the first green run.

## Target architecture (OpenKBS mapping — built incrementally over later rounds)

```
Static site /simulation (HTML/JS + Leaflet via CDN)
  → POST  functions/api  (Node Lambda): validate scenario.json → write job to Postgres
                                          (status=created) + scenario to S3 → return id
WRF worker daemon (Python, on VM/EC2)
  → poll Postgres for queued jobs → build namelists → geogrid→ungrib→metgrid→real→wrf
    (baseline) → cloud injector edits wrfinput → wrf (candidate) → postprocess (PNG + summary)
    → upload to S3 → update Postgres status
Frontend polls GET status/results (Lambda reads Postgres/S3) → result viewer tabs
```

Statuses: `created → validating → preprocessing → initializing → running_baseline →
running_candidate → postprocessing → completed | failed`.

Note: this dev VM is ephemeral; the same `infra/` scripts reproduce the worker on a
persistent OpenKBS Worker EC2 / dedicated VM for production. Phase 1 just needs WRF
installed here and a manual minimal run proven.

---

## PHASE 1 — WRF/WPS foundation (THIS ROUND, executable)

All artifacts committed to the repo under `infra/`; WRF itself installs to `$WRF_ROOT`
(default `/opt/wrf`) on the VM.

### Files to create
```
infra/
  install_wrf.sh        # deps + build deps-if-needed + compile WRF(em_real) + WPS; verify
  check_wrf_install.sh  # assert compilers, netcdf, mpi, jasper, all 5 binaries executable
  run_test_case.sh      # tiny Stara Zagora real-data run; assert wrfout produced
  README.md             # exact run order, env vars, output locations, troubleshooting
  scenario.example.json # the frontend↔backend contract (user's exact schema)
```
Env vars honored: `WRF_ROOT`, `WPS_ROOT`, `WRF_GEOG_PATH`, `SIMULATION_STORAGE_PATH`.

### install_wrf.sh
1. `sudo apt-get update`, then install: gfortran, g++, make, cmake, m4, perl, tcsh,
   python3/pip, wget, curl, git, zlib1g-dev, libpng-dev, libnetcdf-dev, libnetcdff-dev,
   libhdf5-dev, mpich, libmpich-dev, libjasper-dev, netcdf-bin.
   (Earlier apt-cache "NO" results were a stale index; `apt-get update` fixes them. **Source-
   build fallbacks** documented for the classic pain points: netcdf-fortran and jasper.)
2. Consolidate NetCDF into a single prefix (`$WRF_ROOT/deps/netcdf` with symlinked
   include/ + lib/) because WRF wants one `$NETCDF` and Debian multiarch splits C/Fortran.
   Export `NETCDF`, `JASPERLIB`, `JASPERINC`, `HDF5`.
3. Download **WRF 4.6.1** + **WPS 4.6.0**; `./configure` (GNU gfortran/gcc **dmpar**) and
   `./compile em_real -j 16`; then WPS `./configure` (gfortran dmpar, GRIB2/jasper) +
   `./compile`. ARM64 builds with GCC; set known WRF env flags as needed.
4. Verify and print the 5 binaries: `main/wrf.exe`, `main/real.exe`,
   `geogrid.exe`, `ungrib.exe`, `metgrid.exe`.

### check_wrf_install.sh
Hard checks (exit non-zero on any miss): gfortran/gcc present, `nc-config`/`nf-config`
report NetCDF, `mpirun` present, jasper libs present, and each of the 5 binaries exists and
is executable. Print a clear PASS/FAIL table.

### run_test_case.sh  (tiny real-data, Stara Zagora 42.42N 25.62E)
- Single small domain (~40×40, ~10km), ~1–2h, 1 short run.
- Fetch a **small GFS subset** via the NOMADS grib-filter (subset variables + a small
  lat/lon box, 1–2 forecast hours) → keeps the download small.
- Use **low-res mandatory geog** (not the 50GB full set) at `$WRF_GEOG_PATH`.
- Run geogrid → `link_grib.csh` + ungrib (Vtable.GFS) → metgrid → real → wrf.
- **Assert** `wrfout_d01_*` exists with ≥2 time levels (`ncdump -h`); fail loudly otherwise.

### README.md
Exact order (install → check → test), env vars, where outputs land, and a troubleshooting
section (netcdf path split, jasper/GRIB2, ARM configure choice, MPI oversubscribe).

### scenario.example.json
The user's exact contract verbatim (simulation_id, region.target_polygon + domain_center,
time, domain, input_data, objective, `bad_scenario`/`good_scenario` with cloud block).

### Success criteria (do NOT fake)
WRF + WPS compile; `check_wrf_install.sh` PASSES; `run_test_case.sh` produces a real
`wrfout` with ≥2 time levels. If the environment blocks compilation/run, the scripts/README
state **exactly** what failed and the smallest fix (and we fall back to idealized `em_les`
to at least prove the toolchain).

---

## PHASES 2–7 — outline (later rounds, after Phase 1 is green)

2. **Scenario schema + validator** — JSON-schema for the contract; validator (in the Node
   Lambda) enforces polygon-in-domain, duration ≤12h, resolution/size budgets, cloud
   base<top, fraction 0–1, non-negative water/ice, runtime budget. Returns `{valid,field,reason}`.
3. **Queue + worker skeleton** — enable Postgres in `openkbs.json`; `jobs` table; Lambda
   endpoints `POST /api/simulations`, `GET .../{id}`, `/status`, `/results`, `/maps/*`;
   Python worker (`worker/run_simulation.py`, `wrf_case_builder.py`, `cloud_injector.py`,
   `wrf_runner.py`, `postprocessor.py`) polling the queue.
4. **Frontend simulation editor** — `/simulation` page: Leaflet target polygon + larger WRF
   domain, time/domain/input/objective/baseline/cloud panels with sliders + client-side
   validation; emits the exact scenario.json. (Plain HTML/JS + Leaflet CDN to stay
   OpenKBS-static; React only if we add a build step.)
5. **Cloud injector + real run wiring** — map polygon→grid mask, base/top→model levels,
   write QCLOUD/QICE/QVAPOR (and optional T) into `wrfinput`; run baseline vs candidate.
6. **Postprocessing** — deltas (T2, TSK, SWDOWN, GLW, RAIN, QCLOUD) → `summary.json` +
   PNG maps (matplotlib + netCDF4) uploaded to S3.
7. **Result viewer** — Overview / Maps / Time-series / Diagnostics tabs reading S3+Postgres;
   credibility wording ("simulated to produce…, with these risks and assumptions").

---

## Verification (Phase 1)
1. `bash infra/install_wrf.sh` completes; 5 binaries present.
2. `bash infra/check_wrf_install.sh` → all PASS.
3. `bash infra/run_test_case.sh` → real `wrfout_d01_*` with ≥2 time levels; `ncdump -h`
   confirms dimensions/variables.
4. Report results honestly, including any limitation hit and its smallest fix. Commit
   `infra/` to the repo. (No site/Lambda deploy this round — foundation only.)
