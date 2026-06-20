# ChillOut — WRF Simulation Engine (`infra/`)

This directory installs and verifies the **actual** simulation engine behind ChillOut:
the **WRF** (Weather Research & Forecasting) model and its preprocessor **WPS**. Nothing
here is mocked — `run_test_case.sh` runs the real `geogrid → ungrib → metgrid → real → wrf`
chain and only reports success if a genuine `wrfout` NetCDF file is produced.

> WRF is heavy native software (Fortran + MPI). It **cannot** run inside the OpenKBS Lambda
> (`functions/api`). It runs on a real VM/EC2. These scripts target **Debian 12 (bookworm)**
> and are verified on **aarch64** (also work on x86_64).

## Contents

| File | Purpose |
|------|---------|
| `install_wrf.sh` | Install toolchain + libs, compile WRF (`em_real`) and WPS. Idempotent. |
| `check_wrf_install.sh` | PASS/FAIL audit of compilers, NetCDF/MPI/Jasper, and all 5 binaries. |
| `run_test_case.sh` | Minimal real-data run over Stara Zagora, Bulgaria. Asserts a real `wrfout`. |
| `scenario.example.json` | The frontend ↔ backend scenario contract (the product's input schema). |

## Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `WRF_ROOT` | `/opt/wrf` | Install root (holds `WRF/`, `WPS/`, `deps/`, `src/`, `geog/`, `cases/`). |
| `WRF_GEOG_PATH` | `$WRF_ROOT/geog` | WPS static geography datasets. |
| `SIMULATION_STORAGE_PATH` | `$WRF_ROOT/cases` | Per-simulation working dirs + outputs. |
| `WRF_VER` / `WPS_VER` | `4.6.1` / `4.6.0` | Source versions. |
| `JOBS` | `nproc` | Parallel compile jobs. |

`install_wrf.sh` writes `$WRF_ROOT/wrf_env.sh`; the other two scripts source it
automatically, so you normally don't need to export anything.

## Run order

```bash
# 1. Install + compile (one-time; ~20–40 min depending on cores/network)
bash infra/install_wrf.sh

# 2. Verify everything is in place
bash infra/check_wrf_install.sh        # exits non-zero if anything is missing

# 3. Run the minimal end-to-end WRF test
bash infra/run_test_case.sh            # downloads small GFS subset + geog, then runs WRF
```

## Where outputs land

```
$WRF_ROOT/
  install.log                 # full build log
  WRF/  WPS/                  # compiled source trees (symlinks to src/…)
  geog/                       # WPS geography data (downloaded once)
  cases/test_stz/             # the test simulation working dir
    namelist.wps  namelist.input
    geo_em.d01.nc             # geogrid output
    FILE:YYYY-MM-DD_HH        # ungrib intermediates
    met_em.d01.*.nc           # metgrid output
    wrfinput_d01  wrfbdy_d01  # real.exe output
    wrfout_d01_*              # <-- the WRF simulation result (success criterion)
    run.log  rsl.error.*      # logs
```

Inspect the result:

```bash
ncdump -h $WRF_ROOT/cases/test_stz/wrfout_d01_*   # T2, U10, V10, RAINNC, QCLOUD, …
```

## Common errors & fixes

| Symptom | Cause | Fix |
|--------|-------|-----|
| `netcdf.mod not found` | `libnetcdff-dev` missing | re-run install; if apt lacks it, build netcdf-fortran from source (see below). |
| WRF compiles but `wrf.exe` absent | gfortran ≥10 argument-mismatch errors | install script injects `-fallow-argument-mismatch`; check `install.log` for the real error. |
| `could not find GNU dmpar option` | WRF `configure` has no entry for this arch | add an aarch64 GNU block to `arch/configure.defaults`, or use WRF's CMake build. |
| WPS `ungrib` GRIB2 errors | Jasper not linked | ensure `libjasper-dev` is installed; `JASPERLIB`/`JASPERINC` are set by the script. |
| `no usable GFS cycle found` | NOMADS retention (~10 days) or outage | wait/retry, or use the idealized fallback below. |
| `mpirun` oversubscribe error | more MPI ranks than cores | the test uses ≤4 ranks; lower `-np` if needed. |

### Idealized fallback (if live GFS/GRIB is unavailable)

To prove the toolchain without external data, build and run an idealized case (LES), which
needs no WPS/GFS:

```bash
cd $WRF_DIR && ./compile -j $(nproc) em_les
cd test/em_les && ./ideal.exe && mpirun -np 4 ./wrf.exe
ls wrfout_d01_*        # idealized result
```

## How this fits the product

`scenario.example.json` is the contract the frontend produces and the backend validates.
The WRF worker (later phase) converts a validated scenario into `namelist.wps` /
`namelist.input` (never the frontend), runs the baseline (`bad_scenario`) and the
cloud-modified (`good_scenario`) simulations, then post-processes the delta. The frontend
never edits WRF namelists directly.
