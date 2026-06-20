# ChillOut WRF Worker

The simulation engine. A long-running Python daemon that pulls scenarios from the shared
Postgres job queue (written by `functions/api`) and runs them through **real WRF/WPS** on a
VM/EC2 — never in a Lambda, never faked.

## Pipeline

```
claim job (status=created)
  -> build namelists from scenario      (wrf_case_builder.py)
  -> WPS: geogrid -> ungrib -> metgrid  (wrf_runner.run_wps)
  -> real.exe                           (wrf_runner.run_real)
  -> baseline wrf.exe                   (wrf_runner.run_wrf)
  -> inject cloud into wrfinput         (cloud_injector.py)
  -> candidate wrf.exe                  (wrf_runner.run_wrf)
  -> compare + metrics + summary        (postprocessor.py)
  -> status=completed (+ result jsonb)
```

Status vocabulary matches `functions/api/lib/schema.mjs`:
`created → preprocessing → initializing → running_baseline → running_candidate →
postprocessing → completed | failed`.

The full chain is wired and runs live — it mirrors the proven manual sequence in
`infra/run_test_case.sh` (single domain, GFS 3-hour boundaries, lowres geography). GFS is
auto-selected from the latest available NOMADS cycle (real data is retained ~10 days), and
the grid/duration are clamped to a runtime budget so a job finishes in minutes; the *actual*
config used is recorded in the result so it reports honestly rather than implying the full
requested resolution was run.

## Honest-failure contract

- A job reaches `completed` **only** when a real `wrfout` was produced and post-processed.
- If the WRF binaries are missing (`wrf_runner.assert_binaries` mirrors
  `infra/check_wrf_install.sh`), the job is set to `failed` with
  `error = "WRF engine unavailable: …"`.
- Any stage that runs but produces no output raises `WrfRunError`, which fails the job with
  an explanatory message — stages never silently no-op, so a cloud-modified run can never
  report a success that did nothing.

## Crash recovery

If the worker process dies mid-run (VM restart, OOM, kill), the job it held is left in a
running status with no worker. On every poll the daemon calls `db.reclaim_stale_jobs`, which
requeues any job whose `updated_at` is older than `WORKER_STALE_MINUTES` (default 20) back to
`created` so it is retried instead of spinning forever in the UI. Re-running is safe: the WPS
stage re-downloads GFS and re-creates its outputs idempotently in the existing run dir.

## Run

```bash
# 1. Install WRF/WPS on the VM and verify (see ../infra/README.md).
#    infra/ builds from source and works on both aarch64 and x86_64 Debian/Ubuntu.
bash ../infra/install_wrf.sh
bash ../infra/check_wrf_install.sh

# 2. Python deps (on a system-managed Python add --break-system-packages).
#    numpy is pinned <1.25 to keep ABI compatibility with the system netCDF4.
pip install -r requirements.txt

# 3. One-command launch: sources /opt/wrf/wrf_env.sh, pulls DATABASE_URL via the
#    openkbs CLI if unset, and execs the daemon. Repeatable on any worker VM.
./run_worker.sh
```

`run_worker.sh` is the portable entry point — any user on any machine hits the same AWS-hosted
site, and a single central worker VM (reproduced by `infra/`) processes the queue.

## Environment

| Var | Default | Purpose |
|-----|---------|---------|
| `DATABASE_URL` | — | Postgres job queue (required; `run_worker.sh` pulls it via the CLI) |
| `WRF_ROOT` | `/opt/wrf` | install root |
| `WRF_DIR` | `$WRF_ROOT/WRF` | WRF build |
| `WPS_DIR` | `$WRF_ROOT/WPS` | WPS build |
| `WRF_GEOG_PATH` | `$WRF_ROOT/geog` | static geography |
| `SIMULATION_STORAGE_PATH` | `$WRF_ROOT/runs` | per-job working dirs + outputs |
| `WORKER_POLL_INTERVAL` | `5` | queue poll seconds |
| `WORKER_MPI_RANKS` | min(CPU, 4) | MPI ranks for `real.exe` / `wrf.exe` |
| `WORKER_MAX_CELLS` | `60` | grid cells per side cap (runtime budget) |
| `WORKER_MAX_HOURS` | `3` | integration length cap (≥ one 3h GFS interval) |
| `WORKER_STALE_MINUTES` | `20` | requeue a running job idle longer than this |
