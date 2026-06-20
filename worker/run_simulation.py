#!/usr/bin/env python3
"""ChillOut WRF worker daemon.

Polls the shared Postgres job queue for scenarios submitted by the Lambda API, then runs
the real WRF pipeline on this VM:

    claim job -> build namelists -> WPS (geogrid/ungrib/metgrid) -> real.exe
              -> baseline wrf.exe -> cloud_injector -> candidate wrf.exe
              -> postprocess -> store result

Hard rule enforced here: a job only reaches 'completed' when a real wrfout was produced
and post-processed. If the WRF binaries are missing, or any stage is not yet wired, the
job is marked 'failed' with an explanatory error — we never fabricate results.

Run:  DATABASE_URL=... python3 worker/run_simulation.py
"""
import os
import shutil
import time
import traceback

import config
import db
import wrf_case_builder
import wrf_runner
import cloud_injector
import postprocessor


def log(msg):
    print(f"[worker] {msg}", flush=True)


def process_job(job):
    job_id = job["id"]
    scenario = job["scenario"]
    log(f"claimed job {job_id}")

    # 1. Refuse to pretend: the engine must actually be installed.
    wrf_runner.assert_binaries()

    run_dir = os.path.join(config.SIMULATION_STORAGE_PATH, str(job_id))
    os.makedirs(run_dir, exist_ok=True)

    # 2. Backend resolves the scenario contract into a concrete, budget-clamped run config
    #    (frontend never touches namelists). The actual values used are recorded for honesty.
    cfg = wrf_case_builder.resolve_config(scenario)
    log(f"job {job_id} config: {cfg['e_we']}x{cfg['e_sn']} dx={cfg['dx']}m "
        f"run_hours={cfg['run_hours']} (requested {cfg['requested_hours']}h "
        f"@ {cfg['requested_res_m']}m)")

    # 3. WPS: fetch real GFS + geogrid/ungrib/metgrid. Returns the run window + cycle + nmet.
    db.set_status(job_id, "preprocessing")
    wps_info = wrf_runner.run_wps(run_dir, cfg)
    log(f"job {job_id} WPS done; GFS cycle {wps_info['cycle']:%Y-%m-%d %HZ}, "
        f"num_metgrid_levels={wps_info['nmet']}")

    # 4. real.exe builds wrfinput_d01 + wrfbdy_d01 from the met_em files.
    db.set_status(job_id, "initializing")
    wrf_runner.run_real(run_dir, cfg, wps_info)

    # 5. Baseline run. Clear any stale wrfout first so we can identify this run's output,
    #    then move it aside before the candidate run clobbers the name.
    db.set_status(job_id, "running_baseline")
    wrf_runner.clear_wrfout(run_dir)
    baseline_out = wrf_runner.run_wrf(run_dir, os.path.join(run_dir, "wrf_baseline.log"), cfg)
    baseline = os.path.join(run_dir, "wrf_baseline.nc")
    shutil.move(baseline_out, baseline)
    log(f"job {job_id} baseline wrfout -> {baseline}")

    # 6. Inject the cloud into wrfinput_d01, then run the candidate from the same boundaries.
    db.set_status(job_id, "running_candidate")
    inj = cloud_injector.inject_cloud(os.path.join(run_dir, "wrfinput_d01"), scenario)
    wrf_runner.clear_wrfout(run_dir)
    candidate_out = wrf_runner.run_wrf(run_dir, os.path.join(run_dir, "wrf_candidate.log"), cfg)
    candidate = os.path.join(run_dir, "wrf_candidate.nc")
    shutil.move(candidate_out, candidate)
    log(f"job {job_id} candidate wrfout -> {candidate}")

    # 7. Compare baseline vs candidate over the target region/window -> metrics + summary.
    db.set_status(job_id, "postprocessing")
    result = postprocessor.build_result(baseline, candidate, scenario, cfg, wps_info, inj)

    db.mark_completed(job_id, result)
    log(f"job {job_id} completed")


def main():
    log(f"starting; queue={'set' if config.DATABASE_URL else 'MISSING'} "
        f"WRF_DIR={config.WRF_DIR}")
    db.connect()  # fail fast if the queue is unreachable; db self-heals after this
    while True:
        try:
            requeued = db.reclaim_stale_jobs(config.WORKER_STALE_MINUTES)
            if requeued:
                log(f"requeued {requeued} stale job(s) whose worker had died")
            job = db.claim_next_job()
            if job is None:
                time.sleep(config.POLL_INTERVAL_SECONDS)
                continue
            try:
                process_job(job)
            except (wrf_runner.WrfUnavailable, wrf_runner.WrfRunError,
                    NotImplementedError) as e:
                log(f"job {job['id']} failed: {e}")
                db.mark_failed(job["id"], e)
            except Exception as e:  # unexpected — still fail honestly, keep the daemon alive
                log(f"job {job['id']} crashed: {e}\n{traceback.format_exc()}")
                db.mark_failed(job["id"], f"worker crash: {e}")
        except KeyboardInterrupt:
            log("shutting down")
            break
        except Exception as e:
            # Queue/connection error that survived the in-db retry — back off and loop.
            log(f"queue error: {e}; retrying in 10s")
            time.sleep(10)


if __name__ == "__main__":
    main()
