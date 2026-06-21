#!/usr/bin/env python3
"""ChillOut cloud model inference worker daemon.

Polls the shared Postgres `inference_jobs` queue for jobs enqueued by the site, then runs the
local NewModel cloud template inference stack on this machine. The browser no longer uploads a
checkpoint and the API no longer calls any AI image fallback; the worker loads the model-test-app
weights from fixed paths on disk.

Run:  bash worker/run_rl_worker.sh   (or DATABASE_URL=... python worker/rl_inference_worker.py)
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import config
import rl_db

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_MODEL_DIR = os.path.join(REPO_ROOT, "NewModel")
if NEW_MODEL_DIR not in sys.path:
    sys.path.insert(0, NEW_MODEL_DIR)

from cloud_model_testapp_inference import CloudTemplateInference, InferenceConfig  # noqa: E402

_ENGINE = None


def log(msg: str) -> None:
    print(f"[cloud-model-worker] {msg}", flush=True)


def _path_from_env(name: str, default_relative: str) -> str:
    raw = os.environ.get(name)
    if raw:
        return raw
    return os.path.join(NEW_MODEL_DIR, default_relative)


def get_engine() -> CloudTemplateInference:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    data_root = _path_from_env("CLOUD_MODEL_DATA_ROOT", "dataset_cloudforce_radiation_v6_big_clean")
    selector_checkpoint = _path_from_env("CLOUD_SELECTOR_CHECKPOINT", "cloud_template_selector_v13/best_selector.pt")
    reward_checkpoint = _path_from_env("CLOUD_REWARD_CHECKPOINT", "runs/cloud_radiation_v8_clean_direct1/best.pt")
    missing = [p for p in (selector_checkpoint, reward_checkpoint) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Cloud model checkpoint(s) missing: "
            + ", ".join(missing)
            + ". Add the weights or set CLOUD_SELECTOR_CHECKPOINT/CLOUD_REWARD_CHECKPOINT."
        )

    log(
        "loading cloud model test app inference "
        f"selector={selector_checkpoint} reward={reward_checkpoint} data_root={data_root}"
    )
    _ENGINE = CloudTemplateInference(
        InferenceConfig(
            data_root=data_root,
            selector_checkpoint=selector_checkpoint,
            reward_checkpoint=reward_checkpoint,
            split=os.environ.get("CLOUD_MODEL_SPLIT", "val"),
            force_cpu=os.environ.get("CLOUD_MODEL_FORCE_CPU", "").lower() in {"1", "true", "yes"},
            load_dataset=False,
        )
    )
    return _ENGINE


def process_job(job):
    job_id = job["id"]
    sample = job.get("sample")
    if not isinstance(sample, dict):
        raise ValueError("inference job is missing the same-date sample payload")
    if not job.get("mask_data_url"):
        raise ValueError("inference job is missing mask_data_url")

    target_c = float(job["target_temp"]) if job["target_temp"] is not None else float(
        sample.get("target_temperature_c") or sample.get("observed_temperature_c") or 20.0
    )
    sample["target_temperature_c"] = target_c
    log(f"claimed job {job_id} sample={sample.get('sample_id', job.get('place'))} target={target_c:.2f}C")

    engine = get_engine()
    result = engine.generate_live(
        sample=sample,
        mask_data_url=job["mask_data_url"],
        target_temperature_c=target_c,
        wm2_per_c=float(os.environ.get("CLOUD_MODEL_WM2_PER_C", "80")),
        input_mode=os.environ.get("CLOUD_MODEL_INPUT_MODE", "full"),
        run_oracle=os.environ.get("CLOUD_MODEL_RUN_ORACLE", "1").lower() not in {"0", "false", "no"},
        oracle_batch_size=int(os.environ.get("CLOUD_MODEL_ORACLE_BATCH_SIZE", "32")),
        penalty_l1=float(os.environ.get("CLOUD_MODEL_PENALTY_L1", "25")),
        penalty_coverage=float(os.environ.get("CLOUD_MODEL_PENALTY_COVERAGE", "80")),
        seed=int(os.environ.get("CLOUD_MODEL_SEED", "123")),
    )

    images = result.get("images") or {}
    verification = result.get("verification") or {}
    payload = {
        "original_mask_data_url": images.get("original_mask") or images.get("original"),
        "generated_mask_data_url": images.get("generated_mask") or images.get("generated"),
        "property_maps": {
            "generated_cloud_scene": images.get("generated"),
            "generated_channels": images.get("channels"),
            "original_sequence": images.get("original_strip"),
            "generated_sequence": images.get("generated_strip"),
        },
        "actions": [
            {
                "source": (result.get("chosen") or {}).get("source"),
                "mode": (result.get("chosen") or {}).get("mode"),
                "template_index": (result.get("chosen") or {}).get("template_index"),
                "template_sample_id": (result.get("chosen") or {}).get("template_sample_id"),
            }
        ],
        "target_temperature_c": target_c,
        "normalization": "cloud_model_testapp_live",
        "model": "CloudTemplateSelectorV13 + CloudRadiationV8CleanDirect",
        "sample": result.get("sample"),
        "target": result.get("target"),
        "verification": verification,
        "selector": result.get("selector"),
        "chosen": result.get("chosen"),
        "oracle": result.get("oracle"),
        "input": result.get("input"),
        "metrics": {
            "target_abs_error_wm2": verification.get("target_abs_error_wm2"),
            "temperature_abs_error_c": verification.get("temperature_abs_error_c"),
            "verified_temperature_change_c": verification.get("verified_temperature_change_c"),
            "coverage": verification.get("coverage"),
            "clear_sky_wm2": verification.get("clear_sky_wm2"),
        },
    }
    rl_db.mark_completed(job_id, payload)
    log(f"job {job_id} completed mode={(result.get('chosen') or {}).get('mode')}")


def main():
    log(f"starting; queue={'set' if config.DATABASE_URL else 'MISSING'} new_model={NEW_MODEL_DIR}")
    rl_db.connect()
    while True:
        try:
            requeued = rl_db.reclaim_stale_jobs(config.WORKER_STALE_MINUTES)
            if requeued:
                log(f"requeued {requeued} stale job(s) whose worker had died")
            job = rl_db.claim_next_job()
            if job is None:
                time.sleep(config.POLL_INTERVAL_SECONDS)
                continue
            try:
                process_job(job)
            except Exception as e:
                log(f"job {job['id']} crashed: {e}\n{traceback.format_exc()}")
                rl_db.mark_failed(job["id"], f"inference error: {e}")
        except KeyboardInterrupt:
            log("shutting down")
            break
        except Exception as e:
            log(f"queue error: {e}; retrying in 10s")
            time.sleep(10)


if __name__ == "__main__":
    main()
