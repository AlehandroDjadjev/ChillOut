#!/usr/bin/env python3
"""ChillOut RL inference worker daemon.

Polls the shared Postgres `inference_jobs` queue for jobs enqueued by the Lambda API, then
runs the uploaded RL policy on a single Sentinel-2 cloud mask on this machine (the Lambda
cannot run PyTorch):

    claim job -> download checkpoint (+ stats.json) from S3 -> build one-sample batch
              -> CloudActorCritic.deterministic -> rasterize_actions
              -> encode masks to base64 PNG -> store result

Mirrors worker/run_simulation.py in shape (claim/process/mark loop, fail honestly, stay alive).

Run:  bash worker/run_rl_worker.sh   (or DATABASE_URL=... python3 worker/rl_inference_worker.py)
"""
import base64
import io
import os
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image

import config
import rl_db

# Make the cloud_rl package importable (repo_root/cloud_rl contains the `cloud_rl` package).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUD_RL_DIR = os.path.join(REPO_ROOT, "cloud_rl")
if CLOUD_RL_DIR not in sys.path:
    sys.path.insert(0, CLOUD_RL_DIR)

import json  # noqa: E402  (stdlib, kept near the cloud_rl imports for clarity)

from cloud_rl.actions import actions_to_jsonable, rasterize_actions  # noqa: E402
from cloud_rl.dataset import (  # noqa: E402
    FEATURE_KEYS,
    assemble_obs_map,
    build_single_policy_features,
    default_stats_for_keys,
)
from cloud_rl.models import CloudActorCritic  # noqa: E402

DEVICE = torch.device("cpu")
IMAGE_SIZE = (256, 256)


def log(msg):
    print(f"[rl-worker] {msg}", flush=True)


def storage_download(key, dest):
    """Pull an uploaded object over HTTPS from the public CDN. Objects live under the `media/`
    prefix and are served by CloudFront, so a plain GET works (the random job UUID in the key
    keeps them unguessable). Uploaded by the Lambda's presigned PUT, fetched here."""
    url = f"{config.STORAGE_CDN_BASE}/{key.lstrip('/')}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        if resp.status != 200:
            raise RuntimeError(f"download {key} failed: HTTP {resp.status}")
        data = resp.read()
    with open(dest, "wb") as fh:
        fh.write(data)


def decode_mask_data_url(data_url):
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw = base64.b64decode(b64)
    h, w = IMAGE_SIZE
    img = Image.open(io.BytesIO(raw)).convert("L").resize((w, h), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr >= 0.5).astype(np.float32)
    return torch.from_numpy(arr)[None, :, :]


def tensor_to_data_url(t):
    arr = t.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_stats(stats_path):
    """Return (feature_mean, feature_std, target_mean, target_std). Falls back to the
    dataset defaults if no stats.json was uploaded."""
    if stats_path and os.path.exists(stats_path):
        stats = json.loads(open(stats_path, "r", encoding="utf-8").read())
        feature_mean = np.asarray(stats["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(stats["feature_std"], dtype=np.float32)
        target_mean = float(stats["target_temp_mean"][0])
        target_std = float(stats["target_temp_std"][0])
    else:
        feature_mean, feature_std = default_stats_for_keys(FEATURE_KEYS)
        target_mean, target_std = 20.0, 10.0
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    target_std = max(target_std, 1e-6)
    return feature_mean, feature_std, target_mean, target_std


def process_job(job):
    job_id = job["id"]
    log(f"claimed job {job_id} (checkpoint={job['checkpoint_key']})")

    raw_features = np.asarray(job["raw_features"], dtype=np.float32)
    if raw_features.shape[0] != len(FEATURE_KEYS):
        raise ValueError(f"raw_features must have {len(FEATURE_KEYS)} values, got {raw_features.shape[0]}")
    target_c = float(job["target_temp"]) if job["target_temp"] is not None else 20.0

    with tempfile.TemporaryDirectory(prefix=f"rlinfer_{job_id}_") as tmp:
        # 1. Pull the weights (and stats if provided).
        ckpt_path = os.path.join(tmp, "checkpoint.pt")
        storage_download(job["checkpoint_key"], ckpt_path)
        stats_path = None
        if job.get("stats_key"):
            stats_path = os.path.join(tmp, "stats.json")
            storage_download(job["stats_key"], stats_path)

        # 2. Load checkpoint + its training config.
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("cfg") or {}
        lookback = int(cfg.get("lookback", 4))
        policy_cfg = cfg.get("policy", {})
        max_actions = int(policy_cfg.get("max_actions", 6))
        hidden_dim = int(policy_cfg.get("hidden_dim", 128))

        # 3. Build the single-sample inputs exactly as the dataset would (shared helpers).
        feature_mean, feature_std, target_mean, target_std = load_stats(stats_path)
        policy_features, target_norm = build_single_policy_features(
            raw_features, feature_mean, feature_std, target_mean, target_std,
            target_c=target_c, lookback=lookback,
        )
        mask = decode_mask_data_url(job["mask_data_url"])  # [1,H,W]
        obs_map = assemble_obs_map(mask, policy_features, target_norm, IMAGE_SIZE)
        feature_dim = int(policy_features.shape[0])

        obs_map_b = obs_map.unsqueeze(0).to(DEVICE)
        features_b = torch.from_numpy(policy_features).float().unsqueeze(0).to(DEVICE)
        target_norm_b = torch.tensor([[target_norm]], dtype=torch.float32, device=DEVICE)
        original_mask_b = mask.unsqueeze(0).to(DEVICE)  # [1,1,H,W]

        # 4. Build the model exactly as cloud_rl/evaluate.py and load the uploaded weights.
        model = CloudActorCritic(
            obs_channels=1 + feature_dim + 1,
            feature_dim=feature_dim,
            max_actions=max_actions,
            hidden_dim=hidden_dim,
        ).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()

        # 5. Deterministic policy -> action tokens -> rasterized masks.
        with torch.no_grad():
            sampled = model.deterministic(obs_map_b, features_b, target_norm_b)
            rast = rasterize_actions(original_mask_b, sampled["op"], sampled["params"])

        property_maps = rast["property_maps"][0]  # [5,H,W]
        result = {
            "original_mask_data_url": tensor_to_data_url(original_mask_b[0, 0]),
            "generated_mask_data_url": tensor_to_data_url(rast["generated_mask"][0, 0]),
            "property_maps": {
                "cloud_probability": tensor_to_data_url(property_maps[0]),
                "aot_norm": tensor_to_data_url(property_maps[1]),
                "cloud_layer": tensor_to_data_url(property_maps[2]),
                "texture_norm": tensor_to_data_url(property_maps[3]),
                "cirrus_proxy": tensor_to_data_url(property_maps[4]),
            },
            "actions": actions_to_jsonable(sampled["op"][0].cpu(), sampled["params"][0].cpu()),
            "target_temperature_c": target_c,
            "feature_dim": feature_dim,
            "lookback": lookback,
            "normalization": "uploaded_stats" if stats_path else "default_stats",
        }

    rl_db.mark_completed(job_id, result)
    log(f"job {job_id} completed ({len(result['actions'])} actions)")


def main():
    log(f"starting; queue={'set' if config.DATABASE_URL else 'MISSING'} cloud_rl={CLOUD_RL_DIR}")
    rl_db.connect()  # fail fast if the queue is unreachable; db self-heals after this
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
            except urllib.error.URLError as e:
                log(f"job {job['id']} failed (download): {e}")
                rl_db.mark_failed(job["id"], f"checkpoint download failed: {str(e)[:500]}")
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
