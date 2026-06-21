#!/usr/bin/env bash
# One-command launch for the ChillOut RL inference worker daemon.
#
# Runs the uploaded RL policy on a single Sentinel-2 mask (the Lambda can't run PyTorch).
# Uses the cloud_rl CPU venv (torch already installed there) and ensures psycopg2 is present
# for the job queue. Pulls DATABASE_URL from the OpenKBS project (same Postgres the Lambda
# writes to) and execs the polling daemon.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
VENV_PY="${VENV_PY:-$REPO_ROOT/cloud_rl/.venv/bin/python}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "error: python venv not found at $VENV_PY (expected the cloud_rl torch venv)" >&2
  exit 1
fi

# Job-queue driver — torch lives in the venv, but psycopg2 may not. Install once (idempotent).
if ! "$VENV_PY" -c "import psycopg2" >/dev/null 2>&1; then
  echo "[run_rl_worker] installing psycopg2-binary into the venv" >&2
  "$VENV_PY" -m pip install -q "psycopg2-binary>=2.9,<3"
fi

# Job queue connection. Prefer an already-exported DATABASE_URL; otherwise ask the CLI.
if [[ -z "${DATABASE_URL:-}" ]]; then
  if command -v openkbs >/dev/null 2>&1; then
    DATABASE_URL="$(cd "$REPO_ROOT" && openkbs postgres connection)"
    export DATABASE_URL
  else
    echo "error: DATABASE_URL not set and openkbs CLI not found" >&2
    exit 1
  fi
fi

echo "[run_rl_worker] starting daemon (cwd=$HERE)" >&2
cd "$HERE"
exec "$VENV_PY" rl_inference_worker.py
