#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PORT:=5173}"
: "${CLOUD_MODEL_PORT:=7860}"
: "${CLOUD_MODEL_HOST:=127.0.0.1}"
: "${CLOUD_MODEL_TESTAPP_URL:=http://${CLOUD_MODEL_HOST}:${CLOUD_MODEL_PORT}}"
export PORT CLOUD_MODEL_TESTAPP_URL

python NewModel/cloud_model_testapp.py \
  --host "$CLOUD_MODEL_HOST" \
  --port "$CLOUD_MODEL_PORT" \
  ${CLOUD_MODEL_FORCE_CPU:+--force-cpu} &

MODEL_PID="$!"
trap 'kill "$MODEL_PID" 2>/dev/null || true' EXIT

node server.mjs
