#!/usr/bin/env bash
# One-command launch for the ChillOut WRF worker daemon on any prepared worker VM.
#
# Prereqs (reproducible via infra/): WRF/WPS built + verified
#   infra/install_wrf.sh   -> builds WRF 4.6.1 + WPS 4.6.0 under /opt/wrf
#   infra/check_wrf_install.sh -> asserts the five executables exist
#   pip3 install -r worker/requirements.txt  (numpy pinned for netCDF4 ABI)
#
# This script sources the WRF environment, pulls DATABASE_URL from the OpenKBS project
# (the same Postgres the Lambda API writes to), and execs the polling daemon.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. WRF runtime environment (paths + LD_LIBRARY_PATH for the NetCDF/HDF5/Jasper deps).
WRF_ENV="${WRF_ENV:-/opt/wrf/wrf_env.sh}"
if [[ -f "$WRF_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$WRF_ENV"
else
  echo "warning: $WRF_ENV not found; relying on existing environment" >&2
fi

# Map the proven install layout onto the env vars worker/config.py reads.
export WRF_ROOT="${WRF_ROOT:-/opt/wrf}"
export WRF_DIR="${WRF_DIR:-$WRF_ROOT/src/WRFV4.6.1}"
export WPS_DIR="${WPS_DIR:-$WRF_ROOT/src/WPS-4.6.0}"
export WRF_GEOG_PATH="${WRF_GEOG_PATH:-$WRF_ROOT/geog}"
export SIMULATION_STORAGE_PATH="${SIMULATION_STORAGE_PATH:-$WRF_ROOT/runs}"
mkdir -p "$SIMULATION_STORAGE_PATH"

# 2. Job queue connection. Prefer an already-exported DATABASE_URL; otherwise ask the CLI.
if [[ -z "${DATABASE_URL:-}" ]]; then
  if command -v openkbs >/dev/null 2>&1; then
    # openkbs reads openkbs.json from the repo root (parent of worker/).
    DATABASE_URL="$(cd "$HERE/.." && openkbs postgres connection)"
    export DATABASE_URL
  else
    echo "error: DATABASE_URL not set and openkbs CLI not found" >&2
    exit 1
  fi
fi

echo "[run_worker] WRF_DIR=$WRF_DIR runs=$SIMULATION_STORAGE_PATH" >&2
cd "$HERE"
exec python3 run_simulation.py
