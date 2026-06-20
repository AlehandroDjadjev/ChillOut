#!/usr/bin/env bash
# ============================================================================
# check_wrf_install.sh — Verify the WRF/WPS toolchain is fully installed.
# Prints a PASS/FAIL table and exits non-zero if anything is missing.
# ============================================================================
set -uo pipefail

WRF_ROOT="${WRF_ROOT:-/opt/wrf}"
[ -f "$WRF_ROOT/wrf_env.sh" ] && source "$WRF_ROOT/wrf_env.sh"
WRF_DIR="${WRF_DIR:-$WRF_ROOT/WRF}"
WPS_DIR="${WPS_DIR:-$WRF_ROOT/WPS}"

fail=0
ok()   { printf "  \033[32mPASS\033[0m  %-28s %s\n" "$1" "$2"; }
no()   { printf "  \033[31mFAIL\033[0m  %-28s %s\n" "$1" "$2"; fail=1; }

check_cmd() { if command -v "$1" >/dev/null 2>&1; then ok "$1" "$(command -v "$1")"; else no "$1" "not found"; fi; }
check_bin() { if [ -x "$1" ]; then ok "$(basename "$1")" "$1"; else no "$(basename "$1")" "missing/not executable: $1"; fi; }

echo "== Toolchain =="
check_cmd gcc
check_cmd gfortran
check_cmd make
check_cmd mpirun
check_cmd mpif90

echo "== Libraries =="
if command -v nc-config >/dev/null 2>&1; then ok "netcdf-c" "$(nc-config --version 2>/dev/null)"; else no "netcdf-c" "nc-config missing"; fi
if command -v nf-config >/dev/null 2>&1; then ok "netcdf-fortran" "$(nf-config --version 2>/dev/null)"; else no "netcdf-fortran" "nf-config missing"; fi
if [ -e "${NETCDF:-/opt/wrf/deps/netcdf}/include/netcdf.mod" ]; then ok "netcdf.mod" "${NETCDF}/include/netcdf.mod"; else no "netcdf.mod" "Fortran module missing"; fi
if ls "${JASPERLIB:-/usr/lib}"/libjasper.* >/dev/null 2>&1; then ok "jasper" "${JASPERLIB}"; else no "jasper" "libjasper missing"; fi

echo "== WRF binaries =="
check_bin "$WRF_DIR/main/wrf.exe"
check_bin "$WRF_DIR/main/real.exe"

echo "== WPS binaries =="
check_bin "$WPS_DIR/geogrid.exe"
check_bin "$WPS_DIR/ungrib.exe"
check_bin "$WPS_DIR/metgrid.exe"

echo
if [ "$fail" -eq 0 ]; then
  echo -e "\033[32mALL CHECKS PASSED\033[0m — ready to run: bash infra/run_test_case.sh"
else
  echo -e "\033[31mONE OR MORE CHECKS FAILED\033[0m — run infra/install_wrf.sh (see \$WRF_ROOT/install.log)"
fi
exit "$fail"
