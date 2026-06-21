#!/usr/bin/env bash
# ============================================================================
# run_test_case.sh — Minimal REAL-DATA WRF run over Stara Zagora, Bulgaria.
#
# Exercises the real product pipeline:
#   geogrid.exe -> link_grib + ungrib.exe -> metgrid.exe -> real.exe -> wrf.exe
#
# Tiny single domain (~40x40 @ 10 km), short run. Scientific accuracy is NOT
# the point — proving the toolchain end-to-end is. Fails loudly on any stage.
#
# Inputs are fetched live:
#   * WPS low-res mandatory geog  -> $WRF_GEOG_PATH
#   * small GFS 0.25 subset (region-clipped) from NOMADS grib-filter
# GFS on NOMADS is retained ~10 days; the script auto-selects a recent cycle.
#
# Env:
#   WRF_ROOT (default /opt/wrf), WRF_GEOG_PATH (default $WRF_ROOT/geog),
#   SIMULATION_STORAGE_PATH (default $WRF_ROOT/cases), CASE (default test_stz),
#   RUN_HOURS (default 1)
# ============================================================================
set -Eeuo pipefail

WRF_ROOT="${WRF_ROOT:-/opt/wrf}"
[ -f "$WRF_ROOT/wrf_env.sh" ] && source "$WRF_ROOT/wrf_env.sh"
WRF_DIR="${WRF_DIR:-$WRF_ROOT/WRF}"
WPS_DIR="${WPS_DIR:-$WRF_ROOT/WPS}"
WRF_GEOG_PATH="${WRF_GEOG_PATH:-$WRF_ROOT/geog}"
SIMULATION_STORAGE_PATH="${SIMULATION_STORAGE_PATH:-$WRF_ROOT/cases}"
CASE="${CASE:-test_stz}"
# real.exe needs the model window to span >= one boundary interval (GFS = 3h here),
# otherwise it cannot build wrfbdy_d01. Default to one full interval.
RUN_HOURS="${RUN_HOURS:-3}"
RUNDIR="$SIMULATION_STORAGE_PATH/$CASE"
LOG="$RUNDIR/run.log"

log() { echo -e "\033[36m[run]\033[0m $*" | tee -a "$LOG"; }
die() { echo -e "\033[31m[run:ERROR]\033[0m $*" | tee -a "$LOG" >&2; exit 1; }
trap 'echo -e "\033[31m[run:ERROR]\033[0m failed at line $LINENO (see $LOG)" >&2' ERR

[ -x "$WRF_DIR/main/real.exe" ] || die "real.exe missing — run infra/install_wrf.sh first"
[ -x "$WPS_DIR/geogrid.exe" ]   || die "geogrid.exe missing — run infra/install_wrf.sh first"

mkdir -p "$RUNDIR" "$WRF_GEOG_PATH"; : > "$LOG"
# Drop stale intermediates from a prior run so leftovers can never mask a failure
# (grib/ and geo_em are kept to avoid re-downloading / re-running geogrid).
rm -f "$RUNDIR"/FILE:* "$RUNDIR"/GRIBFILE.* "$RUNDIR"/met_em.d01.*.nc \
      "$RUNDIR"/wrfinput_d01 "$RUNDIR"/wrfbdy_d01 "$RUNDIR"/wrfout_d01_* 2>/dev/null || true
log "RUNDIR=$RUNDIR  RUN_HOURS=$RUN_HOURS"

# Domain centred on Stara Zagora (42.42N, 25.62E)
REF_LAT=42.42; REF_LON=25.62
DX=10000; EWE=40; ESN=40

# ---------------------------------------------------------------------------
# 0. Geography data (low-res mandatory)
# ---------------------------------------------------------------------------
if [ ! -d "$WRF_GEOG_PATH/topo_gmted2010_30s" ] && [ ! -d "$WRF_GEOG_PATH/topo_30s" ] \
   && [ -z "$(ls -A "$WRF_GEOG_PATH" 2>/dev/null)" ]; then
  log "0/6 downloading WPS low-res mandatory geog (one-time, large)…"
  TGZ="$WRF_ROOT/src/geog_low_res_mandatory.tar.gz"
  [ -f "$TGZ" ] || wget -q -O "$TGZ" \
    "https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_low_res_mandatory.tar.gz" \
    || die "geog download failed"
  tar -xzf "$TGZ" -C "$WRF_GEOG_PATH" --strip-components=1 \
    || tar -xzf "$TGZ" -C "$WRF_GEOG_PATH"
fi
log "0/6 geog at $WRF_GEOG_PATH"

# ---------------------------------------------------------------------------
# 1. Pick a recent available GFS cycle and fetch a region-clipped subset
# ---------------------------------------------------------------------------
log "1/6 locating a recent GFS cycle on NOMADS"
FILTER="https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
# Region box around Bulgaria (generous margin so the WRF domain is covered)
BOX="leftlon=20&rightlon=31&toplat=46&bottomlat=38"
# Variables WRF needs from GFS
VARS="var_HGT=on&var_TMP=on&var_RH=on&var_UGRD=on&var_VGRD=on&var_SPFH=on&var_PRES=on&var_PRMSL=on&var_LAND=on&var_ICEC=on&var_WEASD=on&var_SNOD=on&var_SOILW=on&var_TSOIL=on"
# Levels (isobaric + surface/special + soil)
LEVS="lev_surface=on&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_mean_sea_level=on"
LEVS="$LEVS&lev_0-0.1_m_below_ground=on&lev_0.1-0.4_m_below_ground=on&lev_0.4-1_m_below_ground=on&lev_1-2_m_below_ground=on"
for mb in 1000 975 950 925 900 850 800 750 700 650 600 550 500 450 400 350 300 250 200 150 100 70 50 30 20 10; do
  LEVS="$LEVS&lev_${mb}_mb=on"
done

GRIBDIR="$RUNDIR/grib"; mkdir -p "$GRIBDIR"
FOUND_CYCLE=""
# try last 3 days, cycles 18,12,06,00, hours f000 + f003
for d in 0 1 2; do
  DAY=$(date -u -d "-$d day" +%Y%m%d)
  for cyc in 18 12 06 00; do
    base="${FILTER}?dir=%2Fgfs.${DAY}%2F${cyc}%2Fatmos&${BOX}&${VARS}&${LEVS}"
    ok=1
    for fh in 000 003; do
      f="gfs.t${cyc}z.pgrb2.0p25.f${fh}"
      url="${base}&file=${f}"
      if ! curl -fsS -m 180 -o "$GRIBDIR/$f" "$url" 2>>"$LOG"; then ok=0; break; fi
      # a valid GRIB starts with 'GRIB'; HTML error pages do not
      if ! head -c4 "$GRIBDIR/$f" | grep -q GRIB; then ok=0; rm -f "$GRIBDIR/$f"; break; fi
    done
    if [ "$ok" = 1 ]; then FOUND_CYCLE="${DAY}${cyc}"; break; fi
  done
  [ -n "$FOUND_CYCLE" ] && break
done
[ -n "$FOUND_CYCLE" ] || die "no usable GFS cycle found on NOMADS (retention/outage). See README fallback (idealized em_les)."
DAY=${FOUND_CYCLE:0:8}; CYC=${FOUND_CYCLE:8:2}
log "1/6 using GFS cycle ${DAY} ${CYC}Z  ($(du -sh "$GRIBDIR" | cut -f1) subset)"

# Derived start/end (start = cycle time). Use an explicit ISO date + "UTC" so GNU
# date does relative-hour math instead of misreading "+N" as a timezone offset.
BASE_ISO="${DAY:0:4}-${DAY:4:2}-${DAY:6:2} ${CYC}:00:00 UTC"
START_UTC=$(date -u -d "$BASE_ISO" +%Y-%m-%d_%H:%M:%S)
END_UTC=$(date -u -d "$BASE_ISO +${RUN_HOURS} hours" +%Y-%m-%d_%H:%M:%S)
SY=${START_UTC:0:4} SM=${START_UTC:5:2} SD=${START_UTC:8:2} SH=${START_UTC:11:2}
EY=${END_UTC:0:4} EM=${END_UTC:5:2} ED=${END_UTC:8:2} EH=${END_UTC:11:2}
# WPS must emit >=2 met_em time levels so real.exe can build wrfbdy_d01. The GFS
# subset above provides f000 and f003 (one 3h interval), so the WPS window spans
# a full interval regardless of the (shorter) model run length.
WPS_END_UTC=$(date -u -d "$BASE_ISO +3 hours" +%Y-%m-%d_%H:%M:%S)

# ---------------------------------------------------------------------------
# 2. namelist.wps + geogrid
# ---------------------------------------------------------------------------
log "2/6 geogrid"
cat > "$RUNDIR/namelist.wps" <<EOF
&share
 wrf_core = 'ARW', max_dom = 1,
 start_date = '${START_UTC}', end_date = '${WPS_END_UTC}',
 interval_seconds = 10800,
/
&geogrid
 parent_id = 1, parent_grid_ratio = 1, i_parent_start = 1, j_parent_start = 1,
 e_we = ${EWE}, e_sn = ${ESN},
 dx = ${DX}, dy = ${DX},
 map_proj = 'lambert',
 ref_lat = ${REF_LAT}, ref_lon = ${REF_LON},
 truelat1 = 30.0, truelat2 = 60.0, stand_lon = ${REF_LON},
 geog_data_res = 'lowres',
 geog_data_path = '${WRF_GEOG_PATH}',
/
&ungrib
 out_format = 'WPS', prefix = 'FILE',
/
&metgrid
 fg_name = 'FILE',
/
EOF

cd "$RUNDIR"
ln -sf "$WPS_DIR/geogrid.exe" .; ln -sf "$WPS_DIR/metgrid.exe" .; ln -sf "$WPS_DIR/ungrib.exe" .
ln -sf "$WPS_DIR/geogrid" .; ln -sf "$WPS_DIR/metgrid" .
ln -sf "$WPS_DIR/ungrib/Variable_Tables/Vtable.GFS" Vtable
"$WPS_DIR/geogrid.exe" >>"$LOG" 2>&1 || die "geogrid failed"
ls geo_em.d01.nc >/dev/null 2>&1 || die "geogrid produced no geo_em.d01.nc"
log "2/6 geo_em.d01.nc OK"

# ---------------------------------------------------------------------------
# 3. link_grib + ungrib
# ---------------------------------------------------------------------------
log "3/6 ungrib"
"$WPS_DIR/link_grib.csh" "$GRIBDIR"/gfs.* >>"$LOG" 2>&1 || ./link_grib.csh "$GRIBDIR"/gfs.* >>"$LOG" 2>&1 || die "link_grib failed"
"$WPS_DIR/ungrib.exe" >>"$LOG" 2>&1 || die "ungrib failed"
ls FILE:* >/dev/null 2>&1 || die "ungrib produced no FILE: intermediates"
log "3/6 ungrib intermediates: $(ls FILE:* | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# 4. metgrid
# ---------------------------------------------------------------------------
log "4/6 metgrid"
"$WPS_DIR/metgrid.exe" >>"$LOG" 2>&1 || die "metgrid failed"
ls met_em.d01.*.nc >/dev/null 2>&1 || die "metgrid produced no met_em files"
MET1=$(ls met_em.d01.*.nc | head -1)
NMET=$(ncdump -h "$MET1" | sed -n 's/.*num_metgrid_levels = \([0-9]*\).*/\1/p' | head -1)
[ -n "$NMET" ] || die "could not read num_metgrid_levels from $MET1"
log "4/6 met_em OK; num_metgrid_levels=$NMET"

# ---------------------------------------------------------------------------
# 5. real.exe
# ---------------------------------------------------------------------------
log "5/6 real.exe"
cat > "$RUNDIR/namelist.input" <<EOF
&time_control
 run_days = 0, run_hours = ${RUN_HOURS}, run_minutes = 0, run_seconds = 0,
 start_year = ${SY}, start_month = ${SM}, start_day = ${SD}, start_hour = ${SH},
 end_year = ${EY}, end_month = ${EM}, end_day = ${ED}, end_hour = ${EH},
 interval_seconds = 10800,
 input_from_file = .true., history_interval = 30, frames_per_outfile = 1000,
 restart = .false., io_form_history = 2, io_form_restart = 2,
 io_form_input = 2, io_form_boundary = 2,
/
&domains
 time_step = 60, max_dom = 1,
 e_we = ${EWE}, e_sn = ${ESN}, e_vert = 35,
 num_metgrid_levels = ${NMET}, num_metgrid_soil_levels = 4,
 sfcp_to_sfcp = .true.,
 dx = ${DX}, dy = ${DX},
 grid_id = 1, parent_id = 1, i_parent_start = 1, j_parent_start = 1,
 parent_grid_ratio = 1, parent_time_step_ratio = 1, feedback = 1, smooth_option = 0,
/
&physics
 mp_physics = 6, ra_lw_physics = 4, ra_sw_physics = 4, radt = 10,
 sf_sfclay_physics = 1, sf_surface_physics = 2, bl_pbl_physics = 1, bldt = 0,
 cu_physics = 1, cudt = 5, num_soil_layers = 4, sf_urban_physics = 0,
/
&dynamics
 w_damping = 1, diff_opt = 1, km_opt = 4, diff_6th_opt = 0, damp_opt = 3,
 zdamp = 5000., dampcoef = 0.2, khdif = 0, kvdif = 0, non_hydrostatic = .true.,
/
&bdy_control
 spec_bdy_width = 5, spec_zone = 1, relax_zone = 4,
 specified = .true., nested = .false.,
/
&namelist_quilt
 nio_tasks_per_group = 0, nio_groups = 1,
/
EOF

ln -sf "$WRF_DIR/main/real.exe" .; ln -sf "$WRF_DIR/main/wrf.exe" .
for t in "$WRF_DIR/run/"*.TBL "$WRF_DIR/run/"*.formatted "$WRF_DIR/run/"RRTMG_* \
         "$WRF_DIR/run/ozone"* "$WRF_DIR/run/CAM"* "$WRF_DIR/run/"*_DATA; do
  [ -e "$t" ] && ln -sf "$t" . || true
done
# met_em.d01.*.nc are already in this dir (metgrid wrote them here) — real.exe reads them directly.
NP=$(( $(nproc) > 4 ? 4 : $(nproc) ))
mpirun -np "$NP" ./real.exe >>"$LOG" 2>&1 || ./real.exe >>"$LOG" 2>&1 || die "real.exe failed (see rsl.* / $LOG)"
ls wrfinput_d01 wrfbdy_d01 >/dev/null 2>&1 || die "real.exe produced no wrfinput_d01/wrfbdy_d01"
log "5/6 real.exe OK: wrfinput_d01 wrfbdy_d01"

# ---------------------------------------------------------------------------
# 6. wrf.exe
# ---------------------------------------------------------------------------
log "6/6 wrf.exe (this is the actual integration)"
mpirun -np "$NP" ./wrf.exe >>"$LOG" 2>&1 || ./wrf.exe >>"$LOG" 2>&1 || die "wrf.exe failed (see rsl.error.* / $LOG)"
WRFOUT=$(ls wrfout_d01_* 2>/dev/null | head -1)
[ -n "$WRFOUT" ] || die "wrf.exe produced no wrfout_d01_* file"
NT=$(ncdump -h "$WRFOUT" | sed -n 's/.*Time = UNLIMITED.*(\([0-9]*\) currently).*/\1/p' | head -1)
log "6/6 SUCCESS: $WRFOUT  (Time levels: ${NT:-?})"
[ "${NT:-0}" -ge 2 ] 2>/dev/null || die "wrfout has <2 time levels (NT=${NT:-0}) — run did not integrate"

echo
echo -e "\033[32m=== WRF MINIMAL TEST PASSED ===\033[0m"
echo "wrfout: $RUNDIR/$WRFOUT  with $NT time levels"
echo "variables: $(ncdump -h "$WRFOUT" | grep -cE '^\s+float|^\s+double') fields (T2, U10, V10, RAINNC, …)"
