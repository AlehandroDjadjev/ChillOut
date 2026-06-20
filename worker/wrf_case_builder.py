"""Converts a validated scenario.json into WRF/WPS namelists. This is the *backend's*
job by design — the frontend never edits namelists, it only emits the scenario contract
(see infra/scenario.example.json). The validator (functions/api/lib/schema.mjs) has
already guaranteed the ranges relied on here.

The namelists mirror the proven manual chain in infra/run_test_case.sh (single domain,
GFS 3-hour boundaries, lowres geog, sfcp_to_sfcp). Multi-domain nesting is a later round;
for now we run one domain honouring the scenario's centre/resolution/size/duration, clamped
to a runtime budget so a job completes in minutes on the worker VM."""
import datetime as dt

import config

KM_PER_DEG_LAT = 111.0
GFS_INTERVAL_SECONDS = 10800  # GFS subset is fetched at f000 + f003 (3-hour spacing)


def _parse_utc(s):
    return dt.datetime.strptime(s.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")


def resolve_config(scenario):
    """Resolve the scenario into a concrete, budget-clamped single-domain run config.

    Returns a dict consumed by wrf_runner (geometry + GFS box) and the namelist builders.
    The clamps (WORKER_MAX_CELLS / WORKER_MAX_HOURS) keep the demo run small enough to
    finish quickly; the *actual* values used are recorded so the result can report them
    honestly rather than implying the full requested resolution was run."""
    dom = scenario["domain"]
    center = scenario["region"]["domain_center"]

    size_km = float(dom["inner_domain_size_km"])
    res_m = float(dom["inner_resolution_m"])

    # Cell count from size/resolution, clamped. If the requested grid is too large for the
    # budget, coarsen the resolution to keep the cell count (rather than shrink the area).
    n = int(round((size_km * 1000.0) / res_m))
    n = max(n, 10)
    if n > config.WORKER_MAX_CELLS:
        n = config.WORKER_MAX_CELLS
        res_m = (size_km * 1000.0) / n

    e_we = n + 1
    e_sn = n + 1
    dx = int(round(res_m))

    # Time step: WRF's stability rule is dt(s) <= 6 * dx(km) (the proven test case ran
    # 60 s at 10 km = exactly 6). We use a coefficient of 5 for margin against map-factor
    # scaling, and crucially DO NOT floor at a coarse value — a high floor (e.g. 10 s) makes
    # fine grids exceed the ratio and wrf.exe aborts with "Time step too large".
    time_step = max(2, min(int(5.0 * dx / 1000.0), 60))

    # Run length: honour duration but clamp, and never below one boundary interval (3 h) —
    # real.exe cannot build wrfbdy_d01 for a window shorter than one interval.
    requested_hours = int(scenario["time"]["duration_hours"])
    run_hours = max(3, min(requested_hours, config.WORKER_MAX_HOURS))

    out_int = int(scenario["time"]["output_interval_minutes"])

    # GFS grib-filter box: domain half-width in degrees + generous margin for the WRF halo.
    half_deg = (size_km / 2.0) / KM_PER_DEG_LAT + 3.0

    return {
        "lat": float(center["lat"]),
        "lon": float(center["lon"]),
        "e_we": e_we,
        "e_sn": e_sn,
        "e_vert": 35,
        "dx": dx,
        "time_step": time_step,
        "run_hours": run_hours,
        "requested_hours": requested_hours,
        "requested_res_m": int(dom["inner_resolution_m"]),
        "actual_res_m": dx,
        "out_int": out_int,
        "interval_seconds": GFS_INTERVAL_SECONDS,
        "box_half_deg": half_deg,
        "scenario_start_utc": scenario["time"]["start_utc"],
    }


def wps_namelist(cfg, start, wps_end, geog_path):
    """namelist.wps for a single domain. `start`/`wps_end` are datetimes; the WPS window
    spans one full GFS interval so >=2 met_em time levels are produced (needed by real)."""
    fmt = "%Y-%m-%d_%H:%M:%S"
    return f"""&share
 wrf_core = 'ARW', max_dom = 1,
 start_date = '{start.strftime(fmt)}', end_date = '{wps_end.strftime(fmt)}',
 interval_seconds = {cfg['interval_seconds']},
/
&geogrid
 parent_id = 1, parent_grid_ratio = 1, i_parent_start = 1, j_parent_start = 1,
 e_we = {cfg['e_we']}, e_sn = {cfg['e_sn']},
 dx = {cfg['dx']}, dy = {cfg['dx']},
 map_proj = 'lambert',
 ref_lat = {cfg['lat']}, ref_lon = {cfg['lon']},
 truelat1 = 30.0, truelat2 = 60.0, stand_lon = {cfg['lon']},
 geog_data_res = 'lowres',
 geog_data_path = '{geog_path}',
/
&ungrib
 out_format = 'WPS', prefix = 'FILE',
/
&metgrid
 fg_name = 'FILE',
/
"""


def input_namelist(cfg, start, end, nmet):
    """namelist.input for real.exe + wrf.exe. `nmet` = num_metgrid_levels detected from the
    met_em files after metgrid. Mirrors the proven settings from infra/run_test_case.sh."""
    return f"""&time_control
 run_days = 0, run_hours = {cfg['run_hours']}, run_minutes = 0, run_seconds = 0,
 start_year = {start.year}, start_month = {start.month:02d}, start_day = {start.day:02d}, start_hour = {start.hour:02d},
 end_year = {end.year}, end_month = {end.month:02d}, end_day = {end.day:02d}, end_hour = {end.hour:02d},
 interval_seconds = {cfg['interval_seconds']},
 input_from_file = .true., history_interval = {cfg['out_int']}, frames_per_outfile = 1000,
 restart = .false., io_form_history = 2, io_form_restart = 2,
 io_form_input = 2, io_form_boundary = 2,
/
&domains
 time_step = {cfg['time_step']}, max_dom = 1,
 e_we = {cfg['e_we']}, e_sn = {cfg['e_sn']}, e_vert = {cfg['e_vert']},
 num_metgrid_levels = {nmet}, num_metgrid_soil_levels = 4,
 sfcp_to_sfcp = .true.,
 dx = {cfg['dx']}, dy = {cfg['dx']},
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
"""
