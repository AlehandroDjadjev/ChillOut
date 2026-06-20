"""Drives the real WRF/WPS executables, mirroring the proven manual chain in
infra/run_test_case.sh. The hard rule: if the binaries are missing, raise WrfUnavailable;
if a stage runs but produces no output, raise WrfRunError. We never fake a completed run."""
import datetime as dt
import glob
import os
import shutil
import subprocess

import config
import wrf_case_builder as nb

# GFS grib-filter (NOMADS). Same variables/levels the proven test case uses.
NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
GFS_VARS = ("var_HGT=on&var_TMP=on&var_RH=on&var_UGRD=on&var_VGRD=on&var_SPFH=on&"
            "var_PRES=on&var_PRMSL=on&var_LAND=on&var_ICEC=on&var_WEASD=on&var_SNOD=on&"
            "var_SOILW=on&var_TSOIL=on")
_PRESSURE_MB = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450,
                400, 350, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10)


class WrfUnavailable(RuntimeError):
    """Raised when the WRF/WPS toolchain is not installed or not executable."""


class WrfRunError(RuntimeError):
    """Raised when an executable runs but does not produce its expected output."""


def assert_binaries():
    """Mirror infra/check_wrf_install.sh: every one of the five executables must exist
    and be executable. Returns nothing; raises WrfUnavailable listing what is missing."""
    missing = []
    for name, path in config.WRF_BINARIES.items():
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            missing.append(f"{name} ({path})")
    if missing:
        raise WrfUnavailable(
            "WRF engine unavailable: missing/non-executable binaries: "
            + ", ".join(missing)
            + ". Run infra/install_wrf.sh then infra/check_wrf_install.sh on the worker VM."
        )


def _run(cmd, cwd, log_path):
    """Run a step, streaming combined stdout/stderr to a per-step log. Raises on
    non-zero exit so the caller can fail the job honestly."""
    with open(log_path, "ab") as log:
        log.write(f"\n$ {' '.join(cmd)}\n".encode())
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise WrfRunError(f"{' '.join(cmd)} exited {proc.returncode} (see {log_path})")


def _ranks_for(cfg):
    """MPI ranks to use for this grid: the configured ceiling, capped so each WRF tile keeps
    a healthy patch (~>=10 cells/side per rank). WRF aborts decomposition if a small/coarsened
    domain is split across too many tiles, so the grid sets the real upper bound here.
    Examples: 60 cells -> 36 (then ceil'd by MPI_RANKS=12); 30 -> 9; 20 -> 4."""
    side = min(int(cfg["e_we"]), int(cfg["e_sn"]))
    max_by_grid = max(1, (side // 10) ** 2)
    return max(1, min(config.MPI_RANKS, max_by_grid))


def _mpi(exe, cwd, log_path, ranks):
    """Run a WRF executable under MPI with `ranks` processes, falling back to a serial run if
    mpirun fails to launch (e.g. no slots). Mirrors the `mpirun … || ./exe` shell pattern."""
    try:
        _run(["mpirun", "-np", str(ranks), exe], cwd, log_path)
    except WrfRunError:
        _run([exe], cwd, log_path)


# ---------------------------------------------------------------------------
# GFS data
# ---------------------------------------------------------------------------
def _fetch_gfs(run_dir, cfg, log_path):
    """Download a small region-clipped GFS subset (f000 + f003) for the most recent
    available NOMADS cycle. Returns (cycle_datetime_utc, gribdir). GFS on NOMADS is
    retained ~10 days, so we auto-select the latest usable cycle rather than the
    scenario's (possibly out-of-range) requested date."""
    gribdir = os.path.join(run_dir, "grib")
    os.makedirs(gribdir, exist_ok=True)

    half = cfg["box_half_deg"]
    box = (f"leftlon={cfg['lon'] - half:.2f}&rightlon={cfg['lon'] + half:.2f}"
           f"&toplat={cfg['lat'] + half:.2f}&bottomlat={cfg['lat'] - half:.2f}")
    levs = ["lev_surface=on", "lev_2_m_above_ground=on", "lev_10_m_above_ground=on",
            "lev_mean_sea_level=on", "lev_0-0.1_m_below_ground=on",
            "lev_0.1-0.4_m_below_ground=on", "lev_0.4-1_m_below_ground=on",
            "lev_1-2_m_below_ground=on"]
    levs += [f"lev_{mb}_mb=on" for mb in _PRESSURE_MB]
    levstr = "&".join(levs)

    now = dt.datetime.now(dt.timezone.utc)
    for d in range(3):
        day = (now - dt.timedelta(days=d)).strftime("%Y%m%d")
        for cyc in ("18", "12", "06", "00"):
            base = (f"{NOMADS_FILTER}?dir=%2Fgfs.{day}%2F{cyc}%2Fatmos"
                    f"&{box}&{GFS_VARS}&{levstr}")
            # Download both boundary files (f000, f003) concurrently — they are independent
            # and dominated by network latency, so overlapping them roughly halves fetch time.
            jobs = []
            for fh in ("000", "003"):
                fname = f"gfs.t{cyc}z.pgrb2.0p25.f{fh}"
                dest = os.path.join(gribdir, fname)
                url = f"{base}&file={fname}"
                proc = subprocess.Popen(
                    ["curl", "-fsS", "-m", "180", "-o", dest, url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                jobs.append((dest, url, proc))

            ok = True
            with open(log_path, "ab") as log:
                for dest, url, proc in jobs:
                    _, err = proc.communicate()
                    log.write(f"\n$ curl -fsS -m 180 -o {dest} {url}\n".encode())
                    if err:
                        log.write(err)
                    if proc.returncode != 0:
                        ok = False
                        continue
                    with open(dest, "rb") as fh_in:
                        if fh_in.read(4) != b"GRIB":  # HTML error page, not a GRIB file
                            os.remove(dest)
                            ok = False
            if ok:
                return dt.datetime(int(day[0:4]), int(day[4:6]), int(day[6:8]),
                                   int(cyc), tzinfo=dt.timezone.utc), gribdir
    raise WrfRunError(
        "no usable GFS cycle found on NOMADS (retention/outage). The WRF engine is "
        "installed but real boundary data could not be fetched for any recent cycle."
    )


# ---------------------------------------------------------------------------
# WPS: geogrid -> ungrib -> metgrid
# ---------------------------------------------------------------------------
def run_wps(run_dir, cfg):
    """Fetch GFS, write namelist.wps, then geogrid -> link_grib -> ungrib -> metgrid.
    Returns a dict {start_utc, end_utc, cycle, nmet} for the real/wrf stages."""
    log = os.path.join(run_dir, "wps.log")
    cycle, gribdir = _fetch_gfs(run_dir, cfg, log)

    start = cycle
    end = start + dt.timedelta(hours=cfg["run_hours"])
    wps_end = start + dt.timedelta(seconds=cfg["interval_seconds"])  # >=2 met_em levels

    with open(os.path.join(run_dir, "namelist.wps"), "w") as f:
        f.write(nb.wps_namelist(cfg, start, wps_end, config.WRF_GEOG_PATH))

    # Symlink the WPS programs + GFS Vtable into the run dir.
    for name in ("geogrid.exe", "metgrid.exe", "ungrib.exe", "geogrid", "metgrid"):
        _symlink(os.path.join(config.WPS_DIR, name), os.path.join(run_dir, name))
    _symlink(os.path.join(config.WPS_DIR, "ungrib", "Variable_Tables", "Vtable.GFS"),
             os.path.join(run_dir, "Vtable"))

    _run([os.path.join(config.WPS_DIR, "geogrid.exe")], run_dir, log)
    if not os.path.exists(os.path.join(run_dir, "geo_em.d01.nc")):
        raise WrfRunError("geogrid produced no geo_em.d01.nc (see wps.log)")

    link_grib = os.path.join(config.WPS_DIR, "link_grib.csh")
    _run([link_grib] + sorted(glob.glob(os.path.join(gribdir, "gfs.*"))), run_dir, log)
    _run([os.path.join(config.WPS_DIR, "ungrib.exe")], run_dir, log)
    if not glob.glob(os.path.join(run_dir, "FILE:*")):
        raise WrfRunError("ungrib produced no FILE: intermediates (see wps.log)")

    _run([os.path.join(config.WPS_DIR, "metgrid.exe")], run_dir, log)
    met = sorted(glob.glob(os.path.join(run_dir, "met_em.d01.*.nc")))
    if len(met) < 2:
        raise WrfRunError(
            f"metgrid produced {len(met)} met_em time level(s); need >=2 for wrfbdy_d01"
        )
    return {"start_utc": start, "end_utc": end, "cycle": cycle,
            "nmet": _num_metgrid_levels(met[0])}


def _num_metgrid_levels(met_path):
    out = subprocess.run(["ncdump", "-h", met_path], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if "num_metgrid_levels =" in line:
            return int(line.split("=")[1].strip().rstrip(";").strip())
    raise WrfRunError(f"could not read num_metgrid_levels from {met_path}")


# ---------------------------------------------------------------------------
# real.exe
# ---------------------------------------------------------------------------
def run_real(run_dir, cfg, wps_info):
    """Write namelist.input (with detected num_metgrid_levels), link the WRF run-dir
    tables, and run real.exe. Asserts wrfinput_d01 + wrfbdy_d01."""
    log = os.path.join(run_dir, "real.log")
    with open(os.path.join(run_dir, "namelist.input"), "w") as f:
        f.write(nb.input_namelist(cfg, wps_info["start_utc"], wps_info["end_utc"],
                                  wps_info["nmet"]))

    _symlink(os.path.join(config.WRF_DIR, "main", "real.exe"),
             os.path.join(run_dir, "real.exe"))
    _symlink(os.path.join(config.WRF_DIR, "main", "wrf.exe"),
             os.path.join(run_dir, "wrf.exe"))
    _link_run_tables(run_dir)

    _mpi("./real.exe", run_dir, log, _ranks_for(cfg))
    for need in ("wrfinput_d01", "wrfbdy_d01"):
        if not os.path.exists(os.path.join(run_dir, need)):
            raise WrfRunError(f"real.exe produced no {need} (see real.log / rsl.* files)")


# ---------------------------------------------------------------------------
# wrf.exe
# ---------------------------------------------------------------------------
def run_wrf(run_dir, log_path, cfg):
    """Run wrf.exe and return the path to the wrfout it produced (asserting >=2 time
    levels). Any pre-existing wrfout_d01_* must be cleared by the caller first so we can
    unambiguously identify this run's output."""
    _mpi("./wrf.exe", run_dir, log_path, _ranks_for(cfg))
    outs = sorted(glob.glob(os.path.join(run_dir, "wrfout_d01_*")))
    if not outs:
        raise WrfRunError("wrf.exe produced no wrfout_d01_* file (see rsl.error.* files)")
    wrfout = outs[-1]
    if _time_levels(wrfout) < 2:
        raise WrfRunError("wrfout has <2 time levels — the model did not integrate")
    return wrfout


def _time_levels(wrfout):
    out = subprocess.run(["ncdump", "-h", wrfout], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if "Time = UNLIMITED" in line and "currently" in line:
            return int(line.split("(")[1].split("currently")[0].strip())
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _symlink(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def _link_run_tables(run_dir):
    """Link the lookup tables wrf/real expect (TBL, RRTMG_*, ozone, CAM, *_DATA)."""
    rundir_src = os.path.join(config.WRF_DIR, "run")
    patterns = ("*.TBL", "*.formatted", "RRTMG_*", "ozone*", "CAM*", "*_DATA")
    for pat in patterns:
        for src in glob.glob(os.path.join(rundir_src, pat)):
            _symlink(src, os.path.join(run_dir, os.path.basename(src)))


def clear_wrfout(run_dir):
    """Remove any wrfout from a previous run so run_wrf can identify fresh output."""
    for f in glob.glob(os.path.join(run_dir, "wrfout_d01_*")):
        os.remove(f)
