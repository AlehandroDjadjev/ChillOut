"""Worker configuration — all knobs come from the environment so the same code runs
on this dev VM and on a production OpenKBS Worker EC2. Mirrors the env vars honored by
infra/install_wrf.sh and infra/check_wrf_install.sh."""
import os

# Job queue (same Postgres the Lambda writes to; DATABASE_URL injected by OpenKBS).
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# CDN base for pulling uploaded RL checkpoints. Objects under the `media/` prefix are served
# publicly through CloudFront, so the RL worker fetches checkpoints over HTTPS (the
# `openkbs storage download` CLI path is unreliable on this runtime). Override per-deploy.
STORAGE_CDN_BASE = os.environ.get(
    "STORAGE_CDN_BASE", "https://d31rm64rx9q92u.cloudfront.net"
).rstrip("/")

# WRF/WPS install layout (defaults match infra/install_wrf.sh).
WRF_ROOT = os.environ.get("WRF_ROOT", "/opt/wrf")
WRF_DIR = os.environ.get("WRF_DIR", os.path.join(WRF_ROOT, "WRF"))
WPS_DIR = os.environ.get("WPS_DIR", os.path.join(WRF_ROOT, "WPS"))
WRF_GEOG_PATH = os.environ.get("WRF_GEOG_PATH", os.path.join(WRF_ROOT, "geog"))

# Where per-job working directories and outputs land.
SIMULATION_STORAGE_PATH = os.environ.get(
    "SIMULATION_STORAGE_PATH", os.path.join(WRF_ROOT, "runs")
)

# The five executables check_wrf_install.sh asserts. The worker refuses to claim
# real work unless all of these exist and are executable.
WRF_BINARIES = {
    "wrf.exe": os.path.join(WRF_DIR, "main", "wrf.exe"),
    "real.exe": os.path.join(WRF_DIR, "main", "real.exe"),
    "geogrid.exe": os.path.join(WPS_DIR, "geogrid.exe"),
    "ungrib.exe": os.path.join(WPS_DIR, "ungrib.exe"),
    "metgrid.exe": os.path.join(WPS_DIR, "metgrid.exe"),
}

# Daemon loop.
POLL_INTERVAL_SECONDS = float(os.environ.get("WORKER_POLL_INTERVAL", "5"))
# A job in a running status whose updated_at is older than this lost its worker (VM
# restart, OOM, killed process) and is requeued instead of stranded. Must exceed the
# longest single WRF stage on the clamped budget (a few minutes), with margin.
WORKER_STALE_MINUTES = int(os.environ.get("WORKER_STALE_MINUTES", "20"))
# real.exe/wrf.exe are launched under MPI. This is the upper rank ceiling; the actual rank
# count per run is further capped to the grid size by wrf_runner._ranks_for (WRF errors out
# if a tiny domain is split across too many tiles). 12 leaves headroom on the 16-core VM for
# the OS, NetCDF I/O, and the daemon.
MPI_RANKS = int(os.environ.get("WORKER_MPI_RANKS", str(min(os.cpu_count() or 4, 12))))

# Runtime budget so a job completes in minutes on the worker VM. The grid is clamped to at
# most WORKER_MAX_CELLS points per side, and the integration to WORKER_MAX_HOURS (never
# below one 3-hour GFS boundary interval). The actual values used are recorded in the result.
WORKER_MAX_CELLS = int(os.environ.get("WORKER_MAX_CELLS", "60"))
WORKER_MAX_HOURS = int(os.environ.get("WORKER_MAX_HOURS", "3"))

# Animation payload caps. The postprocessor embeds quantized per-frame grids in the result
# jsonb so the browser can play the run back. Bound the size: at most ANIM_MAX_SIDE cells
# per side and ANIM_MAX_FRAMES time frames (evenly subsampled from what WRF actually wrote).
ANIM_MAX_SIDE = int(os.environ.get("ANIM_MAX_SIDE", "48"))
ANIM_MAX_FRAMES = int(os.environ.get("ANIM_MAX_FRAMES", "24"))
