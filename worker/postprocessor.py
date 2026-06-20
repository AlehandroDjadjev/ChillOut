"""Compares baseline vs candidate wrfout and produces the result payload the frontend
renders: headline metrics + a plain-language summary + an `animation` block the browser
plays back. Static PNG delta maps remain a fast-follow (maps:[]). Honest wording only —
deltas are modelled signal under stated assumptions, never a guaranteed real-world outcome."""
import base64
import datetime as dt

import numpy as np
import netCDF4

import config
from cloud_injector import _column_mask

# How each target variable is read + displayed.
VAR_INFO = {
    "T2":     {"label": "2 m temperature", "unit": "°C", "kelvin": True},
    "TSK":    {"label": "skin temperature", "unit": "°C", "kelvin": True},
    "RAINNC": {"label": "accumulated precip", "unit": "mm", "kelvin": False},
    "SWDOWN": {"label": "downward shortwave", "unit": "W/m²", "kelvin": False},
}


def _times_hours(ds):
    """Hours since the first history frame, parsed from the Times char variable."""
    raw = ds.variables["Times"][:]
    strs = ["".join(c.decode() if isinstance(c, bytes) else c for c in row) for row in raw]
    t0 = dt.datetime.strptime(strs[0], "%Y-%m-%d_%H:%M:%S")
    return np.array([(dt.datetime.strptime(s, "%Y-%m-%d_%H:%M:%S") - t0).total_seconds() / 3600.0
                     for s in strs])


def _window_indices(hours, window):
    """Frame indices inside [w0, w1] h, clamped to what was actually integrated."""
    w0, w1 = float(window[0]), float(window[1])
    idx = np.where((hours >= w0 - 1e-6) & (hours <= w1 + 1e-6))[0]
    if len(idx) == 0:  # window beyond the clamped run length: use the last available frame
        idx = np.array([len(hours) - 1])
    return idx


def _field_mean(ds, var, tidx, colmask):
    """Region-mean of `var` over the masked columns, averaged across the window frames."""
    v = ds.variables[var][:]  # (Time, sn, we)
    sel = v[tidx][:, colmask]
    return float(np.mean(sel))


def _field_grid(ds, var, tidx, colmask):
    """Per-column window-mean (sn, we) with non-target columns set to NaN — for max/min stats."""
    v = ds.variables[var][:]
    g = np.mean(v[tidx], axis=0).astype(float)
    g[~colmask] = np.nan
    return g


def _downsample(grid, max_side):
    """Stride a 2D array down to <= max_side per dimension (cheap nearest-cell decimation)."""
    ny, nx = grid.shape
    sy = max(1, int(np.ceil(ny / max_side)))
    sx = max(1, int(np.ceil(nx / max_side)))
    return grid[::sy, ::sx]


def _build_animation(b, c, var, info, hours):
    """Extract real per-frame 2D grids of `var` from both runs, quantize to uint8 against a
    shared scale, and base64-encode them so the browser can animate the run. Returns the
    `animation` dict (or None if there is nothing worth animating). The whole domain is kept
    (not just the target columns) so the effect can be seen spreading across the map."""
    n = len(hours)
    if n < 2:
        return None

    # Evenly subsample frames down to the cap (keep first and last).
    if n > config.ANIM_MAX_FRAMES:
        fidx = np.unique(np.linspace(0, n - 1, config.ANIM_MAX_FRAMES).round().astype(int))
    else:
        fidx = np.arange(n)

    kelvin = info.get("kelvin", False)

    def frames_for(ds):
        v = ds.variables[var][:]  # (Time, sn, we)
        out = []
        for t in fidx:
            g = _downsample(np.asarray(v[t], dtype=float), config.ANIM_MAX_SIDE)
            if kelvin:
                g = g - 273.15
            out.append(g)
        return out

    bframes = frames_for(b)
    cframes = frames_for(c)
    ny, nx = bframes[0].shape

    # Shared scale across both runs so baseline/candidate/Δ are directly comparable.
    allv = np.concatenate([np.asarray(bframes).ravel(), np.asarray(cframes).ravel()])
    vmin = float(np.nanmin(allv))
    vmax = float(np.nanmax(allv))
    span = vmax - vmin
    if span <= 0:
        span = 1.0  # flat field — avoid divide-by-zero, everything maps to 0

    def quantize(g):
        q = np.clip((g - vmin) / span * 255.0, 0, 255)
        q = np.nan_to_num(q, nan=0.0).astype(np.uint8)
        return base64.b64encode(q.tobytes()).decode("ascii")

    # Geographic corners (from the downsampled-consistent full grid) for the canvas extent.
    lat = _downsample(np.asarray(b.variables["XLAT"][0], dtype=float), config.ANIM_MAX_SIDE)
    lon = _downsample(np.asarray(b.variables["XLONG"][0], dtype=float), config.ANIM_MAX_SIDE)

    return {
        "var": var,
        "label": info["label"],
        "unit": info["unit"],
        "nx": int(nx),
        "ny": int(ny),
        "times_h": [round(float(hours[t]), 3) for t in fidx],
        "scale": {"vmin": round(vmin, 4), "vmax": round(vmax, 4)},
        "bounds": {
            "lat0": round(float(lat.min()), 5), "lat1": round(float(lat.max()), 5),
            "lon0": round(float(lon.min()), 5), "lon1": round(float(lon.max()), 5),
        },
        "frames": {
            "baseline": [quantize(g) for g in bframes],
            "candidate": [quantize(g) for g in cframes],
        },
    }


def build_result(baseline, candidate, scenario, cfg, wps_info, inj):
    """Return the dict stored in simulations.result and shown in the result viewer.

    Shape (consumed by site/simulation/sim.js renderResult):
        { "metrics": [{"label","value"}], "summary": str, "maps": [] }
    """
    var = scenario["objective"]["target_variable"]
    info = VAR_INFO.get(var, {"label": var, "unit": "", "kelvin": False})
    window = scenario["objective"].get("target_time_window_hours", [0, cfg["run_hours"]])
    desired = scenario["objective"].get("desired_delta_c")

    with netCDF4.Dataset(baseline) as b, netCDF4.Dataset(candidate) as c:
        lats = b.variables["XLAT"][0]
        lons = b.variables["XLONG"][0]
        colmask, coverage = _column_mask(lats, lons, scenario)

        hours = _times_hours(b)
        tidx = _window_indices(hours, window)
        win_lo, win_hi = float(hours[tidx[0]]), float(hours[tidx[-1]])

        base_mean = _field_mean(b, var, tidx, colmask)
        cand_mean = _field_mean(c, var, tidx, colmask)
        delta_grid = _field_grid(c, var, tidx, colmask) - _field_grid(b, var, tidx, colmask)

        # Per-frame grids for the browser playback (whole domain, real frames only).
        animation = _build_animation(b, c, var, info, hours)

        # Precip side-check (modification should not blow past the precip cap).
        precip_cand = None
        if "RAINNC" in c.variables:
            precip_cand = _field_mean(c, "RAINNC", [len(hours) - 1], colmask)

    delta_mean = cand_mean - base_mean
    max_cool = float(np.nanmin(delta_grid))
    max_warm = float(np.nanmax(delta_grid))

    def fmt(v, kelvin_to_c=False):
        if kelvin_to_c:
            v = v - 273.15
        return f"{v:.2f}"

    metrics = [
        {"label": f"Mean Δ {info['label']}", "value": f"{delta_mean:+.2f} {info['unit']}"},
        {"label": f"Baseline mean", "value": f"{fmt(base_mean, info['kelvin'])} {info['unit']}"},
        {"label": f"Candidate mean", "value": f"{fmt(cand_mean, info['kelvin'])} {info['unit']}"},
        {"label": "Strongest decrease", "value": f"{max_cool:+.2f} {info['unit']}"},
        {"label": "Strongest increase", "value": f"{max_warm:+.2f} {info['unit']}"},
        {"label": "Cloud columns / cells", "value": f"{inj['n_columns']} / {inj['n_cells']}"},
    ]
    if precip_cand is not None:
        cap = scenario["objective"].get("max_precipitation_mm")
        val = f"{precip_cand:.2f} mm"
        if cap is not None:
            val += f" (cap {cap} mm{'' if precip_cand <= cap else ' — EXCEEDED'})"
        metrics.append({"label": "Candidate precip", "value": val})

    goal = ""
    if desired is not None:
        hit = "meets" if abs(delta_mean - desired) <= abs(desired) * 0.5 + 0.5 else "differs from"
        goal = (f" The objective asked for a {desired:+.1f} {info['unit']} change; the modelled "
                f"mean {delta_mean:+.2f} {info['unit']} {hit} that target.")

    summary = (
        f"Simulated a {info['label']} change by injecting a "
        f"{inj['base_m_agl']:.0f}–{inj['top_m_agl']:.0f} m AGL cloud "
        f"(QCLOUD≈{inj['qcloud_kg_kg']:.2e}, QICE≈{inj['qice_kg_kg']:.2e} kg/kg) over "
        f"{coverage}. Comparison is baseline vs cloud-modified WRF over the "
        f"{win_lo:.0f}–{win_hi:.0f} h window. Driven by real GFS cycle "
        f"{wps_info['cycle']:%Y-%m-%d %HZ}. Domain: {cfg['e_we']}×{cfg['e_sn']} cells at "
        f"{cfg['actual_res_m']} m, integrated {cfg['run_hours']} h"
        + (f" (requested {cfg['requested_hours']} h @ {cfg['requested_res_m']} m, clamped to a "
           f"runtime budget so the demo completes in minutes)."
           if (cfg['run_hours'] != cfg['requested_hours']
               or cfg['actual_res_m'] != cfg['requested_res_m']) else ".")
        + goal
        + " These are modelled deltas under the stated assumptions, not a guaranteed "
          "real-world outcome."
    )

    if animation is not None:
        poly = scenario["region"].get("target_polygon") or {}
        coords = poly.get("coordinates") or []
        # Extract every outer ring: MultiPolygon → coordinates[i][0]; Polygon → coordinates[0].
        if poly.get("type") == "MultiPolygon":
            rings = [p[0] for p in coords if p and len(p) > 0]
        else:
            rings = [coords[0]] if coords else []
        animation["polygons"] = rings
        animation["polygon"] = rings[0] if rings else None  # back-compat: first ring

    return {"metrics": metrics, "summary": summary, "maps": [], "animation": animation}
