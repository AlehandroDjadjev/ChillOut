"""Applies the good_scenario cloud to wrfinput_d01 in place, producing the candidate run's
initial conditions. The target polygon selects horizontal grid columns; base/top heights
select model levels within each column; the liquid/ice mixing ratios are written into
QCLOUD/QICE (with QVAPOR nudged toward saturation so the added condensate is physically
plausible rather than instantly evaporated).

This is the scientific heart of the 'cloud modification'. It must never silently no-op: if
it cannot select a single cell it raises WrfRunError, so a job can never report a
cloud-modified 'completed' that changed nothing."""
import numpy as np
import netCDF4

from wrf_runner import WrfRunError

G = 9.81  # m s^-2, for geopotential -> geometric height


def _points_in_ring(lons, lats, ring):
    """Vectorised ray-casting point-in-polygon. `ring` is a list of [lon, lat] vertices.
    Returns a boolean array matching the shape of `lons`/`lats`. Pure numpy so the worker
    needs no shapely/matplotlib (which would drag in a numpy ABI conflict with netCDF4)."""
    xs = np.asarray([p[0] for p in ring], dtype=float)
    ys = np.asarray([p[1] for p in ring], dtype=float)
    px = lons.ravel()
    py = lats.ravel()
    inside = np.zeros(px.shape, dtype=bool)
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        crosses = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi
        )
        inside ^= crosses
        j = i
    return inside.reshape(lons.shape)


def _saturation_qvapor(t_k, p_pa):
    """Approximate saturation mixing ratio (kg/kg) via Tetens over water. Used only to nudge
    QVAPOR up so injected condensate isn't immediately evaporated; not a precise scheme."""
    t_c = t_k - 273.15
    es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))  # Pa
    return 0.622 * es / np.maximum(p_pa - es, 1.0)


def _column_mask(lats, lons, scenario):
    """Boolean (south_north, west_east) mask of columns inside the target polygon. Falls back
    to a disk around domain_center when the polygon is empty (a whole-domain target), and
    reports which path was taken so the result wording stays honest."""
    poly = scenario["region"].get("target_polygon") or {}
    coords = poly.get("coordinates") or []
    # Targets may be many shapes (MultiPolygon) or one (Polygon). Extract every outer ring:
    #   MultiPolygon → coordinates[i][0];  Polygon → coordinates[0].
    if poly.get("type") == "MultiPolygon":
        rings = [p[0] for p in coords if p and len(p) > 0]
    else:
        rings = [coords[0]] if coords else []
    rings = [r for r in rings if r and len(r) >= 3]

    if rings:
        # OR the per-ring masks so the target is the union of all drawn shapes.
        mask = np.zeros(lons.shape, dtype=bool)
        for r in rings:
            mask |= _points_in_ring(lons, lats, r)  # ring is [[lon, lat], ...]
        if mask.any():
            n = len(rings)
            return mask, (f"{n} target polygons" if n > 1 else f"target polygon ({len(rings[0])} vertices)")

    # Fallback: disk around the domain center sized to a quarter of the inner domain.
    center = scenario["region"]["domain_center"]
    size_km = float(scenario["domain"]["inner_domain_size_km"])
    radius_deg = (size_km / 4.0) / 111.0
    clat = float(center["lat"])
    clon = float(center["lon"])
    coslat = np.cos(np.radians(clat)) or 1e-6
    d2 = ((lats - clat)) ** 2 + ((lons - clon) * coslat) ** 2
    mask = d2 <= radius_deg ** 2
    if not mask.any():
        mask = d2 == d2.min()  # at least the single nearest cell
    return mask, f"disk r≈{radius_deg * 111.0:.0f} km around domain center (no polygon drawn)"


def inject_cloud(wrfinput_path, scenario):
    """Write the scenario cloud into wrfinput_d01. Returns a dict describing what was changed
    (coverage, column/level counts) for the result summary."""
    cloud = scenario["good_scenario"]["cloud"]  # validated upstream
    base = float(cloud["base_height_m_agl"])
    top = float(cloud["top_height_m_agl"])
    frac = float(cloud.get("cloud_fraction", 1.0))
    qc_val = float(cloud.get("liquid_water_mixing_ratio_kg_kg", 0.0)) * frac
    qi_val = float(cloud.get("ice_mixing_ratio_kg_kg", 0.0)) * frac

    with netCDF4.Dataset(wrfinput_path, "r+") as ds:
        lats = ds.variables["XLAT"][0]   # (sn, we)
        lons = ds.variables["XLONG"][0]
        hgt = ds.variables["HGT"][0]     # (sn, we) terrain
        ph = ds.variables["PH"][0]       # (bt_stag, sn, we)
        phb = ds.variables["PHB"][0]

        # Geometric height (m, MSL) on staggered levels, then mass-level height AGL.
        z_stag = (ph + phb) / G
        z_mass = 0.5 * (z_stag[:-1] + z_stag[1:])  # (bt, sn, we)
        z_agl = z_mass - hgt[np.newaxis, :, :]

        col_mask, coverage = _column_mask(lats, lons, scenario)
        # 3D mask: inside polygon column AND within [base, top] height band.
        band = (z_agl >= base) & (z_agl <= top)
        mask3d = band & col_mask[np.newaxis, :, :]
        n_cells = int(mask3d.sum())
        if n_cells == 0:
            raise WrfRunError(
                f"cloud injection selected 0 cells: no model levels fall in "
                f"{base:.0f}-{top:.0f} m AGL within the target region. "
                "Widen the height band or the polygon."
            )

        qcloud = ds.variables["QCLOUD"][0]
        qice = ds.variables["QICE"][0]
        qvapor = ds.variables["QVAPOR"][0]

        qcloud[mask3d] = np.maximum(qcloud[mask3d], qc_val)
        qice[mask3d] = np.maximum(qice[mask3d], qi_val)

        # Nudge vapor toward saturation in the cloud body so it doesn't evaporate at t=0.
        if "T" in ds.variables and "P" in ds.variables and "PB" in ds.variables:
            theta = ds.variables["T"][0] + 300.0          # perturbation theta + base state
            pres = ds.variables["P"][0] + ds.variables["PB"][0]
            tk = theta * (pres / 1.0e5) ** 0.2854
            qsat = _saturation_qvapor(tk, pres)
            qvapor[mask3d] = np.maximum(qvapor[mask3d], qsat[mask3d])

        ds.variables["QCLOUD"][0] = qcloud
        ds.variables["QICE"][0] = qice
        ds.variables["QVAPOR"][0] = qvapor

        n_cols = int(col_mask.sum())

    return {
        "coverage": coverage,
        "n_columns": n_cols,
        "n_cells": n_cells,
        "base_m_agl": base,
        "top_m_agl": top,
        "qcloud_kg_kg": qc_val,
        "qice_kg_kg": qi_val,
    }
