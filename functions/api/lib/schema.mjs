/**
 * Scenario contract + validator for ChillOut WRF simulations.
 *
 * The contract mirrors infra/scenario.example.json — the single source of truth shared by
 * the frontend editor and the WRF worker. This validator is dependency-free so the Lambda
 * stays light. It returns { valid, errors: [{ field, reason }] } — never throws on bad input.
 */

// Hard product budgets (from the spec): keep a single WRF run tractable.
export const LIMITS = {
  MAX_DURATION_HOURS: 12,
  MIN_INNER_RESOLUTION_M: 1000,
  MAX_DOMAIN_SIZE_KM: 300,
  MIN_DOMAIN_SIZE_KM: 10,
  MAX_DOMAINS: 3,
  TARGET_VARIABLES: ['T2', 'RAINNC', 'SWDOWN', 'TSK'],
  CLOUD_TYPES: ['low_stratus', 'cumulus', 'stratocumulus', 'altostratus'],
  INPUT_SOURCES: ['GFS'],
};

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const isObj = (v) => v && typeof v === 'object' && !Array.isArray(v);

// Approximate degrees-per-km; lon scaled by latitude. Good enough for a domain-bounds check.
const KM_PER_DEG_LAT = 111.0;
function domainBounds(center, sizeKm) {
  const half = sizeKm / 2;
  const dLat = half / KM_PER_DEG_LAT;
  const cos = Math.cos((center.lat * Math.PI) / 180) || 1e-6;
  const dLon = half / (KM_PER_DEG_LAT * Math.abs(cos));
  return {
    latMin: center.lat - dLat,
    latMax: center.lat + dLat,
    lonMin: center.lon - dLon,
    lonMax: center.lon + dLon,
  };
}

export function validateScenario(s) {
  const errors = [];
  const err = (field, reason) => errors.push({ field, reason });

  if (!isObj(s)) return { valid: false, errors: [{ field: '', reason: 'scenario must be an object' }] };

  // --- region ---
  const region = s.region;
  if (!isObj(region)) {
    err('region', 'missing');
  } else {
    const c = region.domain_center;
    if (!isObj(c) || !isNum(c.lat) || !isNum(c.lon)) {
      err('region.domain_center', 'lat/lon required');
    } else {
      if (c.lat < -90 || c.lat > 90) err('region.domain_center.lat', 'must be within [-90, 90]');
      if (c.lon < -180 || c.lon > 180) err('region.domain_center.lon', 'must be within [-180, 180]');
    }
  }

  // --- domain ---
  const domain = s.domain;
  let sizeKm = null;
  if (!isObj(domain)) {
    err('domain', 'missing');
  } else {
    if (!isNum(domain.inner_resolution_m) || domain.inner_resolution_m < LIMITS.MIN_INNER_RESOLUTION_M) {
      err('domain.inner_resolution_m', `must be ≥ ${LIMITS.MIN_INNER_RESOLUTION_M} m`);
    }
    if (isNum(domain.outer_resolution_m) && isNum(domain.inner_resolution_m) &&
        domain.outer_resolution_m < domain.inner_resolution_m) {
      err('domain.outer_resolution_m', 'must be ≥ inner_resolution_m');
    }
    if (!isNum(domain.inner_domain_size_km)) {
      err('domain.inner_domain_size_km', 'required');
    } else {
      sizeKm = domain.inner_domain_size_km;
      if (sizeKm < LIMITS.MIN_DOMAIN_SIZE_KM || sizeKm > LIMITS.MAX_DOMAIN_SIZE_KM) {
        err('domain.inner_domain_size_km', `must be within [${LIMITS.MIN_DOMAIN_SIZE_KM}, ${LIMITS.MAX_DOMAIN_SIZE_KM}] km`);
      }
    }
    if (isNum(domain.max_domains) && (domain.max_domains < 1 || domain.max_domains > LIMITS.MAX_DOMAINS)) {
      err('domain.max_domains', `must be within [1, ${LIMITS.MAX_DOMAINS}]`);
    }
  }

  // --- target rings ⊆ domain box ---
  // Targets may be one Polygon or many shapes (MultiPolygon). Extract every outer ring:
  //   MultiPolygon → coordinates[i][0] for each poly;  Polygon → coordinates[0].
  // Empty (no rings) is allowed — whole-domain target; the worker treats it as full coverage.
  if (isObj(region) && isObj(region.domain_center) && isNum(region.domain_center.lat) &&
      isNum(region.domain_center.lon) && isNum(sizeKm)) {
    const poly = region.target_polygon;
    const coords = poly && Array.isArray(poly.coordinates) ? poly.coordinates : [];
    const rings = poly && poly.type === 'MultiPolygon'
      ? coords.map((p) => (Array.isArray(p) ? p[0] : null)).filter(Boolean)
      : (coords[0] ? [coords[0]] : []);
    if (rings.length > 0) {
      const b = domainBounds(region.domain_center, sizeKm);
      const bad = rings.some((ring) => {
        if (!Array.isArray(ring) || ring.length < 3) return true;
        return ring.some(
          (pt) => !Array.isArray(pt) || pt.length < 2 ||
            pt[0] < b.lonMin || pt[0] > b.lonMax || pt[1] < b.latMin || pt[1] > b.latMax
        );
      });
      if (bad) err('region.target_polygon', 'a target area is malformed or extends outside the WRF inner domain');
    }
  }

  // --- time ---
  const time = s.time;
  let durationH = null;
  if (!isObj(time)) {
    err('time', 'missing');
  } else {
    if (typeof time.start_utc !== 'string' || Number.isNaN(Date.parse(time.start_utc))) {
      err('time.start_utc', 'must be an ISO-8601 UTC timestamp');
    }
    if (!isNum(time.duration_hours) || time.duration_hours <= 0) {
      err('time.duration_hours', 'must be > 0');
    } else if (time.duration_hours > LIMITS.MAX_DURATION_HOURS) {
      err('time.duration_hours', `must be ≤ ${LIMITS.MAX_DURATION_HOURS} h`);
    } else {
      durationH = time.duration_hours;
    }
    if (isNum(time.output_interval_minutes) && time.output_interval_minutes <= 0) {
      err('time.output_interval_minutes', 'must be > 0');
    }
  }

  // --- input_data ---
  const input = s.input_data;
  if (!isObj(input)) {
    err('input_data', 'missing');
  } else if (!LIMITS.INPUT_SOURCES.includes(input.source)) {
    err('input_data.source', `must be one of ${LIMITS.INPUT_SOURCES.join(', ')}`);
  }

  // --- objective ---
  const obj = s.objective;
  if (!isObj(obj)) {
    err('objective', 'missing');
  } else {
    if (!LIMITS.TARGET_VARIABLES.includes(obj.target_variable)) {
      err('objective.target_variable', `must be one of ${LIMITS.TARGET_VARIABLES.join(', ')}`);
    }
    if (isNum(obj.max_precipitation_mm) && obj.max_precipitation_mm < 0) {
      err('objective.max_precipitation_mm', 'must be ≥ 0');
    }
    const w = obj.target_time_window_hours;
    if (Array.isArray(w) && w.length === 2 && isNum(w[0]) && isNum(w[1])) {
      if (w[0] < 0 || w[1] <= w[0]) err('objective.target_time_window_hours', 'must be [start, end] with end > start ≥ 0');
      else if (durationH != null && w[1] > durationH) err('objective.target_time_window_hours', 'window end exceeds time.duration_hours');
    }
  }

  // --- bad_scenario (baseline) ---
  if (!isObj(s.bad_scenario) || s.bad_scenario.type !== 'baseline') {
    err('bad_scenario.type', "must be 'baseline'");
  }

  // --- good_scenario (cloud modification) ---
  const good = s.good_scenario;
  if (!isObj(good)) {
    err('good_scenario', 'missing');
  } else if (good.type !== 'cloud_modification') {
    err('good_scenario.type', "must be 'cloud_modification'");
  } else {
    const cl = good.cloud;
    if (!isObj(cl)) {
      err('good_scenario.cloud', 'missing');
    } else {
      if (cl.cloud_type !== undefined && !LIMITS.CLOUD_TYPES.includes(cl.cloud_type)) {
        err('good_scenario.cloud.cloud_type', `must be one of ${LIMITS.CLOUD_TYPES.join(', ')}`);
      }
      const base = cl.base_height_m_agl;
      const top = cl.top_height_m_agl;
      if (!isNum(base) || base < 0) err('good_scenario.cloud.base_height_m_agl', 'must be ≥ 0');
      if (!isNum(top) || top < 0) err('good_scenario.cloud.top_height_m_agl', 'must be ≥ 0');
      if (isNum(base) && isNum(top) && base >= top) err('good_scenario.cloud.top_height_m_agl', 'must be > base_height_m_agl');
      if (!isNum(cl.cloud_fraction) || cl.cloud_fraction < 0 || cl.cloud_fraction > 1) {
        err('good_scenario.cloud.cloud_fraction', 'must be within [0, 1]');
      }
      if (isNum(cl.liquid_water_mixing_ratio_kg_kg) && cl.liquid_water_mixing_ratio_kg_kg < 0) {
        err('good_scenario.cloud.liquid_water_mixing_ratio_kg_kg', 'must be ≥ 0');
      }
      if (isNum(cl.ice_mixing_ratio_kg_kg) && cl.ice_mixing_ratio_kg_kg < 0) {
        err('good_scenario.cloud.ice_mixing_ratio_kg_kg', 'must be ≥ 0');
      }
      const off = cl.start_offset_hours;
      const cdur = cl.duration_hours;
      if (isNum(off) && off < 0) err('good_scenario.cloud.start_offset_hours', 'must be ≥ 0');
      if (isNum(cdur) && cdur <= 0) err('good_scenario.cloud.duration_hours', 'must be > 0');
      if (isNum(off) && isNum(cdur) && durationH != null && off + cdur > durationH) {
        err('good_scenario.cloud.duration_hours', 'cloud window (start_offset + duration) exceeds time.duration_hours');
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

// Ordered status pipeline shared with the worker + frontend.
export const STATUSES = [
  'created', 'validating', 'preprocessing', 'initializing',
  'running_baseline', 'running_candidate', 'postprocessing', 'completed', 'failed',
];
