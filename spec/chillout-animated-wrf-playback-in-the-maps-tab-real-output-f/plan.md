# ChillOut — Animated WRF playback in the Maps tab (real output, fast-forwarded)

## Context

The simulation now runs the **real WRF model** end-to-end (baseline vs. cloud-modified) and
returns honest metrics + a summary. But the user only sees number cards — and WRF actually
produces a **time series of full 2D grids** (one frame every `output_interval_minutes`), which
`postprocessor.py` currently *collapses to scalars* (`_field_mean`, line 39) and throws away.
The user wants to *watch* the run: a fast-forwarded animation built from the real model output —
the cloud appearing, the cooling spreading and fading over the hours, baseline vs. candidate.

The "Maps" tab already exists as a placeholder (`index.html:230`) and was the parked fast-follow.
This delivers it as an **interactive canvas animation** (user's chosen style): Play/pause, a time
scrubber, and a **Baseline / Candidate / Δ** layer toggle, for the objective's target variable.

**Key architectural finding (keeps this simple + robust):** the result already travels
worker → Postgres `result` jsonb → `getResults` → `sim.js`. So we **embed the frame grids in the
result payload** (quantized + base64, size-capped) — *no* S3/boto3, *no* ffmpeg, *no* matplotlib
(so the numpy/netCDF4 ABI pin in `requirements.txt` is untouched), and *no* Lambda change. The
browser decodes the grids and animates them on a `<canvas>`, reusing the warm `heatColor` ramp
already in `site/js/map.js:27`.

Honesty rule preserved: we animate **only real model frames**. Playback defaults to real
keyframes; any smoothing is optional, off by default, and labelled "interpolated".

## Approach

### A. Worker — emit frame grids (`worker/postprocessor.py`)
Add a `_build_animation(b, c, scenario, var, info, hours)` helper and call it in `build_result`,
adding an `"animation"` key to the returned dict (alongside the existing `metrics`/`summary`;
keep `maps: []`). Backwards compatible — older results without `animation` just don't show the player.

- Read the **full-domain** 2D field `var` for **all** time frames from both `baseline` and
  `candidate` (not the column-masked subset — we want to see the effect spread across the map).
- **Downsample** each frame to ≤ `ANIM_MAX_SIDE` (default 48) per side by integer striding.
- **Subsample frames** to ≤ `ANIM_MAX_FRAMES` (default 24) evenly if the run wrote more.
- Convert Kelvin→°C when `info["kelvin"]`.
- Compute a **shared scale** `[vmin, vmax]` across baseline+candidate so both layers and the
  Δ are comparable; **quantize** each frame to `uint8` against that scale, `base64`-encode.
- Return:
  ```
  animation = {
    "var": var, "label": info["label"], "unit": info["unit"],
    "nx": nx, "ny": ny, "times_h": [...],          # real frame hours
    "scale": {"vmin": .., "vmax": ..},             # for baseline/candidate dequant
    "bounds": {"lat0":..,"lat1":..,"lon0":..,"lon1":..},  # from XLAT/XLONG corners
    "polygon": [[lon,lat],...] or null,            # target ring, for canvas outline
    "frames": {"baseline": [b64,...], "candidate": [b64,...]}
  }
  ```
  Δ is derived client-side (candidate−baseline after dequant) — no separate storage.
- Guard: if `< 2` frames exist, set `animation = null` (nothing to animate) and the player hides.
- Payload budget: 48×48×24×2 bytes ≈ 110 KB raw → ~150 KB base64; fine for jsonb + the
  Lambda/CloudFront response. Caps bound it; document them.

### B. Worker config (`worker/config.py`)
Add `ANIM_MAX_SIDE` (48) and `ANIM_MAX_FRAMES` (24) env-tunable knobs, mirroring the existing
`WORKER_MAX_*` style.

### C. Frontend player (`site/simulation/`)
- **`index.html`**: replace the Maps placeholder (line ~230) with player markup — a `<canvas>`,
  a Play/pause button, a `<input type=range>` scrubber, a segmented **Baseline | Candidate | Δ**
  toggle, a time readout (`t = +X.X h`), and a legend (colour ramp + min/max).
- **New `site/simulation/player.js`** (kept separate from `sim.js`): `WrfPlayer(canvas, controls)`
  with `load(animation)`, decode base64→`Uint8Array` per frame, render to an offscreen
  `ImageData` (nx×ny) then `drawImage`-scale onto the visible canvas (nearest-neighbour, so cells
  read as model grid boxes). Two ramps: **absolute** layers reuse a warm temp ramp derived from
  `heatColor`; **Δ** uses a diverging ramp (cool blue ↓ / warm terracotta ↑) centred on 0 with a
  symmetric scale. Draw the target polygon outline + a t-readout. rAF play loop honouring
  `prefers-reduced-motion` (start paused, single frame).
- **`sim.js`**: in `renderResult` completed branch, if `data.result.animation`, reveal the Maps
  tab content and call the player's `load(...)`; on `newRun`, stop/reset it. Tabs already switch
  panes (line 303) — no change there.
- **`sim.css`**: styles for the player controls/legend/canvas in the warm-light theme.

### D. (Optional, small) smoother playback
Leave `output_interval_minutes` to the user. Client offers an off-by-default "smooth" checkbox
that linearly tweens between real keyframes for display only, clearly labelled "interpolated".
Real frames remain the data of record. (Can drop this if you'd rather keep it strictly stepwise.)

## Critical files
- `worker/postprocessor.py` — add `_build_animation` + `animation` key (reuses `_times_hours`,
  `VAR_INFO`, the open `netCDF4.Dataset` handles `b`/`c`).
- `worker/config.py` — `ANIM_MAX_SIDE`, `ANIM_MAX_FRAMES`.
- `site/simulation/index.html` — Maps-tab player markup.
- `site/simulation/player.js` — **new** canvas player (decode + animate + ramps).
- `site/simulation/sim.js` — wire `result.animation` into the Maps tab; reset on New run.
- `site/simulation/sim.css` — player styling.
- Reuse: `site/js/map.js:27` `heatColor` ramp; `getResults` passthrough needs **no** API change.

## Verification (end-to-end, real run)
1. Restart the worker (`worker/run_worker.sh`) so it picks up the new `postprocessor.py`.
2. Submit a scenario via the API; confirm the completed `result` jsonb now contains an
   `animation` object with two non-empty frame arrays and sane `scale`/`bounds`
   (curl `getResults`, check sizes).
3. `agent-browser`: load `/simulation`, draw a polygon, submit, wait for completion; open the
   **Maps** tab — confirm the canvas plays, Play/pause + scrubber work, and the **Δ** view shows
   the cooling pattern over the target area changing across frames. Confirm baseline≠candidate.
4. Confirm warm-light theme + legend render correctly; reduced-motion starts paused.
5. Commit, then `openkbs site deploy` (frontend) — worker change is live on the VM already.
   No `fn deploy` needed (API unchanged).

## Out of scope (next)
Static downloadable MP4/GIF export (Option B), Time-series line charts, multi-variable overlay
(QCLOUD layer) — easy adds once the player exists.
