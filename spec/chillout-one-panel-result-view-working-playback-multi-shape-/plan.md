# ChillOut — One-panel result view, working playback, multi-shape targets

## Context

Three problems the user hit, in their words: (1) "Animations do not appear" after a run
completes; (2) "why did you add like 5 tabs on the result box — make the result stats and the
videos in one place"; (3) "the shape should be able to be custom and not only one shape, but a
lot of shapes … clouds have complicated shapes so add that."

Root causes found during investigation:

- **Animation never shows.** Two compounding causes:
  - *Stale cache*: `sim.js`/`sim.css`/`styles.css`/`animations.css`/`player.js` are served
    `cache-control: immutable, max-age=31536000` under unchanging filenames. `index.html` is
    `no-cache` (fresh) and loads the *new* `player.js`, but the browser reuses the year-cached
    *old* `sim.js` — which has no `WrfPlayer.load()` wiring — so nothing ever feeds the player.
    Confirmed live: edge has new code, but returning browsers are pinned to old assets.
  - *Hidden-tab 0px render*: on completion the active tab is **Overview**, so the Maps pane is
    `display:none`. `WrfPlayer.load()` runs while `#playerCanvas` has `clientWidth === 0`
    (reproduced live), so it draws at 0×0 and never re-renders when Maps is later clicked.
- **Tabs add friction.** The result box has 4 tabs (Overview / Maps / Time-series /
  Diagnostics); Time-series is an empty placeholder and Diagnostics only holds the error text.
- **Single shape only.** `setTarget()` removes the previous layer on every draw; the whole
  stack (frontend → schema → worker mask → animation outline) assumes one ring via
  `coordinates[0]`.

Outcome: one consolidated result panel (stats + animation together, no tabs — which also kills
the hidden-tab bug since the canvas is always laid out), assets that actually update, and
support for drawing many polygons as a GeoJSON **MultiPolygon**.

## Approach

### A. Collapse the result box into one panel (fixes the animation visibility too)
- **`site/simulation/index.html`** (result block ~207–270): remove the `.tabs` button row
  (219–223) and the `.tabpane` wrappers (226–263). Keep the `#result` shell, `#resId`,
  `#newRun`, and the `#pipeline` status strip. Replace the panes with a single
  `#resultBody` container, and move the **player markup** (`#player` + `#playerEmpty` +
  canvas/controls/legend, currently 231–254) directly into `#result` so it is always laid out
  (not in a hidden pane). Drop Time-series entirely. Keep an inline `#resultError` region for
  failures (replaces the Diagnostics pane).
- **`site/simulation/sim.js`** `renderResult()` (232–269): write into the single
  `#resultBody`. On **completed**: render the player (call `WrfPlayer.load(r.animation)` while
  the canvas is visible) **and** the `.metrics` grid **and** the summary together in one view.
  On **working/created**: show the existing spinner + `.working-bar` + elapsed timer + the
  `#pipeline` strip. On **failed**: show the error inline in `#resultError`. Remove the tab
  click-handler (308–315). Keep `poll()`, submit, and `newRun` (still call `WrfPlayer.reset()`).
- **`site/simulation/sim.css`**: delete the now-dead `.tabs`/`.tab`/`.tabpane` rules (77–81);
  add layout so the player + metrics + summary stack cleanly in one card.
- **`site/simulation/player.js`**: make rendering robust regardless of timing — in `load()`,
  after `#player` is shown, defer the first `resize()+drawFrame()` to a `requestAnimationFrame`
  so `clientWidth` is non-zero; expose/keep a `resize()` path. With the player no longer in a
  hidden pane this is belt-and-suspenders, but prevents any first-frame 0px draw.

### B. Cache-bust local assets (so redeploys are actually picked up)
- Append a version query string to every **local** `<link>`/`<script>` in
  `site/index.html` (css/styles.css, css/animations.css, js/clouds.js, js/map.js, js/demo.js,
  js/reveal.js) and `site/simulation/index.html` (../css/styles.css, sim.css, player.js,
  sim.js): e.g. `sim.js?v=8`. `index.html` is `no-cache`, so bumping the token forces fresh
  fetches. Leave external unpkg/Leaflet URLs untouched. Bump the token on every future asset
  change. (No build step exists; query-string busting is the pragmatic fit.)

### C. Multiple custom shapes → GeoJSON MultiPolygon
Represent targets as `{ type: "MultiPolygon", coordinates: [ [ring], [ring], … ] }` where each
`coordinates[i][0]` is one area ring `[[lon,lat],…]`. Define one extraction rule reused
everywhere: *MultiPolygon → for each poly take `poly[0]`; Polygon → `coordinates[0]`* (keeps
backward-compat with existing single-Polygon jobs).

- **`site/simulation/sim.js`** (map section ~12–68): stop replacing on draw — in the
  `draw:created` handler add each new layer to `drawnItems` (accumulate, don't call the
  replacing `setTarget`). Enable Leaflet.draw's remove tool (`edit:{ featureGroup: drawnItems,
  remove: true }`) and handle `draw:deleted`/`draw:edited` by re-syncing. `polygonCoords()`:
  iterate `drawnItems.getLayers()` (the domain rect is on `map`, not in `drawnItems`, so it is
  excluded), emit one ring per layer → return MultiPolygon `coordinates`. `buildScenario()`:
  set `target_polygon: { type: "MultiPolygon", coordinates: polygonCoords() }` (empty array →
  whole-domain fallback, unchanged). Update the "Clear target" button to clear all shapes and
  the maphint copy to say multiple areas are allowed.
- **`functions/api/lib/schema.mjs`** `validateScenario()` (85–99): replace the single
  `coordinates[0]` read with the extraction rule; validate **every** area ring (each ≥3 verts,
  every vertex inside the domain bbox). Accept both `Polygon` and `MultiPolygon`. Empty → still
  allowed (whole-domain). **Requires `openkbs fn deploy api`.**
- **`worker/cloud_injector.py`** `_column_mask()` (47–71): extract all area rings; OR the
  `_points_in_ring` masks across them into one boolean column mask; coverage string like
  "N target polygons". Empty/none → existing disk fallback.
- **`worker/postprocessor.py`** (210–212): replace `animation["polygon"] = poly[0]` with
  `animation["polygons"] = <list of all area rings>` (keep emitting `polygon` = first ring for
  back-compat).
- **`site/simulation/player.js`** `drawPolygon()`: read `anim.polygons` (fallback to
  `[anim.polygon]`) and stroke each ring.

## Critical files
- `site/simulation/index.html` — remove tabs; single result body; player moved inline; `?v=`.
- `site/simulation/sim.js` — `renderResult` one-panel; drop tab handler; multi-shape draw +
  `polygonCoords()` + `buildScenario` MultiPolygon.
- `site/simulation/sim.css` — delete tab rules; one-card result layout.
- `site/simulation/player.js` — rAF-safe first render; draw multiple polygon rings.
- `site/index.html` — `?v=` on local assets.
- `functions/api/lib/schema.mjs` — validate Polygon **and** MultiPolygon (all rings).
- `worker/cloud_injector.py` — union mask across all rings.
- `worker/postprocessor.py` — emit `animation.polygons`.

## Verification (end-to-end, real run)
1. Local serve `site/` (`python3 -m http.server 8099`); hard-reload `/simulation/` and confirm
   the warm-light single-panel result box (no tabs) and that local assets fetch with `?v=`.
2. Draw **two+** polygons; confirm both persist, the trash tool removes one, "Clear target"
   clears all, and `buildScenario()` emits a `MultiPolygon` with one ring per shape.
3. Submit; while `created`/running confirm the spinner + animated bar + elapsed timer + pipeline
   show (queued behind other jobs is expected — worker is single-threaded).
4. On completion confirm the **animation plays inline next to the metric cards + summary** in
   one view (no tab click needed), with all drawn polygon outlines drawn over the field, and
   Baseline/Candidate/Δ toggle + scrubber working.
5. Deploy: commit, then `openkbs fn deploy api` (schema), `openkbs site deploy` (frontend),
   and restart the worker (`worker/run_worker.sh`) so `cloud_injector.py`/`postprocessor.py`
   reload. Re-verify on CloudFront with a hard reload; confirm a brand-new run shows the
   animation and multi-shape outline live.

## Out of scope (unless you want it)
- Queue-position indicator ("N runs ahead") in the `created` state — the earlier scope question;
  can fold in if desired.
- Polygon holes/self-intersection validation; MP4/GIF export; time-series charts.
