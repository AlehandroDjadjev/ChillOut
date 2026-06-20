# ChillOut — Fix WRF playback not showing after a run

## Context

The user runs a simulation; when it completes, the Maps-tab animation doesn't play.
Two compounding root causes, **both reproduced against the live site**:

1. **Stale browser cache.** `simulation/index.html` is served `no-cache` (always fresh),
   but `sim.js` / `sim.css` / `../css/styles.css` / `css/animations.css` / `player.js` are
   served `cache-control: public, max-age=31536000, immutable` under **unchanging filenames**.
   A returning visitor's browser reuses the year-old `sim.js` — which predates the `WrfPlayer`
   wiring — so `renderResult` never calls `window.WrfPlayer.load()` on completion. (`player.js`
   is a new filename so it loads, but it's never fed data.) This is also why the user still
   sees the **old dark theme + old "created" text with no spinner** — those come from the
   year-cached old `styles.css` / `sim.js`.

2. **Hidden-tab render bug.** On completion the active tab is **Overview**, so the Maps pane
   is `display:none`. `WrfPlayer.load()` runs while the canvas `clientWidth` is **0**
   (confirmed live), so `resize()` falls back to a 480px backing store and `drawFrame()`
   paints at 0×0. Nothing re-renders when the user later clicks **Maps**, so the canvas shows
   blank or a wrongly-sized (480px, left-aligned) frame — only partially filled by autoplay,
   and fully blank under `prefers-reduced-motion`.

Separately (not a bug): a job legitimately sits in `created` because the worker is
single-threaded (`claim_next_job` → `LIMIT 1`) and each WRF run takes minutes — at the time of
the screenshot job `ca53ebc9` was actively running and `e780fb10` was correctly queued behind
it. A "Queued — N ahead" indicator is noted as an optional follow-up, not core to this fix.

## Approach

### 1. Cache-bust local assets — so the new code actually reaches returning users
Append a version query to every **local** CSS/JS reference in the HTML. Because the HTML is
`no-cache`, bumping the token forces browsers to fetch fresh copies of the immutable assets.
- `site/simulation/index.html` (lines 18,19,276,277): `../css/styles.css`, `sim.css`,
  `player.js`, `sim.js` → add `?v=8`
- `site/index.html` (lines 19,20,323-326): `css/styles.css`, `css/animations.css`,
  `js/clouds.js`, `js/map.js`, `js/demo.js`, `js/reveal.js` → add `?v=8`
- Leave the external unpkg Leaflet `<link>`/`<script>` refs unchanged.
- Add a one-line HTML comment noting to bump `?v=` whenever these assets change.

### 2. Fix the hidden-tab render — so the player draws when Maps is revealed
- `site/simulation/player.js`: expose a `refresh()` method on the returned `WrfPlayer` API
  that, when an animation is loaded, calls `resize()` then `drawFrame()` (reuses existing
  `resize` at lines 79-86 and `drawFrame` at 95-122).
- `site/simulation/sim.js`: in the existing tab-click handler (lines 308-315), after a pane is
  activated, if the activated tab is **Maps** and `window.WrfPlayer` exists, call
  `window.WrfPlayer.refresh()`. This re-sizes the canvas to the now-visible stage and repaints,
  deterministically fixing the blank/mis-sized canvas regardless of when `load()` ran.
- Net effect: completing a sim on Overview then clicking Maps yields a correctly sized,
  populated canvas; reduced-motion users see a visible frame 0 instead of blank.

**Critical files:** `site/simulation/player.js`, `site/simulation/sim.js`,
`site/simulation/index.html`, `site/index.html`.

## Verification (end-to-end, live)
1. `openkbs site deploy`, then `agent-browser` against the **live** CloudFront URL:
   - With a warm cache (simulate returning user): hard-load once, then normal load — confirm
     `sim.js?v=8` is fetched and the **warm-light** theme + spinner now appear.
   - Load a completed job (or submit one), **stay on Overview through completion**, then click
     **Maps** → canvas fills the stage and animates; scrubber, Baseline/Candidate/Δ toggle, and
     play/pause all work.
   - Emulate `prefers-reduced-motion: reduce` → player shows a visible paused frame 0, not blank.
2. `curl` the deployed `simulation/index.html` → confirms `sim.js?v=8` / `player.js?v=8`, and
   the referenced `sim.js` contains the `WrfPlayer.load()` wiring.
3. Commit (conventional style), then deploy. **Frontend only — worker and API are unchanged**,
   so no `fn deploy`.

## Out of scope (optional follow-up)
- "Queued — N run(s) ahead" indicator (needs an API queue-count + `getResults` change +
  `fn deploy`) so a job waiting behind another run reads as in-line rather than stuck.
- Static MP4/GIF export; header logo/tabs overlap cosmetic fix.
