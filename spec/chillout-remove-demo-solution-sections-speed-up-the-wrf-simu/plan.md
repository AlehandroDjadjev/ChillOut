# ChillOut — Remove Demo/Solution sections + speed up the WRF simulation

## Context

Two requests from the user, verbatim: *"Remove the demo and solution tabs from the
site (remove them), and make the simulation faster because at the moment it takes around
6-8 minutes to make the simulation."*

1. **Landing page cleanup.** `site/index.html` has two large interactive sections —
   **Solution** (a canned animated "solution map") and **Demo** (a fake in-page
   mini-simulator) — that predate the real Simulation Studio at `simulation/`. They are now
   redundant with (and weaker than) the real product, so the user wants them gone entirely:
   sections, nav links, footer links, the now-orphaned JS, and the dead CSS.

2. **Simulation is slow (~6-8 min).** Investigation found the wall-clock is dominated by the
   two WRF integrations (baseline `wrf.exe` + candidate `wrf.exe`, plus `real.exe`) which run
   via `mpirun -np {MPI_RANKS}` where `MPI_RANKS` is capped at **4** (`worker/config.py:38`,
   `min(os.cpu_count() or 4, 4)`) — but the worker VM has **16 cores** (`nproc`=16). A second,
   smaller cost is the GFS input fetch, which downloads its two boundary files (f000, f003)
   **sequentially** (`worker/wrf_runner.py:93-106`).

   The user chose **"Parallelism only"** — use more cores + parallelize the download, with
   **no loss of scientific fidelity** (same grid resolution cap, same run length). We do NOT
   touch `WORKER_MAX_CELLS` (60) or `run_hours` (3). Expected ~1.5-2× faster.

   Critical safety constraint: WRF's domain decomposition aborts if too many MPI ranks are
   thrown at a small grid (each tile becomes too small). Since the grid is coarsened down to
   ≤60 cells/side, we must cap ranks **per grid size**, not blindly at 16.

## Approach

### A. Remove the Demo and Solution sections (frontend only)

All in `site/index.html` unless noted. Work top-to-bottom so line numbers stay valid (edit
by content match, not line number).

- **Nav links** (~lines 39-40): delete `<a href="#solution">Solution</a>` and
  `<a href="#demo">Demo</a>`. Keep Problem, Use cases, Vision, and the
  `Launch Simulator` → `simulation/` CTA (line 44).
- **Hero "Try Demo" CTA** (~lines 66-69): currently `href="#demo"`. **Repoint to the real
  product** — change to `href="simulation/"` and relabel to "Try the Simulator" (keeps a
  primary hero action; the `#demo` target is being deleted). Leave "Learn More" → `#problem`
  (line 70) untouched.
- **Solution section** (~lines 146-173, `id="solution"`, contains `#solutionMap`): delete the
  whole `<section>`.
- **Demo section** (~lines 176-246, `id="demo"`, contains `#demoMap`, `#demoForm`,
  `#simulateBtn`, `#demoResults`): delete the whole `<section>`.
- **Footer links** (~lines 312-313): delete `<a href="#solution">` and `<a href="#demo">`.
- **Script tags** (~lines 324-327):
  - Remove `js/demo.js` (demo-only).
  - Remove `js/map.js` — it only ever rendered `#solutionMap` + `#demoMap`, both now gone.
    (Verify no other file references `map.js`/its globals before deleting the tag; grep first.)
  - Keep `clouds.js` (hero animation) and `reveal.js` (global scroll reveals).
- **Delete the orphaned source files**: `site/js/demo.js` and `site/js/map.js`.
- **Dead CSS in `site/css/styles.css`** — remove rules now unused: `.solution__grid`
  (~400-405), `.ticks/.tick` (~406-419), `.solution__map` (~420-424), `#solutionMap,#demoMap`
  (~445-450), and the demo block `.demo*/.field*/.select-wrap*/.result*/.conf*` + range-input
  styling (~455-579). For `.map-caption/.legend/.swatch` (~425-444): grep the remaining HTML —
  delete only if nothing else uses them. (When in doubt, leave a CSS rule rather than risk
  removing one still in use; dead HTML is the real cleanup.)
- After editing, **bump every local asset `?v=9` → `?v=10`** in `site/index.html` (and the
  `simulation/` pages too, so the cache-bust is consistent across the deploy).

### B. Speed up the simulation — parallelism only, zero fidelity change (worker)

**B1. Use more CPU cores for the WRF integrations, capped per grid size.**
- `worker/config.py:38`: raise the rank ceiling from 4. New default:
  `MPI_RANKS = int(os.environ.get("WORKER_MPI_RANKS", str(min(os.cpu_count() or 4, 12))))`
  (12, not 16 — leave headroom for the OS, NetCDF I/O threads, and the daemon).
- **Grid-aware cap** so a coarsened small grid never over-decomposes and aborts `wrf.exe`.
  Add a helper in `worker/wrf_runner.py` and apply it where `real.exe`/`wrf.exe` launch:
  ```python
  def _ranks_for(cfg):
      # WRF wants each tile to keep a healthy patch; ~>=10 cells/side per rank.
      side = min(int(cfg["e_we"]), int(cfg["e_sn"]))
      max_by_grid = max(1, (side // 10) ** 2)   # e.g. 60 cells -> 36; 30 -> 9; 20 -> 4
      return max(1, min(config.MPI_RANKS, max_by_grid))
  ```
  Change `_mpi()` (lines 56-62) to take an explicit `ranks` argument
  (`mpirun -np {ranks} ...`), and have `run_real` (169-187) and `run_wrf` (192-203) pass
  `_ranks_for(cfg)`. Keep the existing serial-fallback-on-launch-failure behaviour.
  This is the single biggest win: both `wrf.exe` runs + `real.exe` go from 4 → up to ~12
  ranks (grid permitting), with no change to physics, resolution, or run length.

**B2. Download the two GFS boundary files in parallel.**
- `worker/wrf_runner.py:93-106`: the inner `for fh in ("000","003")` loop curls sequentially.
  Launch both `curl -fsS -m 180` downloads concurrently (e.g. start both as background
  subprocesses then `wait`/`communicate`, or a 2-worker `ThreadPoolExecutor`), then keep the
  **existing** success/validation checks per file. Same files, same source — just overlapped.
  Preserve the current "try next cycle/day on failure" outer fallback semantics: only treat the
  pair as successful if BOTH files arrive valid; otherwise fall through to the next candidate
  cycle exactly as today.

**No fidelity-affecting changes**: `WORKER_MAX_CELLS` (60), `WORKER_MAX_HOURS`/`run_hours` (3),
timestep formula, GFS 3-h boundary interval, and all namelist physics stay exactly as they are.

## Critical files
- `site/index.html` — remove nav/footer links, Solution + Demo sections; repoint hero CTA to
  `simulation/`; drop `demo.js`/`map.js` script tags; bump `?v=` to 10.
- `site/js/demo.js`, `site/js/map.js` — delete (orphaned).
- `site/css/styles.css` — delete dead Solution/Demo rules (grep-verify shared classes first).
- `site/simulation/index.html` — bump local `?v=9` → `?v=10` for consistent cache-bust.
- `worker/config.py` — raise `MPI_RANKS` ceiling 4 → 12.
- `worker/wrf_runner.py` — add `_ranks_for(cfg)`; `_mpi(ranks, ...)`; parallel GFS download.

## Verification

**Frontend (local, before deploy):**
1. `python3 -m http.server 8099` in `site/`; hard-reload `http://localhost:8099/`.
2. Confirm: no Solution or Demo sections; nav shows only Problem / Use cases / Vision /
   Launch Simulator; footer has no Solution/Demo links; hero "Try the Simulator" button
   navigates to `simulation/`; "Learn More" still scrolls to Problem.
3. Open devtools Network: confirm no 404s, `demo.js`/`map.js` no longer requested, hero clouds
   + scroll reveals still work, and local assets fetch with `?v=10`.

**Worker (real end-to-end run, the real test of the speedup):**
4. Apply worker changes, restart the daemon
   (`nohup bash worker/run_worker.sh > /tmp/worker.log 2>&1 &`).
5. Submit a job from the live Simulation Studio (default Stara Zagora scenario). Watch
   `/tmp/worker.log`: confirm `mpirun -np <N>` shows N>4 (grid-capped value), both GFS files
   download concurrently, and **no WRF decomposition/abort error**.
6. Confirm the job reaches `completed` with a valid baseline-vs-candidate result + animation
   (i.e. the speedup didn't break correctness), and note the new wall-clock vs the ~6-8 min
   baseline (target ~3-5 min).
7. If `wrf.exe` aborts on decomposition, the grid-aware cap is too high — lower the
   `side // 10` divisor (e.g. to `side // 14`) and re-run. Serial fallback in `_mpi()` remains
   the safety net.

**Deploy (commit BEFORE each deploy — version tag reuses the commit message):**
8. Commit. Then `openkbs site deploy` (frontend). Worker is not deployed via CLI — the restart
   in step 4 is what picks up `config.py`/`wrf_runner.py`. No `fn deploy` needed (API/schema
   unchanged). Re-verify on CloudFront (`https://d1r38lp75h40vp.cloudfront.net`) with a hard
   reload.

## Out of scope
- Coarser grid / shorter run window (the fidelity-tradeoff speed lever) — user chose
  parallelism only.
- Any API/schema (`functions/`) changes — none needed.
- Queue-position indicator; MP4/GIF export; time-series charts.
