# Bring up ChillOut WRF app on project 30dd4f8523d0

## Context
The deployed Simulation Studio (https://d1r38lp75h40vp.cloudfront.net/simulation/) was built for a
different OpenKBS project (`5ca3e61fce09`); its frontend still calls that project's Lambda
(`n2gmhl4peh4rpdhamtioq6kmcu0ltwpw...`). We must re-point the app at **this** project
(`30dd4f8523d0`, "ChillOut", eu-central-1) and make it work end-to-end: frontend → API → Postgres
queue → WRF worker.

The WRF source is **not** in the studio workspace (`/home/user/project` is an unrelated ML repo). It
was cloned to **`/home/user/ChillOut_test`** from https://github.com/Kir4o-code/ChillOut_test — full
app intact (`site/simulation/sim.js`, `functions/api` + `lib/db.mjs`/`schema.mjs`, `worker/`, `infra/`).

Architecture: Lambda API (`functions/api/index.mjs`) validates scenarios and inserts job rows into
Postgres (`simulations` table, auto-created by `lib/db.mjs:ensureSchema`). A separate **WRF worker
daemon** polls the *same* Postgres and runs the real model. The only link between them is a shared
`DATABASE_URL`.

### Decisions (confirmed with user)
- Provision Postgres (+storage) for this project via `openkbs deploy` — it is currently NOT enabled.
- Deploy from the **clone** `/home/user/ChillOut_test` (non-destructive; leaves `/home/user/project` alone).
- The **WRF worker runs on a different container** (another project the user controls). I cannot run
  commands there, so I prepare everything here and hand the user a copy-paste block that exports
  **this** project's `DATABASE_URL` and launches the worker.

### Preconditions found
- Project `30dd4f8523d0` exists, region eu-central-1; container authed to it (`openkbs list` shows it).
- `openkbs postgres info` → "Postgres is not enabled" → must run `openkbs deploy`.
- Two project-specific values to change: `openkbs.json` projectId, and `sim.js` `API_BASE`.

## Plan — Part A: here (project 30dd4f8523d0), from /home/user/ChillOut_test

1. **Re-point binding** — edit `/home/user/ChillOut_test/openkbs.json`: `projectId` `5ca3e61fce09` →
   `30dd4f8523d0`. Leave region/postgres/storage/functions/site unchanged (already correct:
   `postgres:true`, eu-central-1).
2. **Commit** (deploys auto-tag from the commit msg): `feat: re-point ChillOut WRF app at project 30dd4f8523d0`.
3. **Provision infra** — `openkbs deploy` (creates Postgres + storage declared in openkbs.json).
   Verify with `openkbs postgres info`.
4. **Deploy API** — `openkbs fn deploy api`. Capture the new Lambda function URL from output
   (or `openkbs fn list`). Smoke test: `openkbs fn invoke api -d '{"action":"status"}'` →
   expect `postgres:true`.
5. **Re-point frontend** — edit `site/simulation/sim.js:7` `API_BASE` → the new Lambda URL (keep
   trailing slash, matching the existing format).
6. **Commit** then **deploy site** — `openkbs site deploy`. Note the CloudFront URL.
7. **Capture DATABASE_URL** for the worker — `openkbs postgres connection` (run from the repo root).
   This is the value the worker on the other container must use.

## Plan — Part B: worker container (user runs; I provide the block)
On the other container, from a clone of the WRF repo (repo root, NOT bound to this project):
```
git clone https://github.com/Kir4o-code/ChillOut_test.git && cd ChillOut_test
bash infra/install_wrf.sh            # builds WRF 4.6.1 + WPS 4.6.0 under /opt/wrf (long, ~30-90 min)
bash infra/check_wrf_install.sh      # must PASS: gcc/gfortran/mpi/netcdf + 5 binaries
pip3 install -r worker/requirements.txt
export DATABASE_URL='<value from Part A step 7>'   # THIS project's Postgres, not the container's own
bash worker/run_worker.sh            # daemon: polls simulations table, runs WRF
```
Key point (`worker/run_worker.sh:33-37`): exporting `DATABASE_URL` first bypasses the CLI fallback so
the worker binds to project 30dd4f8523d0's queue even though the container belongs to another project.

## Critical files
- `/home/user/ChillOut_test/openkbs.json` — projectId re-point (step 1)
- `/home/user/ChillOut_test/site/simulation/sim.js:7` — API_BASE re-point (step 5)
- `/home/user/ChillOut_test/functions/api/index.mjs` + `lib/db.mjs`, `lib/schema.mjs` — API (no edits; deploy only)
- `/home/user/ChillOut_test/worker/run_worker.sh`, `worker/config.py`, `infra/install_wrf.sh`, `infra/check_wrf_install.sh` — worker bring-up

## Verification (end-to-end)
- `openkbs postgres info` shows Postgres enabled; `fn invoke api {"action":"status"}` → `postgres:true`.
- Open the live `/simulation/` (new CloudFront), submit the default **Stara Zagora** scenario; the
  request hits the new Lambda (Network tab shows the new URL), returns a job id (201).
- Browser polls `getStatus`; with the worker running, the job advances
  created → preprocessing → real → baseline → candidate → postprocessing → completed, ending with a
  baseline-vs-candidate result + animation. Report wall-clock.
- Failure triage: "Run simulation" does nothing → recheck `API_BASE` (step 5). Job stuck in `created`
  forever → worker not pointed at THIS project's Postgres (Part B `DATABASE_URL`).

## Open item
- User to identify the worker container's project (`openkbs list` there) and confirm it's eu-central-1
  for low DB latency. Not blocking Part A.
