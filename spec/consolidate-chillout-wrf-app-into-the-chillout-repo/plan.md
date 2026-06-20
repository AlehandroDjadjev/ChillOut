# Consolidate ChillOut WRF app into the ChillOut repo

## Context
Part A is already DONE: the WRF app (in `/home/user/ChillOut_test`, GitHub `Kir4o-code/ChillOut_test`)
is re-pointed at OpenKBS project `30dd4f8523d0` and deployed — API
`https://7mfql7bgxtukoux34v3g26nbie0axyyc.lambda-url.eu-central-1.on.aws/` (`status` → `postgres:true`),
site live at `https://d31rm64rx9q92u.cloudfront.net/simulation/`, Postgres provisioned, one test job
queued. The worker (Part B) still needs to run on container `5ca3e61fce09`.

**New goal (user):** make `AlehandroDjadjev/ChillOut` the canonical repo for the whole thing — port the
WRF app (API, **frontend**, worker, infra) *into* `ChillOut` (which currently holds only the ML project),
deploy from `ChillOut`, and have the worker container clone `ChillOut`. Push access to `ChillOut` now
works (`gh api repos/AlehandroDjadjev/ChillOut` → `permissions.push:true`).

### Key facts (verified)
- `/home/user/project` = local clone of `ChillOut` = the ML project (`NewModel/`, `train_cloud_temp*.py`,
  `cloud_rl/`, `server.mjs`, root `index.html`, `package.json`, `requirements.txt`). Its local
  `functions/`/`site/`/`openkbs.json` are **unpushed scaffold placeholders** — remote `main` is pure ML.
- Real WRF app is only in `/home/user/ChillOut_test`: `functions/api` (+`lib/db.mjs`,`schema.mjs`),
  `site/` (incl. `simulation/sim.js` already pointed at the deployed Lambda), `worker/`, `infra/`,
  `openkbs.json` (projectId `30dd4f8523d0`, `postgres:true`, storage media). Total ~1.3 MB.
- OpenKBS deploys only `functions/` + `site/` + infra services per `openkbs.json`. The ML files
  (`server.mjs`, root `index.html`, training scripts) are NOT touched by OpenKBS — they coexist in the
  repo harmlessly. Full ML↔WRF unification is explicitly deferred ("not now").
- Same project `30dd4f8523d0` → redeploy reuses the SAME Lambda URL + CloudFront distribution, so
  `sim.js` `API_BASE` should stay valid (verify; repoint only if it changed).

## Plan — Part A′: port WRF app into ChillOut, redeploy from there

Work in a **fresh clone** (non-destructive; avoids `/home/user/project`'s divergent scaffold commits).

1. **Fresh clone** — `git clone https://github.com/AlehandroDjadjev/ChillOut.git /home/user/ChillOut`
   (gets canonical ML `main`). cd into it.
2. **Copy WRF app in** — `rsync -a --exclude .git` from `/home/user/ChillOut_test/` the dirs/files:
   `functions/`, `infra/`, `worker/`, `site/`, `openkbs.json`, and merge `spec/`. This overwrites
   nothing ML (different paths) except adds `openkbs.json`.
3. **Reconcile `.gitignore`** — union ChillOut's (`node_modules/`, `.*`, `config.js`, `.env`,
   `dataset.out/*`) with ChillOut_test's Python ignores (`__pycache__/`, `*.pyc`). Also ignore
   `worker`'s `runs/`/`/opt/wrf` artifacts if present. Keep large ML datasets ignored.
4. **Sanity check** `git status` — confirm only intended WRF files staged, no datasets / `__pycache__` /
   `node_modules`.
5. **Commit + push** to `origin main`:
   `feat: consolidate ChillOut WRF app (api, frontend, worker, infra) into this repo`.
6. **Bind + deploy from ChillOut** (openkbs.json projectId `30dd4f8523d0`, already provisioned):
   - `openkbs postgres info` → confirm still enabled.
   - `openkbs fn deploy api` → verify Lambda URL == `7mfql7bgxtukoux34v3g26nbie0axyyc...`;
     smoke test `{"action":"status"}` → `postgres:true`. If URL changed, edit
     `site/simulation/sim.js` `API_BASE` and recommit.
   - `openkbs site deploy` → confirm CloudFront serves the WRF `/simulation/`.
7. **Capture DATABASE_URL** — `openkbs postgres connection` (from `/home/user/ChillOut` root) for Part B.

## Plan — Part B: worker container `5ca3e61fce09` (user runs; I provide the block)
Now clones **ChillOut** instead of ChillOut_test:
```
git clone https://github.com/AlehandroDjadjev/ChillOut.git && cd ChillOut
bash infra/install_wrf.sh            # builds WRF 4.6.1 + WPS 4.6.0 under /opt/wrf (~30-90 min)
bash infra/check_wrf_install.sh      # must PASS: gcc/gfortran/mpi/netcdf + 5 binaries
pip3 install -r worker/requirements.txt
export DATABASE_URL='<value from Part A′ step 7>'   # project 30dd4f8523d0's Postgres
bash worker/run_worker.sh            # daemon: polls simulations table, runs WRF
```
`worker/run_worker.sh:33-37`: exporting `DATABASE_URL` first bypasses the CLI fallback, so the worker
binds to `30dd4f8523d0`'s queue even though the container belongs to project `5ca3e61fce09`.

## Critical files
- `/home/user/ChillOut/openkbs.json` — projectId `30dd4f8523d0`, `postgres:true` (copied from ChillOut_test)
- `/home/user/ChillOut/site/simulation/sim.js` — `API_BASE` (verify == deployed Lambda; repoint if changed)
- `/home/user/ChillOut/functions/api/index.mjs` + `lib/db.mjs`,`lib/schema.mjs` — API (deploy only)
- `/home/user/ChillOut/worker/run_worker.sh`,`worker/config.py`,`infra/install_wrf.sh`,`infra/check_wrf_install.sh`
- `/home/user/ChillOut/.gitignore` — merged

## Verification (end-to-end)
- `git ls-remote origin main` shows the new commit; remote tree now has `functions/`,`site/`,`worker/`,`infra/`,`openkbs.json`.
- `openkbs fn deploy api` → `status` returns `postgres:true`; `openkbs site deploy` succeeds.
- Open live `/simulation/`, submit default **Stara Zagora** scenario → 201 + job id; Network tab hits the
  correct Lambda URL.
- With Part B worker running, job advances created → preprocessing → real → baseline → candidate →
  postprocessing → completed (baseline-vs-candidate result + animation). Report wall-clock.
- Triage: "Run simulation" does nothing → recheck `API_BASE`. Job stuck in `created` → worker not pointed
  at `30dd4f8523d0`'s Postgres (Part B `DATABASE_URL`).

## Notes
- ML↔WRF deployment unification is NOT in scope now — the ML code rides along in the repo but OpenKBS
  only deploys `functions/`+`site/`. Doing the full merge cleanly is a later step.
- `/home/user/project` and `/home/user/ChillOut_test` are left untouched.
