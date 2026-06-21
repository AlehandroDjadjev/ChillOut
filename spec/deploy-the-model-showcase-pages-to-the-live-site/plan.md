# Deploy the model showcase pages to the live site

## Context
The user pulled the latest `main` and found the model pages exist in `site/` but are
not visible on the deployed website. Investigation confirms the pages are **already
fully built and integrated in code** — they're just **not deployed yet**.

What's on disk and committed at HEAD (`9ab7baf`, via commit `15cf11d "models showcase pages"`):
- `site/model-temperature/` (`index.html`, `temp-model.js`) — Temperature Predictor page
- `site/cloud-rl/` (`index.html`, `cloud-rl.js`) — Cloud RL Planner page
- `site/model-lab/` (`model-lab.css`, `model-contracts.js`) — shared styles + model I/O contracts

Integration is already correct and consistent:
- `site/index.html` links to both pages (`#models` section, lines 153 & 158) and loads
  `model-lab/model-lab.css?v=12` (line 22).
- Both subpages load `../model-lab/model-lab.css?v=12` and `../model-lab/model-contracts.js?v=12`.
- Working tree is **clean**; nothing uncommitted.

Root cause: the OpenKBS static site was last deployed at tag `v2`, **before** these pages
were committed. CloudFront is therefore serving the old build. No code fix is required —
only a redeploy.

## Plan
Single step — deploy the already-committed `site/` to S3 + CloudFront:

```
openkbs site deploy
```

`openkbs.json` has `"site": "./site"`, so this publishes the whole site folder including
the new model pages. (Per project convention deploy auto-tags the next version, `v3`.)

## Verification
1. Run `openkbs site deploy`; confirm it completes and reports the new version/CloudFront URL.
2. Hard-refresh the live homepage → the `#models` section's two cards ("Temperature Predictor",
   "Cloud RL Planner") link to working pages.
3. Open `/model-temperature/` and `/cloud-rl/` on the live URL; confirm they load with
   `model-lab.css` styling applied and no 404s for `model-contracts.js` (check browser console/network).

## Out of scope
- No code changes (pages already integrated and committed).
- No backend/Lambda/function changes.
- The separate `cloud-rl before/after demo` (`spec/…`) remains future work, untouched here.
