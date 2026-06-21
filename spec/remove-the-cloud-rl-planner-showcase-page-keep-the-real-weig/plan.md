# Remove the Cloud RL "Planner Showcase" page; keep the real-weights RL Inference Runner

## Context
The site has three model-demo pages. Two of them are about the **same** cloud-RL policy:

- `site/cloud-rl/` — "Planner Showcase": shows the input→output contract but **never runs real
  weights** (uses a hosted endpoint if configured, otherwise a labelled preview fallback).
- `site/rl-inference/` — "RL Inference Runner": the real one — you upload a trained `.pt`
  checkpoint + `stats.json`, it's stored, and a backend VM worker actually runs the policy.

The user wants only the page that runs real weights. The Planner Showcase is redundant and
"doesn't work" (no real inference), so it should be deleted. The **Temperature Predictor**
(`site/model-temperature/`) is a separate model and **stays** (confirmed with the user).

Outcome: `site/cloud-rl/` is gone, every link to it is redirected or removed, and the homepage
surfaces the RL Inference Runner instead.

## Changes

### 1. Delete the page directory
- Remove `site/cloud-rl/` entirely (`index.html` + `cloud-rl.js`). Both are self-contained;
  all CSS/JS they pull (`css/styles.css`, `css/animations.css`, `model-lab/model-lab.css`,
  `js/reveal.js`, `model-lab/model-contracts.js`) is shared and stays — `model-contracts.js`
  is still used by both remaining model pages, so it is NOT touched.

### 2. Homepage card — `site/index.html` (lines 158-162)
Replace the "Cloud RL Planner" card with an **RL Inference Runner** card linking to
`rl-inference/index.html`:
```html
<a class="model-link glass reveal" data-reveal data-delay="1" href="rl-inference/index.html">
  <span class="lab-pill">model 02</span>
  <h3>RL Inference Runner</h3>
  <p>Upload trained RL weights and a stats file, fetch a live Sentinel-2 cloud mask, and run the
     real policy on a backend worker to get the generated mask and action data.</p>
</a>
```
(Keeps the existing "model 01" Temperature Predictor card above it unchanged.)

### 3. Fix cross-page nav CTAs (each page's header links to a sibling demo)
- `site/rl-inference/index.html:30` — currently `../cloud-rl/index.html` "Planner Showcase".
  Repoint to the temperature page: `../model-temperature/index.html`, label **"Temperature Model"**.
- `site/model-temperature/index.html:30` — currently `../cloud-rl/index.html` "Cloud RL Demo".
  Repoint to the inference page: `../rl-inference/index.html`, label **"RL Inference Runner"**.

### 4. Simulation page nav — `site/simulation/index.html:37`
- Replace the `<a href="../cloud-rl/index.html">Cloud RL</a>` link in the "Model demos" nav with
  `<a href="../rl-inference/index.html">RL Inference</a>` (keep the Temperature model link above it).

## Critical files
- `site/cloud-rl/` — **delete** (index.html, cloud-rl.js).
- `site/index.html` — swap homepage model card (lines 158-162).
- `site/rl-inference/index.html` — nav CTA line 30.
- `site/model-temperature/index.html` — nav CTA line 30.
- `site/simulation/index.html` — nav link line 37.
- `site/model-lab/model-contracts.js` — **keep, untouched** (shared by remaining pages).

## Verification
1. Grep the whole `site/` tree for `cloud-rl` and `Planner Showcase` → expect **zero** matches.
2. Commit (`refactor:`-style), then `openkbs site deploy`.
3. agent-browser smoke test on the live CloudFront URL:
   - Homepage shows two model cards (Temperature Predictor + RL Inference Runner); the RL card
     navigates to `/rl-inference/`.
   - `/cloud-rl/index.html` no longer resolves (page gone).
   - From `/rl-inference/` the header CTA goes to `/model-temperature/`, and from
     `/model-temperature/` the header CTA goes to `/rl-inference/` (no dead links).
   - `/simulation/` "Model demos" nav has Temperature + RL Inference, no Cloud RL.

## Out of scope
- No changes to the RL inference upload/worker pipeline or the shared model-contracts logic.
- Temperature Predictor page behavior is unchanged (still preview-until-endpoint).
