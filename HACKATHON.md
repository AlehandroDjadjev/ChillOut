# ZaraHack 2026 - Project Submission

## 1. Team

- **Team name:** ChillOut
- **Members (name - what each person did):** coming soon
- **How did you split the tasks? Who did what?:** We split the work into data collection, model training, simulation/backend wiring, and demo UI. The data work builds satellite/weather datasets from public sources, the ML work trains temperature and cloud-planning models, the backend handles API/proxy/simulation jobs, and the frontend shows the user-facing before/after result.

## 2. What Problem Are You Solving?

Cities are getting hotter, and local teams do not have an easy way to explore how clouds, radiation, humidity, and wind relate to temperature over a specific place and day. Our project helps climate researchers, city planners, and hackathon reviewers test cloud-cooling scenarios using public satellite and weather data instead of guessing from static charts.

## 3. How Do You Solve It? (in plain language)

A user chooses a place and a date. The app shows the real satellite/weather situation, then creates a changed version where clouds are added or adjusted, and compares what happened to temperature-related features. The goal is to make the question simple: "If this area had different cloud cover, would the heat situation move in the right direction?"

## 4. What Technologies Do You Use?

- **Languages:** JavaScript, Python, HTML, CSS, shell scripts
- **Frontend:** Static HTML/CSS/JavaScript demo pages
- **Backend:** Node.js HTTP server, local API proxy, server-side dataset/image requests
- **ML/Data:** PyTorch, NumPy, Pillow, JSONL manifests, train/validation/test splits
- **Weather/satellite APIs:** Copernicus Data Space Ecosystem / Sentinel Hub, Open-Meteo historical weather API with ERA5-based data
- **Simulation pieces:** WRF-oriented Python worker code, NetCDF processing, cloud injection and postprocessing scripts
- **Deployment/dev:** Local Node server, `.env` configuration, OpenKBS-related worker/API scaffolding

## 5. How Do You Wire Them Together?

Public satellite and weather APIs feed the local Node server. The server proxies Copernicus requests safely, renders imagery, fetches ERA5-based weather values, and supports the dataset builder. The Python dataset builder exports city/day samples with Sentinel-2 cloud masks, Sentinel-5P atmospheric values, Sentinel-3 cloud/atmospheric proxies, and Open-Meteo weather features. PyTorch models use those samples to predict temperature and to propose cloud-mask edits. The frontend then shows the original scene, the proposed cloud change, and a compact before/after comparison of temperature-related features.

```text
Copernicus + Open-Meteo
  -> Node proxy / dataset exporter
  -> JSONL + masks + PyTorch tensors
  -> temperature model + RL cloud planner
  -> demo UI / simulation result view
```

## 6. Do You Train an ML Model?

Yes. We train a temperature model that predicts target temperature from cloud imagery and weather features, and we built an RL planner that proposes cloud-mask edits such as create, remove, or modify cloud areas. The models are trained from scratch in PyTorch using exported samples from Sentinel and ERA5-based weather data. The dataset is split into train/validation/test JSONL files, and diagnostics compare normal predictions against baselines such as removing cloud inputs or shuffling cloud data. The RL planner can use the trained temperature model as a reward function, so proposed cloud edits are scored by whether the predicted temperature moves closer to the target.

## 7. What Datasets Do You Use, and How?

**Dataset 1: Sentinel-2 L2A via Copernicus Data Space / Sentinel Hub**
- **Source + link:** https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html
- **Why this data:** Sentinel-2 provides visible satellite imagery and cloud masks, which are the spatial input for cloud cover.
- **What we did to it:** We request scene imagery through the local proxy, export cloud masks, resize them into model-ready image tensors, and drop blank or transparent outputs.

**Dataset 2: Sentinel-5P L2 via Copernicus Data Space / Sentinel Hub**
- **Source + link:** https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S5PL2.html
- **Why this data:** It gives cloud and atmospheric measurements such as cloud fraction, cloud optical thickness, cloud top height, cloud top pressure, aerosol index, and trace gases.
- **What we did to it:** We fetch daily statistical summaries and merge them into each city/day sample as numeric model features.

**Dataset 3: Sentinel-3 OLCI and SLSTR via Copernicus Data Space / Sentinel Hub**
- **Source + link:** https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3OLCI.html and https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3SLSTR.html
- **Why this data:** These products add atmospheric and cloud-proxy fields such as humidity, water vapour, cloud screening, cloud flagging, cirrus detection, and cloud clearing.
- **What we did to it:** We use them as supporting daily statistics and cloud-type proxies, not as perfect ground-truth cloud labels.

**Dataset 4: Open-Meteo historical weather API**
- **Source + link:** https://open-meteo.com/en/docs/historical-weather-api
- **Why this data:** It provides ERA5-based temperature, wind, precipitation, cloud cover, pressure, humidity, and radiation values.
- **What we did to it:** We aggregate daily and hourly weather fields, join them with the satellite samples, and use temperature as the model target.

## 8. How Will the Platform Scale?

The current version works as a local prototype and can scale moderately by caching API responses, storing generated datasets, and serving precomputed model outputs. The first bottlenecks would be live Copernicus API calls, model inference cost, and WRF simulation runtime. To support many users, we would move long-running jobs to a queue, cache city/date products, store model artifacts separately, and run simulations on dedicated worker machines.

## 9. What Challenges Did You Face?

The hardest part was aligning data from different sources: Sentinel products, ERA5-based weather values, and city/date windows do not all arrive at the same resolution or reliability. We handled this by building a manifest-based exporter, filtering bad image outputs, and keeping source notes with each sample. The second challenge was keeping the ML honest: cloud intervention is not directly observed in the real world, so the RL model is framed as a planner prototype scored by a learned simulator, not as proof of real atmospheric control.

## 10. Did You Check What Already Exists?

Yes. We looked at Sentinel Hub/Copernicus viewers, Open-Meteo weather access, and weather simulation tools such as WRF. Existing tools can show satellite layers or run weather models, but they usually do not connect public cloud imagery, weather features, a temperature model, and a cloud-edit planning workflow in one simple demo. Our difference is the combined workflow: choose a place/date, inspect real data, generate a cloud-change proposal, and compare the result.

## 11. Where Did You Use AI, and What's Not Yours?

- **AI tools used (and for what):** ChatGPT/Codex was used for coding help, debugging, documentation drafting, and building small demo/test UI pieces.
- **Third-party code / templates / tutorials you reused:** The project uses public API documentation and standard libraries rather than copying a full external app template.
- **Their licences:** Node.js and Python ecosystem packages keep their own licences. Copernicus/Sentinel and Open-Meteo data are accessed through their public services and terms. WRF-related code depends on the WRF/NetCDF ecosystem and should follow those upstream licences and data-use rules.

## 12. Honesty Box

- The ML/RL system is a prototype. It demonstrates the pipeline and model interfaces, but it does not prove that real cloud changes will cool a city.
- Some demo comparisons are simplified so the workflow is understandable in a hackathon setting.
- The RL planner needs a trained policy checkpoint and a validated reward model before its outputs should be treated as meaningful.
- The WRF worker path exists in code, but running full weather simulations requires the correct WRF environment, input data, and worker infrastructure.
- API keys must stay in `.env`; no secrets should be committed.
- Team member names and exact role split still need final confirmation before submission if the team wants personal attribution.
