# Stara Zagora Field Explorer

A tiny static HTML/JavaScript demo that keeps Sentinel-2 imagery front and center, while also letting you switch to Sentinel-5P cloud and chemistry fields, Sentinel-3 OLCI atmospheric fields, Sentinel-3 SLSTR cloud-proxy bands, and ERA5 weather fields for the area around Stara Zagora, Bulgaria.

The page finds the latest 7 samples for the selected source and field, then renders them as a simple visual strip.

## Setup

1. Create a Copernicus Data Space Ecosystem account.
2. Open the Sentinel Hub dashboard and create an OAuth client.
3. Copy `.env.example` to `.env`.
4. Fill in `COPERNICUS_CLIENT_ID` and `COPERNICUS_CLIENT_SECRET`.
5. Start the local proxy/server:

```powershell
node server.mjs
```

Then open `http://localhost:5173`.

If you already created `config.js` from the earlier browser-only version, move those credential values into `.env`. The server will read `config.js` as a temporary local fallback, but it will not serve that file to the browser.

## Environment Variables

The browser cannot call the Copernicus OAuth token endpoint directly because of CORS, and the client secret should not live in browser JavaScript. `server.mjs` reads `.env`, requests the token server-side, caches it, and proxies Sentinel Hub Process API requests through `/api/process`.

Required values:

- `COPERNICUS_CLIENT_ID`: OAuth client ID from the Sentinel Hub dashboard.
- `COPERNICUS_CLIENT_SECRET`: OAuth client secret from the Sentinel Hub dashboard.

Optional values:

- `COPERNICUS_TOKEN_URL`: defaults to `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`.
- `COPERNICUS_PROCESS_URL`: defaults to `https://sh.dataspace.copernicus.eu/process/v1`.
- `COPERNICUS_CATALOG_URL`: defaults to `https://sh.dataspace.copernicus.eu/catalog/v1/search`.
- `COPERNICUS_STATS_URL`: defaults to `https://sh.dataspace.copernicus.eu/statistics/v1` and powers the bar-chart view for numeric Copernicus bands.
- `COPERNICUS_METEO_URL`: defaults to `https://archive-api.open-meteo.com/v1/archive` for the ERA5 weather source.
- `COPERNICUS_BBOX`: defaults to `25.43,42.31,25.86,42.57`, a bounding box around Stara Zagora.
- `COPERNICUS_IMAGE_WIDTH`: defaults to `768`.
- `COPERNICUS_IMAGE_HEIGHT`: defaults to `480`.

Backend RL runner fallback:

- `OPENAI_API_KEY`: set this on the deployed backend/API function, not in browser JavaScript. It is used only when the RL inference runner is submitted without an uploaded `.pt` checkpoint.
- `OPENAI_IMAGE_MODEL`: optional, defaults to `gpt-image-1`.
- `OPENAI_IMAGE_SIZE`: optional, defaults to `1024x1024`.

## Security Note

Keep `.env` private. The page no longer sends your client secret to the browser.

## API Notes

- Sentinel-2 scenes come from the Sentinel Hub Catalog and Process APIs, and support natural color imagery, cloud mask, and cloud probability.
- Sentinel-5P scenes come from the Sentinel Hub Catalog and Process APIs, using the exact cloud-property products in Sentinel Hub, plus atmospheric columns like cloud fraction, cloud optical thickness, cloud top height, cloud top pressure, NO2, O3, CO, CH4, SO2, and aerosol index.
- Sentinel-3 OLCI scenes come from the Sentinel Hub Catalog and Process APIs, and give a much finer atmospheric view than Sentinel-5P for humidity, sea level pressure, water vapour, and a cloud-pressure proxy band. It does not replace the exact Sentinel-5P cloud-property products.
- Sentinel-3 SLSTR scenes come from the Sentinel Hub Catalog and Process APIs, and expose cloud-screening, cloud-flagging, cirrus-detection, and cloud-clearing bands. That is the closest Copernicus source in this app for a cloud-type proxy, but it is still a proxy rather than a clean categorical cloud-type classifier.
- ERA5 scenes are rendered server-side from Open-Meteo archive data, which is ERA5-based reanalysis. That is where wind, radiation, humidity, and cloud-layer fields come from.
- The radiation picker now includes shortwave, direct, diffuse, direct normal irradiance, global tilted radiation, and terrestrial radiation, all charted as numeric values instead of a single blended field.
- The chart view uses the Sentinel Hub Statistical API, which returns actual numeric summaries for supported bands instead of rendered imagery.
- No single satellite in this app covers the full set of requested variables, so the field picker switches between Sentinel-2, Sentinel-5P, Sentinel-3 OLCI, Sentinel-3 SLSTR, and ERA5 depending on what you want to inspect.

Official references:

- [Sentinel Hub authentication](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html)
- [Sentinel-2 L2A data docs](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html)
- [Sentinel-5P L2 data docs](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S5PL2.html)
- [Sentinel-3 OLCI L1B data docs](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3OLCI.html)
- [Sentinel-3 SLSTR L1B data docs](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3SLSTR.html)
- [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api)

## Training Dataset Export

If you want a PyTorch-friendly dataset bundle, start the local proxy first:

```powershell
npm start
```

Then run the exporter:

```powershell
python build_temperature_dataset.py --out dataset_out --years 3
```

If you omit `--years`, the exporter falls back to the shorter `--days` window. With `--years 3`, it builds a three-year, per-city manifest for six Balkan cities. It downloads:

- Sentinel-2 cloud masks
- Sentinel-5P cloud and atmospheric statistics
- Sentinel-3 OLCI atmospheric statistics
- Open-Meteo daily weather values for temperature, wind, precipitation, cloud cover, and radiation

Outputs land in `dataset_out/`:

- `dataset.jsonl`
- `splits/train.jsonl`, `splits/val.jsonl`, `splits/test.jsonl`
- `masks/`
- `metadata.json`
- `dataset.pt` if `torch` is installed in the Python environment

`shortwave_radiation_sum` is still used as a net-radiation proxy in the training export, but the viewer now exposes the fuller hourly radiation set separately in the ERA5 source.

Blank or fully transparent satellite renders are dropped automatically, and the whole city-day sample is removed from the manifest so you do not train on empty masks.
