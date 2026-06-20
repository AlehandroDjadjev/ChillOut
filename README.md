# Stara Zagora Field Explorer

A tiny static HTML/JavaScript demo that keeps Sentinel-2 imagery front and center, while also letting you switch to Sentinel-5P atmospheric fields and ERA5 weather fields for the area around Stara Zagora, Bulgaria.

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
- `COPERNICUS_METEO_URL`: defaults to `https://archive-api.open-meteo.com/v1/archive` for the ERA5 weather source.
- `COPERNICUS_BBOX`: defaults to `25.43,42.31,25.86,42.57`, a bounding box around Stara Zagora.
- `COPERNICUS_IMAGE_WIDTH`: defaults to `768`.
- `COPERNICUS_IMAGE_HEIGHT`: defaults to `480`.

## Security Note

Keep `.env` private. The page no longer sends your client secret to the browser.

## API Notes

- Sentinel-2 scenes come from the Sentinel Hub Catalog and Process APIs, and support natural color imagery, cloud mask, and cloud probability.
- Sentinel-5P scenes come from the Sentinel Hub Catalog and Process APIs, using cloud properties and atmospheric columns like cloud fraction, cloud optical thickness, cloud top height, NO2, O3, CO, CH4, SO2, and aerosol index.
- ERA5 scenes are rendered server-side from Open-Meteo archive data, which is ERA5-based reanalysis. That is where wind, radiation, humidity, and cloud-layer fields come from.
- No single satellite in this app covers the full set of requested variables, so the field picker switches between Sentinel-2, Sentinel-5P, and ERA5 depending on what you want to inspect.

Official references:

- [Sentinel Hub authentication](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html)
- [Sentinel-2 L2A data docs](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html)
- [Sentinel-5P L2 data docs](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S5PL2.html)
- [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api)
