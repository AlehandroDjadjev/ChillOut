# Stara Zagora Sentinel-2 Cloud Viewer

A tiny static HTML/JavaScript demo that requests Sentinel-2 L2A imagery from the Copernicus Data Space Ecosystem Sentinel Hub Process API for the area around Stara Zagora, Bulgaria.

The page finds the latest 7 Sentinel-2 L2A images available for the Stara Zagora bounding box, then renders them with:

- true color imagery with cloud overlay
- binary cloud mask
- cloud probability view

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

If you already created `config.js` from the earlier browser-only version, move those credential values into `.env`. The new server will read `config.js` as a temporary local fallback, but it will not serve that file to the browser.

## Environment Variables

The browser cannot call the Copernicus OAuth token endpoint directly because of CORS, and the client secret should not live in browser JavaScript. `server.mjs` reads `.env`, requests the token server-side, caches it, and proxies Sentinel Hub Process API requests through `/api/process`.

Required values:

- `COPERNICUS_CLIENT_ID`: OAuth client ID from the Sentinel Hub dashboard.
- `COPERNICUS_CLIENT_SECRET`: OAuth client secret from the Sentinel Hub dashboard.

Optional values:

- `COPERNICUS_TOKEN_URL`: defaults to `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`.
- `COPERNICUS_PROCESS_URL`: defaults to `https://sh.dataspace.copernicus.eu/process/v1`.
- `COPERNICUS_CATALOG_URL`: defaults to `https://sh.dataspace.copernicus.eu/catalog/v1/search`.
- `COPERNICUS_BBOX`: defaults to `25.43,42.31,25.86,42.57`, a bounding box around Stara Zagora.
- `COPERNICUS_IMAGE_WIDTH`: defaults to `768`.
- `COPERNICUS_IMAGE_HEIGHT`: defaults to `480`.

## Security Note

Keep `.env` private. The page no longer sends your client secret to the browser.

## API Notes

- The app uses OAuth2 client credentials to request an access token.
- The app uses the Sentinel Hub Catalog API to find recent `sentinel-2-l2a` scenes for the configured bounding box.
- The imagery request uses the Sentinel Hub Process API with `input.data.type` set to `sentinel-2-l2a`.
- Cloud rendering uses Sentinel-2 L2A cloud-related bands/classes: `CLM`, `CLP`, and `SCL`.
- The default `mosaickingOrder` is `leastCC`, and `maxCloudCoverage` is editable in the page.

Official references:

- [Sentinel Hub authentication](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html)
- [Sentinel Hub beginner guide](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/UserGuides/BeginnersGuide.html)
- [Sentinel-2 L2A data docs](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html)
- [Sentinel Hub cloud masks](https://docs.sentinel-hub.com/api/latest/user-guides/cloud-masks/)
