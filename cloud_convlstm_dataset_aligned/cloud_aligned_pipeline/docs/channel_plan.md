# Channel Plan

The aim is about 20 input channels: enough to force useful cloud/radiation structure, but small enough for paid GPU training.

## Input channels

### Cloud / cloud-physics channels

0. `cloud_mask_or_fraction`
   - Cloud-only mask/fraction. If made from Sentinel-2 RGB, ground is zeroed.
   - Range should be normalized to roughly 0..1.

1. `low_cloud_cover`
2. `mid_cloud_cover`
3. `high_cloud_cover`
   - Separates low bright cooling clouds from high thin clouds.

4. `cloud_liquid_water`
5. `cloud_ice_water`
   - Helps differentiate liquid/rain-support clouds vs ice clouds.

6. `cloud_optical_or_brightness_proxy`
   - Optical thickness if available; otherwise filtered brightness proxy.

7. `cloud_top_height_or_pressure_proxy`
   - Cloud vertical structure proxy.

### Radiation bridge channels

8. `surface_shortwave_down`
9. `surface_shortwave_clear_sky`
10. `surface_longwave_down`
11. `surface_longwave_clear_sky`
12. `surface_net_shortwave`
13. `surface_net_longwave`

These are the bridge between cloud state and thermal effect.

### Dynamic atmosphere / rain-support channels

14. `temperature_anomaly`
15. `dewpoint_or_relative_humidity`
16. `total_column_water_vapour`
17. `surface_pressure`
18. `precipitation`
19. `wind_u_or_speed`

Optional later additions:
- wind_v
- vertical_velocity_700hPa
- boundary_layer_height
- aerosol_index_or_aod
- sea_salt/sulfate/dust AOD

## Output channels

0. `shortwave_anomaly_next_24h`
1. `longwave_anomaly_next_24h`
2. `net_radiation_anomaly_next_24h`
3. `temperature_anomaly_next_24h`

Loss weights:
- shortwave: 1.0
- longwave: 1.0
- net radiation: 1.0
- temperature: 0.7

## Data source grounding

Suggested sources:
- ERA5 single levels: aligned hourly/3-hourly cloud/radiation/atmosphere backbone.
- Satellite cloud properties: cloud fraction/height/liquid/ice/optical properties.
- Sentinel-5P: cloud fraction, cloud top pressure/height, optical thickness/albedo, aerosol index.
- Sentinel-2: optional filtered cloud-only visual mask/brightness channel, not raw RGB ground.

## Cloud filter template

For raw RGB cloud imagery:
- convert RGB to HSV/YUV-like brightness and saturation
- keep high-brightness, low-saturation pixels as cloud candidates
- set non-cloud/ground pixels to zero
- output at least:
  - cloud mask
  - cloud brightness proxy

The exact threshold should be tuned on the region/sensor.
