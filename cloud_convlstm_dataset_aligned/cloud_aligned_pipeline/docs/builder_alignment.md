# Dataset Builder ↔ ConvLSTM Alignment

This package aligns two systems:

1. `tools/cloud_state_dataset_downloader.py`
   - Finds Sentinel-2 L2A scenes.
   - Downloads Sentinel-2 cloud-only tensors.
   - Downloads ERA5 single-level fields.
   - Optionally downloads ERA5 pressure-level fields.
   - Builds per-state `.npz` files in `states_npz/`.

2. ConvLSTM training code in `src/`
   - Expects normalized tensors and window indices:
     - `x_series.npy`: `[N_state, C_in, H, W]`
     - `y_series.npy`: `[N_state, 4, H, W]`
     - `train_windows.csv`, `val_windows.csv`, `test_windows.csv`

The bridge is:

```bash
python src/build_training_tensors_from_downloader.py   --dataset-dir ./cloud_state_dataset   --out-dir data/processed_aligned   --profile builder24   --height 64   --width 64   --input-len 8   --horizon 1
```

Then train with:

```bash
python src/train.py --config configs/cloud_forward_builder24_5day.yaml
```

## Why an indexed dataset?

The downloader collects multiple locations. If we just concatenate all states and use
normal contiguous slicing, one training window could accidentally cross from Sofia
into Athens. That would corrupt training.

So the converter writes location-safe window CSVs:

```text
train_windows.csv
val_windows.csv
test_windows.csv
```

Each row explicitly tells the training code:

```text
x_start, x_end, y_start, y_end, location
```

This guarantees that a sequence window stays inside the same location.

## Input profiles

### `builder24`

Recommended if you can afford slightly more than 20 channels. It uses:

- Sentinel-2 cloud mask, brightness, cloud probability, cirrus flag
- ERA5 total/low/mid/high cloud cover
- ERA5 cloud liquid/ice water
- Sentinel-2 AOT and WVP proxies
- ERA5 shortwave/longwave all-sky and clear-sky radiation
- ERA5 net solar/thermal radiation
- ERA5 temperature anomaly
- ERA5 dewpoint, total column water vapour, pressure, precipitation
- ERA5 10m wind speed

This is still small but captures more of the downloader's useful fields.

### `core20`

Safer/lighter. It drops some extra S2 fields and keeps closer to the earlier
20-channel model.

## Output targets

For each state, the converter creates current-state target fields:

```text
shortwave_anomaly = surface_solar_radiation_downwards
                  - surface_solar_radiation_downwards_clear_sky

longwave_anomaly = surface_thermal_radiation_downwards
                 - surface_thermal_radiation_downwards_clear_sky

net_radiation_anomaly =
    (surface_net_solar_radiation - surface_net_solar_radiation_clear_sky)
  + (surface_net_thermal_radiation - surface_net_thermal_radiation_clear_sky)

temperature_anomaly =
  2m_temperature - same-location seasonal-bin baseline
```

The training Dataset then uses future windows, e.g.:

```text
past 8 states -> next 1 state
~40 days history -> next 5-day response
```

## Image vs numeric fields

Sentinel-2 cloud tensor:
- arrives as image-like `[H, W, 12]`
- converted into selected map channels and resized to the training grid.

ERA5:
- arrives as numeric grids, often lower spatial resolution
- resized to the same training grid using simple interpolation.

Everything becomes:

```text
[channel, height, width]
```

before entering the model.
