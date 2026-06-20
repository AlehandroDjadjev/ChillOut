# Cloud/World Interaction Temperature Model

This package applies the same cleanup discovered in the ConvLSTM/cloudforce experiments to the older CNN + tabular model.

## Main changes

- Keeps the target as a temperature number.
- Removes temperature inputs.
- Removes Open-Meteo cloud-cover scalar shortcuts.
- Removes duplicate radiation fields and keeps one solar context field.
- Makes Sentinel-2 clouds the main cloud source:
  - cloud-only mask image
  - non-cloud pixels black
  - cloud pixels white/probability-weighted
  - cloud scalar statistics derived from the Sentinel-2 cloud tensor
- Splits the model into:
  - cloud image/scalar branch
  - world-state branch
  - cloud-to-world bridge
  - interaction head
  - exposed cloud/world/interaction component predictions
- Adds optional GRU sequence heads over recent states.

## Files

- `build_temperature_dataset_cloudforce.py`
- `train_cloud_temp_interaction.py`
- `diagnose_cloud_temp_interaction.py`

## Build dataset

```cmd
python build_temperature_dataset_cloudforce.py ^
  --start-date 2023-01-01 ^
  --end-date 2026-01-01 ^
  --out dataset_cloudforce_interaction ^
  --max-scenes-per-location 200 ^
  --resolution-m 250 ^
  --patch-km 10 ^
  --s2-workers 8 ^
  --openmeteo-workers 6 ^
  --target-offset-days 5
```

## Train

```cmd
python train_cloud_temp_interaction.py ^
  --data-root dataset_cloudforce_interaction ^
  --out-dir runs/cloud_temp_interaction ^
  --image-height 160 ^
  --image-width 160 ^
  --lookback 4 ^
  --max-gap-days 12 ^
  --epochs 120 ^
  --batch-size 24 ^
  --augment ^
  --world-drop-prob 0.20 ^
  --cloud-aux-weight 0.35
```

Use `--lookback 1` if your data is too sparse. Use `--num-workers 0` on Windows.

## Diagnose

```cmd
python diagnose_cloud_temp_interaction.py ^
  --data-root dataset_cloudforce_interaction ^
  --checkpoint runs/cloud_temp_interaction/best.pt ^
  --split val ^
  --out-dir diagnostics/interaction_val ^
  --feature-importance
```

The most important checks:

- `normal` should beat `no_cloud_all`
- `normal` should beat `cloud_shuffle`
- `normal` should beat `no_world` less dramatically than the old model did
- `component_metrics.csv` should show the cloud component has non-useless correlation
- `input_feature_importance.csv` should rank Sentinel-derived cloud fields high

## Recommended dataset size

Quick smoke test:
- 20 scenes/location
- `lookback 1` or `lookback 2`

First useful run:
- 150-250 scenes/location
- 10 locations
- `lookback 4`
- max gap 12 days

Better:
- 20-40 locations
- 200+ scenes/location
- `lookback 4`
- train 120-200 epochs

If the diagnostic says `no_cloud_all` is close to `normal`, add more cloudy scenes/locations and reduce world dropout less aggressively.
