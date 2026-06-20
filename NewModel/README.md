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

You can also let the trainer run the extractor first:

```cmd
python NewModel\train_cloud_temp_interaction.py ^
  --build-dataset ^
  --start-date 2023-01-01 ^
  --end-date 2026-01-01 ^
  --dataset-out dataset_cloudforce_interaction ^
  --out-dir runs/cloud_temp_interaction ^
  --max-scenes-per-location 200 ^
  --resolution-m 250 ^
  --patch-km 10 ^
  --s2-workers 8 ^
  --openmeteo-workers 1 ^
  --target-offset-days 5
```

## Train

Fresh run, ignoring any existing `last.pt`:

```cmd
python NewModel\train_cloud_temp_interaction.py ^
  --data-root dataset_cloudforce_interaction ^
  --out-dir runs/cloud_temp_interaction ^
  --fresh ^
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

Continue from `runs/cloud_temp_interaction/last.pt`:

```cmd
python NewModel\train_cloud_temp_interaction.py ^
  --data-root dataset_cloudforce_interaction ^
  --out-dir runs/cloud_temp_interaction ^
  --resume auto ^
  --epochs 200 ^
  --batch-size 24 ^
  --num-workers 0
```

Resume from a specific checkpoint:

```cmd
python NewModel\train_cloud_temp_interaction.py ^
  --data-root dataset_cloudforce_interaction ^
  --out-dir runs/cloud_temp_interaction ^
  --resume runs/cloud_temp_interaction/last.pt ^
  --epochs 200
```

Use `--lookback 1` if your data is too sparse. Use `--num-workers 0` on Windows.

## Use As RL Reward

The RL reward adapter can now load this model directly. After the first model has
`runs/cloud_temp_interaction/best.pt` or `last.pt`, start RL with:

```cmd
python cloud_rl\train_rl.py ^
  --config cloud_rl\configs\default.yaml ^
  --data-root dataset_cloudforce_interaction ^
  --split train ^
  --resume auto ^
  --reward-checkpoint auto
```

`--reward-checkpoint auto` looks for `runs/cloud_temp_interaction` first. You can
also pass the checkpoint explicitly:

```cmd
python cloud_rl\train_rl.py ^
  --config cloud_rl\configs\default.yaml ^
  --data-root dataset_cloudforce_interaction ^
  --split train ^
  --resume auto ^
  --reward-checkpoint runs/cloud_temp_interaction/best.pt
```

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
