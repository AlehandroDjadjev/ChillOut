# Cloud/Radiation ConvLSTM Training v2

This package is a first-pass PyTorch training scaffold for the forward model:

```text
past dynamic cloud + radiation + atmosphere maps
    -> next-24h radiation anomaly + temperature anomaly maps
```

It is designed for the dataset idea from the conversation:

- image-like cloud fields, not raw RGB ground photos
- around 20 strong dynamic input channels
- 4 output channels used as the supervised training target
- small GPU-safe ConvLSTM defaults
- mixed precision training
- validation metrics per output channel
- no-cloud and cloud-shuffle ablation tests

## Expected tensors

The training code expects preprocessed NumPy files:

```text
data/processed/train_x.npy  [N_time, C_in, H, W]
data/processed/train_y.npy  [N_time, C_out, H, W]
data/processed/val_x.npy    [N_time, C_in, H, W]
data/processed/val_y.npy    [N_time, C_out, H, W]
```

The Dataset turns them into windows:

```text
X = x_data[t - input_len : t]          # [T_in, C_in, H, W]
Y = y_data[t : t + horizon].mean(0)    # [C_out, H, W]
```

Default:
- input_len = 40, for 5 days at 3-hourly resolution
- horizon = 8, for next 24 hours at 3-hourly resolution
- target_mode = mean, for next-24h average response

## Run smoke test

```bash
pip install -r requirements.txt
python src/make_dummy_dataset.py --config configs/cloud_forward_small.yaml
python src/train.py --config configs/cloud_forward_small.yaml
python src/ablation_eval.py --config configs/cloud_forward_small.yaml --checkpoint checkpoints/best.pt
```

The dummy dataset is only a shape/code test. Replace the `.npy` files with real preprocessed Copernicus/Sentinel tensors.

## Design rule

The ConvLSTM should receive:
- cloud-only or cloud-physics channels
- radiation bridge channels
- dynamic atmosphere/rain-support channels

It should not receive raw colored ground pixels in the first training wave.

Static context can be added later through a separate branch, but this v2 scaffold keeps the first wave dynamic-focused.


## Important: 5-day cadence config

If your actual data is one state every ~5 days over 10 years, use:

```bash
python src/make_dummy_dataset.py --config configs/cloud_forward_5day.yaml --train-time 600 --val-time 160 --test-time 160
python src/train.py --config configs/cloud_forward_5day.yaml
```

This config uses:

```text
input_len = 8     # about 40 days of history
horizon = 1       # next 5-day response
```

At this cadence, the model learns cloud/radiation regime response, not hourly cloud movement.


## Aligned downloader → training workflow

This version includes the downloader under:

```text
tools/cloud_state_dataset_downloader.py
```

and a bridge/converter:

```text
src/build_training_tensors_from_downloader.py
```

Typical flow:

```bash
# 1) Download/build raw per-state bundles.
python tools/cloud_state_dataset_downloader.py \
  --start-date 2017-01-01 \
  --end-date 2026-01-01 \
  --out-dir ./cloud_state_dataset \
  --max-cloud 100 \
  --resolution-m 100 \
  --patch-km 10 \
  --build-npz

# 2) Convert builder output to training tensors.
python src/build_training_tensors_from_downloader.py \
  --dataset-dir ./cloud_state_dataset \
  --out-dir data/processed_aligned \
  --profile builder24 \
  --height 64 \
  --width 64 \
  --input-len 8 \
  --horizon 1

# 3) Train.
python src/train.py --config configs/cloud_forward_builder24_5day.yaml

# 4) Check whether cloud fields actually matter.
python src/ablation_eval.py \
  --config configs/cloud_forward_builder24_5day.yaml \
  --checkpoint checkpoints/best_builder24.pt
```

See `docs/builder_alignment.md`.


## Smoke test for the aligned indexed format

```bash
python src/make_dummy_aligned_dataset.py --config configs/cloud_forward_builder24_5day.yaml
python src/train.py --config configs/cloud_forward_builder24_5day.yaml
```

This tests the indexed multi-location data format without real Copernicus downloads.
