# Cloud RL Planner Prototype

This repo is a zero-shot PyTorch scaffold for the second model: an RL planner that proposes cloud-mask edits and cloud properties from the current weather situation.

It is intentionally built as a **pipeline and architecture prototype**. The reward can come from a trained first-model temperature checkpoint, but you can still fall back to a dummy score for smoke tests. The point of this version is to verify that the data loader, batching, policy, action rasterization, reward interface, checkpointing, and evaluation outputs all work end-to-end.

## Design choices

### Input format

Point the trainer at the exporter output folder, usually `../dataset_out` when
you run from inside `cloud_rl/`. The loader understands:

- `dataset.jsonl`
- `splits/train.jsonl`, `splits/val.jsonl`, `splits/test.jsonl`
- a folder of per-sample JSON files, if you want to use the older prototype layout

Each sample still looks like your manifest entry:

```json
{
  "sample_id": "plovdiv_2026-05-22",
  "city": "Plovdiv",
  "target_temperature_c": 17.7,
  "mask_path": "masks/plovdiv/2026-05-22.png",
  "inputs": {
    "s5p_cloud_fraction": 0.999,
    "s5p_cloud_optical_thickness": 21.66,
    "s5p_cloud_top_height_m": 3502.8,
    "s5p_cloud_top_pressure_pa": 67212.5,
    "s5p_aerosol_index": -0.787,
    "s3_humidity_pct": 78.04,
    "s3_sea_level_pressure_hpa": 1017.23,
    "s3_water_vapour_kg_m2": 27.38,
    "wind_speed_10m_mean": 17.0,
    "wind_direction_10m_dominant": 255.0,
    "shortwave_radiation_sum": 12.39,
    "precipitation_sum": 7.1,
    "cloud_cover_mean": 76.0,
    "surface_pressure_mean": 997.5
  },
  "feature_vector": [0.999, 21.66, 3502.8, 67212.5, -0.787, 78.04, 1017.23, 27.38, 17.0, 255.0, 12.39, 7.1, 76.0, 997.5]
}
```

`mask_path` can be relative to the dataset root or relative to the manifest file.

### Mask normalization

The exporter writes square `256x256` masks by default, so the default training size matches that:

```text
256 x 256  width x height
```

You can change this in `configs/default.yaml` if you export a different mask size:

```yaml
image_height: 256
image_width: 256
```

### Observation tensor

Each batch contains:

```text
obs_map:        [B, 16, H, W]
original_mask:  [B, 1, H, W]
features:       [B, 14]
raw_features:   [B, 14]
target_temp:    [B, 1]
target_temp_norm:[B, 1]
```

`obs_map` is:

```text
cloud mask channel
+ 14 scalar weather features repeated as image planes
+ target-temperature plane
```

The model also receives the 14 scalar features and target temperature separately through an MLP. This gives both spatial and tabular paths.

### Policy output

The policy emits up to **3 action tokens**. It can use fewer by emitting `noop`.

Each action token has:

```text
op: one of noop, remove, modify, create
x01: normalized x location, 0..1
y01: normalized y location, 0..1
radius_px: 2..22 pixels at the normalized output resolution
mass_proxy: 0..1
humidity_pct: 0..100
optical_thickness: 0..100
cloud_top_height_m: 0..12000
cloud_type: 0..1 continuous placeholder
```

The action tokens are rasterized into:

```text
generated_mask: [B, 1, H, W]
property_maps:  [B, 5, H, W]
  channel 0: humidity_norm
  channel 1: optical_thickness_norm
  channel 2: cloud_top_height_norm
  channel 3: mass_proxy
  channel 4: cloud_type
action_maps:    [B, 8, H, W]
  noop, remove, modify, create, mass, humidity, thickness, height
```

`remove` subtracts a soft Gaussian blob from existing white cloud regions. `create` adds a blob. `modify` changes property maps and slightly reinforces an existing cloud area, but it does not create a large new cloud by itself.

## Install

```bash
cd cloud_rl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected data layout

The default exporter layout works like this:

```text
dataset_out/
  dataset.jsonl
  splits/
    train.jsonl
    val.jsonl
    test.jsonl
  masks/
    plovdiv/
      2026-05-22.png
      2026-05-23.png
```

If `data_root=../dataset_out`, then a JSON field like this works:

```json
"mask_path": "masks/plovdiv/2026-05-22.png"
```

## Compute normalization stats

Run this once when you have the exported dataset:

```bash
python train_rl.py --config configs/default.yaml --data-root ../dataset_out --split train --write-stats
```

This writes:

```text
../dataset_out/stats.json
```

The dataset will use that file automatically. If it is missing, fallback default means/stds are used.

## Resume and reward checkpoint

The RL trainer resumes from `policy_latest.pt` automatically when it exists. It
can score actions with the deep first-model checkpoint only:

```bash
python train_rl.py \
  --config configs/default.yaml \
  --data-root ../dataset_out \
  --split train \
  --resume auto \
  --reward-checkpoint /workspace/cloud_project/runs/cloud_temp_deep_480x300/best.pt
```

To use a local deep first-model run folder instead, pass:

```bash
python train_rl.py \
  --config configs/default.yaml \
  --data-root ../dataset_out \
  --split train \
  --resume auto \
  --reward-checkpoint auto \
  --reward-run-dir ../runs/cloud_temp
```

Use `--reward-checkpoint none` only when you explicitly want the dummy reward.

## Train with dummy reward

```bash
python train_rl.py \
  --config configs/default.yaml \
  --data-root ../dataset_out \
  --split train \
  --out-dir ./runs/cloud_rl \
  --resume auto \
  --reward-checkpoint none
```

The dummy reward returns `1.0` for every action. This verifies the full PPO loop, but it does not create meaningful learning. For a smoke test where you want the policy to at least prefer smaller edits, use:

```bash
python train_rl.py \
  --config configs/default.yaml \
  --data-root ../dataset_out \
  --split train \
  --out-dir ./runs/cloud_rl \
  --resume auto \
  --reward-checkpoint none \
  --dummy-budget-penalty 0.05
```

## Resume the first model

The temperature model now resumes from `last.pt` automatically when it exists.
Use the deep trainer for new runs and continued training:

```bash
python train_cloud_temp_deep_ddp.py --data-root ../dataset_out --out-dir ../runs/cloud_temp_deep_480x300 --resume auto
python train_cloud_temp_deep_ddp.py --data-root ../dataset_out --out-dir ../runs/cloud_temp_deep_480x300 --resume none
```

The older lightweight trainer is kept for compatibility, but the RL reward path
now expects deep checkpoints only.

## Zip weights

Bundle both model outputs into one zip:

```bash
python export_weights.py \
  --first-model /workspace/cloud_project/runs/cloud_temp_deep_480x300 \
  --rl-model ./runs/cloud_rl \
  --out ./exports/model_weights.zip
```

If you only have the single checkpoint file from the first model, that works too:

```bash
python export_weights.py \
  --first-model /workspace/cloud_project/runs/cloud_temp_deep_480x300/best.pt \
  --rl-model ./runs/cloud_rl \
  --out ./exports/model_weights.zip
```

## Evaluate / generate proposed masks

```bash
python evaluate.py \
  --checkpoint ./runs/cloud_rl/policy_latest.pt \
  --data-root ../dataset_out \
  --split test \
  --out-dir ./eval_outputs \
  --num-samples 16
```

This writes generated masks and property maps:

```text
eval_outputs/
  predictions.json
  <sample_id>_original.png
  <sample_id>_generated.png
  <sample_id>_humidity_norm.png
  <sample_id>_thickness_norm.png
  <sample_id>_height_norm.png
```

## How to plug in the future ConvLSTM reward model

The reward interface is already fixed as:

```python
reward, info = reward_fn(
    original_mask,      # [B,1,H,W]
    generated_mask,     # [B,1,H,W]
    feature_vector,     # [B,14], raw unnormalized features
    target_temperature, # [B,1], Celsius
    property_maps       # [B,5,H,W]
)
```

Replace this line in `train_rl.py`:

```python
reward_fn = DummyMaxReward(max_score=1.0, optional_budget_penalty=args.dummy_budget_penalty).to(device)
```

with the checkpoint-backed first-model reward:

```python
from cloud_rl.first_model_reward import CloudTempCheckpointReward

reward_fn = CloudTempCheckpointReward(
    "/workspace/cloud_project/runs/cloud_temp_deep_480x300/best.pt",
).to(device)
```

The checkpoint reward uses the generated mask plus the raw feature vector and
scores the policy by how close the first model thinks the resulting temperature
is to the target temperature.

with something like:

```python
from cloud_rl.rewards import WorldModelReward
from my_world_model import ConvLSTMRewardModel

world_model = ConvLSTMRewardModel(...)
world_model.load_state_dict(torch.load("world_model.pt", map_location=device)["model"])
world_model.to(device)
reward_fn = WorldModelReward(world_model).to(device)
```

The included `WorldModelReward` builds a single combined tensor:

```text
concat(original_mask, generated_mask, property_maps, feature_planes, target_plane)
```

That matches your idea of giving the reward model the convolution/combination of the RL input and the produced output. If your ConvLSTM uses a temporal input instead, make a new reward adapter with the same external five-argument interface and internally build the sequence tensor it needs.

## Current limitations

This repo does not prove that cloud edits can reduce real-world temperature. It only implements the RL planner pipeline. The real scientific bottleneck is the reward/world model: without real intervention labels or a carefully validated causal surrogate, PPO will only optimize the learned simulator, not the atmosphere.
