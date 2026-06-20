from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cloud_rl.actions import actions_to_jsonable, rasterize_actions
from cloud_rl.dataset import CloudFolderDataset, collate_cloud_batch
from cloud_rl.models import CloudActorCritic
from cloud_rl.rewards import DummyMaxReward
from cloud_rl.utils import load_config, resolve_existing_path, to_device


def save_mask(t: torch.Tensor, path: Path) -> None:
    arr = t.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test", help="Dataset split to use: train, val, test, or all.")
    parser.add_argument("--out-dir", default="./eval_outputs")
    parser.add_argument("--num-samples", type=int, default=16)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config_path = resolve_existing_path(args.config or "configs/default.yaml")
    cfg = ckpt.get("cfg", load_config(config_path))
    if args.config is not None:
        cfg.update(load_config(config_path))
    if args.data_root is not None:
        cfg["data_root"] = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = resolve_existing_path(cfg["data_root"])
    dataset = CloudFolderDataset(
        data_root,
        image_size=(int(cfg["image_height"]), int(cfg["image_width"])),
        split=args.split,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_cloud_batch)

    model = CloudActorCritic(
        obs_channels=16,
        feature_dim=14,
        max_actions=int(cfg["policy"]["max_actions"]),
        hidden_dim=int(cfg["policy"]["hidden_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    reward_fn = DummyMaxReward().to(device)

    records = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples:
                break
            batch = to_device(batch, device)
            sampled = model.sample(batch["obs_map"], batch["features"], batch["target_temp_norm"])
            rast = rasterize_actions(batch["original_mask"], sampled["op"], sampled["params"])
            reward, info = reward_fn(
                batch["original_mask"], rast["generated_mask"], batch["raw_features"], batch["target_temp"], rast["property_maps"]
            )
            sid = batch["sample_id"][0]
            save_mask(batch["original_mask"][0, 0], out_dir / f"{sid}_original.png")
            save_mask(rast["generated_mask"][0, 0], out_dir / f"{sid}_generated.png")
            save_mask(rast["property_maps"][0, 0], out_dir / f"{sid}_humidity_norm.png")
            save_mask(rast["property_maps"][0, 1], out_dir / f"{sid}_thickness_norm.png")
            save_mask(rast["property_maps"][0, 2], out_dir / f"{sid}_height_norm.png")
            records.append({
                "sample_id": sid,
                "target_temperature_c": float(batch["target_temp"][0, 0].cpu()),
                "reward": float(reward[0, 0].cpu()),
                "actions": actions_to_jsonable(sampled["op"][0].cpu(), sampled["params"][0].cpu()),
                "outputs": {
                    "original_mask": f"{sid}_original.png",
                    "generated_mask": f"{sid}_generated.png",
                    "humidity_norm": f"{sid}_humidity_norm.png",
                    "thickness_norm": f"{sid}_thickness_norm.png",
                    "height_norm": f"{sid}_height_norm.png",
                },
            })

    (out_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} predictions to {out_dir}")


if __name__ == "__main__":
    main()
