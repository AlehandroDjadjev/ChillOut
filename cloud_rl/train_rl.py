from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_rl.actions import rasterize_actions
from cloud_rl.dataset import CloudFolderDataset, collate_cloud_batch, compute_stats
from cloud_rl.models import CloudActorCritic
from cloud_rl.ppo import RolloutBatch, ppo_update
from cloud_rl.rewards import DummyMaxReward
from cloud_rl.utils import load_config, resolve_existing_path, save_json, seed_everything, to_device


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def collect_rollout(model, reward_fn, loader_iter, steps: int, device: torch.device) -> RolloutBatch:
    chunks: Dict[str, List[torch.Tensor]] = {
        "obs_map": [], "features": [], "target_temp_norm": [], "original_mask": [],
        "raw_features": [], "target_temp": [], "op": [], "params": [],
        "old_log_prob": [], "returns": [], "advantages": [],
    }
    model.eval()
    for _ in range(steps):
        batch = to_device(next(loader_iter), device)
        with torch.no_grad():
            sampled = model.sample(batch["obs_map"], batch["features"], batch["target_temp_norm"])
            rast = rasterize_actions(batch["original_mask"], sampled["op"], sampled["params"])
            reward, _ = reward_fn(
                batch["original_mask"],
                rast["generated_mask"],
                batch["raw_features"],
                batch["target_temp"],
                rast["property_maps"],
            )
            returns = reward.view(-1, 1)
            advantages = returns - sampled["value"]

        for k in ["obs_map", "features", "target_temp_norm", "original_mask", "raw_features", "target_temp"]:
            chunks[k].append(batch[k].detach().cpu())
        chunks["op"].append(sampled["op"].detach().cpu())
        chunks["params"].append(sampled["params"].detach().cpu())
        chunks["old_log_prob"].append(sampled["log_prob"].detach().cpu())
        chunks["returns"].append(returns.detach().cpu())
        chunks["advantages"].append(advantages.detach().cpu())

    cat = {k: torch.cat(v, dim=0).to(device) for k, v in chunks.items()}
    return RolloutBatch(**cat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="train", help="Dataset split to use: train, val, test, or all.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--write-stats", action="store_true", help="Compute data_root/stats.json before training.")
    parser.add_argument("--dummy-budget-penalty", type=float, default=0.0)
    args = parser.parse_args()

    config_path = resolve_existing_path(args.config)
    cfg = load_config(config_path)
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir

    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.resolved.json")

    data_root = resolve_existing_path(cfg["data_root"])

    if args.write_stats:
        stats = compute_stats(data_root, split=None if args.split == "all" else args.split)
        (data_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats to {data_root / 'stats.json'}")

    image_size = (int(cfg["image_height"]), int(cfg["image_width"]))
    dataset = CloudFolderDataset(data_root, image_size=image_size, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_cloud_batch,
        drop_last=True if len(dataset) >= int(cfg["batch_size"]) else False,
    )
    loader_iter = infinite_loader(loader)

    obs_channels = 1 + 14 + 1
    pcfg = cfg["policy"]
    model = CloudActorCritic(
        obs_channels=obs_channels,
        feature_dim=14,
        max_actions=int(pcfg["max_actions"]),
        hidden_dim=int(pcfg["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(pcfg["lr"]), weight_decay=1e-4)
    reward_fn = DummyMaxReward(max_score=1.0, optional_budget_penalty=args.dummy_budget_penalty).to(device)

    train_cfg = cfg["training"]
    pbar = tqdm(range(1, int(train_cfg["updates"]) + 1), desc="PPO updates")
    history = []
    for update in pbar:
        rollout = collect_rollout(
            model, reward_fn, loader_iter, steps=int(train_cfg["steps_per_update"]), device=device
        )
        model.train()
        logs = {}
        for _ in range(int(pcfg["ppo_epochs"])):
            logs = ppo_update(
                model,
                rollout,
                optimizer,
                clip_eps=float(pcfg["clip_eps"]),
                value_coef=float(pcfg["value_coef"]),
                entropy_coef=float(pcfg["entropy_coef"]),
                grad_clip=float(pcfg["grad_clip"]),
            )
        logs["update"] = update
        history.append(logs)
        pbar.set_postfix({k: round(v, 4) for k, v in logs.items() if isinstance(v, float)})

        if update % int(train_cfg["save_every"]) == 0 or update == int(train_cfg["updates"]):
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
                "update": update,
            }
            torch.save(ckpt, out_dir / f"policy_update_{update:06d}.pt")
            torch.save(ckpt, out_dir / "policy_latest.pt")
            save_json({"history": history[-200:]}, out_dir / "history.tail.json")

    print(f"Done. Latest checkpoint: {out_dir / 'policy_latest.pt'}")


if __name__ == "__main__":
    main()
