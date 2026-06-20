from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from cloud_rl.actions import CREATE, MODIFY, NOOP, REMOVE, rasterize_actions
from cloud_rl.dataset import CloudFolderDataset, collate_cloud_batch
from cloud_rl.first_model_reward import CloudTempCheckpointReward, resolve_first_model_checkpoint
from cloud_rl.models import CloudActorCritic
from cloud_rl.utils import load_config, resolve_existing_path, save_json, to_device


OP_NAMES = {NOOP: "noop", REMOVE: "remove", MODIFY: "modify", CREATE: "create"}


def _tensor_mean(x: torch.Tensor) -> float:
    return float(x.detach().float().mean().cpu())


def _corr(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ax = np.asarray(a, dtype=np.float32)
    by = np.asarray(b, dtype=np.float32)
    if np.std(ax) < 1e-8 or np.std(by) < 1e-8:
        return float("nan")
    return float(np.corrcoef(ax, by)[0, 1])


def _sample_random_actions(batch_size: int, max_actions: int, device: torch.device) -> Dict[str, torch.Tensor]:
    op = torch.randint(0, 4, (batch_size, max_actions), device=device)
    params = torch.rand(batch_size, max_actions, 8, device=device) * 2.0 - 1.0
    return {"op": op, "params": params}


def _forced_actions(batch_size: int, max_actions: int, op_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
    op = torch.full((batch_size, max_actions), op_id, device=device, dtype=torch.long)
    params = torch.rand(batch_size, max_actions, 8, device=device) * 2.0 - 1.0
    params[..., 2:] = 1.0
    return {"op": op, "params": params}


def _empty_actions(batch_size: int, max_actions: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "op": torch.zeros((batch_size, max_actions), device=device, dtype=torch.long),
        "params": torch.zeros((batch_size, max_actions, 8), device=device, dtype=torch.float32),
    }


def _op_histogram(op: torch.Tensor) -> Dict[str, int]:
    flat = op.detach().cpu().reshape(-1)
    counts = Counter(int(x) for x in flat.tolist())
    return {OP_NAMES[k]: counts.get(k, 0) for k in sorted(OP_NAMES)}


def _non_noop_rate(op: torch.Tensor) -> float:
    return float((op != NOOP).float().mean().detach().cpu())


def _summarize_tensor_list(chunks: Dict[str, List[torch.Tensor]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, values in chunks.items():
        if not values:
            continue
        out[key] = torch.cat(values, dim=0).detach().cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cloud_rl/configs/default.yaml")
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", help="train, val, test, or all")
    parser.add_argument("--num-batches", type=int, default=12)
    parser.add_argument("--out-dir", default="./rl_diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(resolve_existing_path(args.config))
    if args.data_root is not None:
        cfg["data_root"] = args.data_root

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reward_ckpt = resolve_first_model_checkpoint(resolve_existing_path(args.reward_checkpoint), prefer_best=False)
    reward_fn = CloudTempCheckpointReward(reward_ckpt).to(device)

    policy_model: Optional[CloudActorCritic] = None
    if args.policy_checkpoint:
        policy_path = resolve_existing_path(args.policy_checkpoint)
        policy_state = torch.load(policy_path, map_location="cpu")
        policy_model = CloudActorCritic(
            obs_channels=1 + 14 + 1,
            feature_dim=14,
            max_actions=int(cfg["policy"]["max_actions"]),
            hidden_dim=int(cfg["policy"]["hidden_dim"]),
        ).to(device)
        policy_model.load_state_dict(policy_state["model"])
        policy_model.eval()
        print(f"Loaded policy checkpoint: {policy_path}")

    data_root = resolve_existing_path(cfg["data_root"])
    image_size = (int(cfg["image_height"]), int(cfg["image_width"]))
    dataset = CloudFolderDataset(data_root, image_size=image_size, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_cloud_batch,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    print(f"Device: {device}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Reward checkpoint: {reward_ckpt}")
    if policy_model is None:
        print("Policy checkpoint: none, running noop/random diagnostics only")

    all_records: List[Dict[str, Any]] = []
    chunks: Dict[str, List[torch.Tensor]] = defaultdict(list)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            batch = to_device(batch, device, non_blocking=(device.type == "cuda"))

            noop_actions = _empty_actions(batch["original_mask"].shape[0], int(cfg["policy"]["max_actions"]), device)
            random_actions = _sample_random_actions(batch["original_mask"].shape[0], int(cfg["policy"]["max_actions"]), device)
            create_actions = _forced_actions(batch["original_mask"].shape[0], int(cfg["policy"]["max_actions"]), CREATE, device)
            remove_actions = _forced_actions(batch["original_mask"].shape[0], int(cfg["policy"]["max_actions"]), REMOVE, device)

            noop_rast = rasterize_actions(batch["original_mask"], noop_actions["op"], noop_actions["params"])
            random_rast = rasterize_actions(batch["original_mask"], random_actions["op"], random_actions["params"])
            create_rast = rasterize_actions(batch["original_mask"], create_actions["op"], create_actions["params"])
            remove_rast = rasterize_actions(batch["original_mask"], remove_actions["op"], remove_actions["params"])

            noop_reward, noop_info = reward_fn(
                batch["original_mask"],
                noop_rast["generated_mask"],
                batch["raw_features"],
                batch["target_temp"],
                noop_rast["property_maps"],
            )
            random_reward, random_info = reward_fn(
                batch["original_mask"],
                random_rast["generated_mask"],
                batch["raw_features"],
                batch["target_temp"],
                random_rast["property_maps"],
            )
            create_reward, create_info = reward_fn(
                batch["original_mask"],
                create_rast["generated_mask"],
                batch["raw_features"],
                batch["target_temp"],
                create_rast["property_maps"],
            )
            remove_reward, remove_info = reward_fn(
                batch["original_mask"],
                remove_rast["generated_mask"],
                batch["raw_features"],
                batch["target_temp"],
                remove_rast["property_maps"],
            )

            policy_reward = None
            policy_info = None
            policy_change = None
            policy_temp_err = None
            op_hist = None
            non_noop = float("nan")

            if policy_model is not None:
                sampled = policy_model.sample(batch["obs_map"], batch["features"], batch["target_temp_norm"])
                policy_rast = rasterize_actions(batch["original_mask"], sampled["op"], sampled["params"])
                policy_reward, policy_info = reward_fn(
                    batch["original_mask"],
                    policy_rast["generated_mask"],
                    batch["raw_features"],
                    batch["target_temp"],
                    policy_rast["property_maps"],
                )
                policy_change = (policy_rast["generated_mask"] - batch["original_mask"]).abs().mean(dim=(1, 2, 3))
                policy_temp_err = policy_info["temp_error_c"]
                op_hist = _op_histogram(sampled["op"])
                non_noop = _non_noop_rate(sampled["op"])
                chunks["policy_reward"].append(policy_reward.view(-1).cpu())
                chunks["policy_change"].append(policy_change.view(-1).cpu())
                chunks["policy_temp_err"].append(policy_temp_err.view(-1).cpu())
                chunks["policy_entropy"].append(sampled["entropy"].view(-1).cpu())
                chunks["reward_delta"].append((policy_reward - noop_reward).view(-1).cpu())
                chunks["temp_delta"].append((noop_info["temp_error_c"] - policy_temp_err).view(-1).cpu())

            noop_change = (noop_rast["generated_mask"] - batch["original_mask"]).abs().mean(dim=(1, 2, 3))
            random_change = (random_rast["generated_mask"] - batch["original_mask"]).abs().mean(dim=(1, 2, 3))
            create_change = (create_rast["generated_mask"] - batch["original_mask"]).abs().mean(dim=(1, 2, 3))
            remove_change = (remove_rast["generated_mask"] - batch["original_mask"]).abs().mean(dim=(1, 2, 3))

            chunks["noop_reward"].append(noop_reward.view(-1).cpu())
            chunks["random_reward"].append(random_reward.view(-1).cpu())
            chunks["create_reward"].append(create_reward.view(-1).cpu())
            chunks["remove_reward"].append(remove_reward.view(-1).cpu())
            chunks["noop_change"].append(noop_change.view(-1).cpu())
            chunks["random_change"].append(random_change.view(-1).cpu())
            chunks["create_change"].append(create_change.view(-1).cpu())
            chunks["remove_change"].append(remove_change.view(-1).cpu())
            chunks["noop_temp_err"].append(noop_info["temp_error_c"].view(-1).cpu())
            chunks["random_temp_err"].append(random_info["temp_error_c"].view(-1).cpu())
            chunks["create_temp_err"].append(create_info["temp_error_c"].view(-1).cpu())
            chunks["remove_temp_err"].append(remove_info["temp_error_c"].view(-1).cpu())

            record = {
                "batch_idx": batch_idx,
                "sample_ids": batch["sample_id"],
                "noop_reward_mean": _tensor_mean(noop_reward),
                "random_reward_mean": _tensor_mean(random_reward),
                "create_reward_mean": _tensor_mean(create_reward),
                "remove_reward_mean": _tensor_mean(remove_reward),
                "noop_temp_err_mean": _tensor_mean(noop_info["temp_error_c"]),
                "random_temp_err_mean": _tensor_mean(random_info["temp_error_c"]),
                "create_temp_err_mean": _tensor_mean(create_info["temp_error_c"]),
                "remove_temp_err_mean": _tensor_mean(remove_info["temp_error_c"]),
                "noop_change_mean": _tensor_mean(noop_change),
                "random_change_mean": _tensor_mean(random_change),
                "create_change_mean": _tensor_mean(create_change),
                "remove_change_mean": _tensor_mean(remove_change),
            }

            if policy_model is not None and policy_reward is not None and policy_temp_err is not None:
                record.update(
                    {
                        "policy_reward_mean": _tensor_mean(policy_reward),
                        "policy_temp_err_mean": _tensor_mean(policy_temp_err),
                        "policy_change_mean": _tensor_mean(policy_change),
                        "reward_delta_mean": _tensor_mean(policy_reward - noop_reward),
                        "temp_delta_mean": _tensor_mean(noop_info["temp_error_c"] - policy_temp_err),
                        "non_noop_rate": non_noop,
                        "op_histogram": op_hist,
                    }
                )

                print(
                    f"batch={batch_idx:03d} "
                f"policy_reward={_tensor_mean(policy_reward):.4f} "
                f"noop_reward={_tensor_mean(noop_reward):.4f} "
                f"random_reward={_tensor_mean(random_reward):.4f} "
                f"create_reward={_tensor_mean(create_reward):.4f} "
                f"remove_reward={_tensor_mean(remove_reward):.4f} "
                f"delta={_tensor_mean(policy_reward - noop_reward):+.4f} "
                f"policy_temp_err={_tensor_mean(policy_temp_err):.4f} "
                f"noop_temp_err={_tensor_mean(noop_info['temp_error_c']):.4f} "
                f"random_temp_err={_tensor_mean(random_info['temp_error_c']):.4f} "
                f"create_temp_err={_tensor_mean(create_info['temp_error_c']):.4f} "
                f"remove_temp_err={_tensor_mean(remove_info['temp_error_c']):.4f} "
                f"policy_change={_tensor_mean(policy_change):.4f} "
                f"noop_rate={non_noop:.3f}"
                )
            else:
                print(
                    f"batch={batch_idx:03d} "
                    f"noop_reward={_tensor_mean(noop_reward):.4f} "
                    f"random_reward={_tensor_mean(random_reward):.4f} "
                    f"create_reward={_tensor_mean(create_reward):.4f} "
                    f"remove_reward={_tensor_mean(remove_reward):.4f} "
                    f"noop_temp_err={_tensor_mean(noop_info['temp_error_c']):.4f} "
                    f"random_temp_err={_tensor_mean(random_info['temp_error_c']):.4f} "
                    f"create_temp_err={_tensor_mean(create_info['temp_error_c']):.4f} "
                    f"remove_temp_err={_tensor_mean(remove_info['temp_error_c']):.4f} "
                    f"noop_change={_tensor_mean(noop_change):.4f} "
                    f"random_change={_tensor_mean(random_change):.4f} "
                    f"create_change={_tensor_mean(create_change):.4f} "
                    f"remove_change={_tensor_mean(remove_change):.4f}"
                )

            all_records.append(record)

    flat = _summarize_tensor_list(chunks)

    summary: Dict[str, Any] = {
        "device": str(device),
        "reward_checkpoint": str(reward_ckpt),
        "policy_checkpoint": args.policy_checkpoint,
        "split": args.split,
        "num_batches": args.num_batches,
        "batch_size": int(cfg["batch_size"]),
        "noop_reward_mean": float(flat["noop_reward"].mean()),
        "random_reward_mean": float(flat["random_reward"].mean()),
        "create_reward_mean": float(flat["create_reward"].mean()),
        "remove_reward_mean": float(flat["remove_reward"].mean()),
        "noop_temp_err_mean": float(flat["noop_temp_err"].mean()),
        "random_temp_err_mean": float(flat["random_temp_err"].mean()),
        "create_temp_err_mean": float(flat["create_temp_err"].mean()),
        "remove_temp_err_mean": float(flat["remove_temp_err"].mean()),
        "noop_change_mean": float(flat["noop_change"].mean()),
        "random_change_mean": float(flat["random_change"].mean()),
        "create_change_mean": float(flat["create_change"].mean()),
        "remove_change_mean": float(flat["remove_change"].mean()),
        "records": all_records,
    }

    if "policy_reward" in flat:
        summary.update(
            {
                "policy_reward_mean": float(flat["policy_reward"].mean()),
                "policy_temp_err_mean": float(flat["policy_temp_err"].mean()),
                "policy_change_mean": float(flat["policy_change"].mean()),
                "policy_entropy_mean": float(flat["policy_entropy"].mean()),
                "reward_delta_mean": float(flat["reward_delta"].mean()),
                "temp_delta_mean": float(flat["temp_delta"].mean()),
                "reward_delta_temp_delta_corr": _corr(flat["reward_delta"].tolist(), flat["temp_delta"].tolist()),
                "non_noop_rate_mean": float(np.mean([r["non_noop_rate"] for r in all_records if "non_noop_rate" in r])),
                "op_histogram_total": {
                    key: sum(int((record.get("op_histogram") or {}).get(key, 0)) for record in all_records)
                    for key in OP_NAMES.values()
                },
            }
        )
    else:
        summary.update(
            {
                "policy_reward_mean": None,
                "policy_temp_err_mean": None,
                "policy_change_mean": None,
                "policy_entropy_mean": None,
                "reward_delta_mean": None,
                "temp_delta_mean": None,
                "reward_delta_temp_delta_corr": None,
                "non_noop_rate_mean": None,
                "op_histogram_total": None,
            }
        )

    save_json(summary, out_dir / "rl_diagnostics.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    print(f"Wrote diagnostics to {out_dir / 'rl_diagnostics.json'}")


if __name__ == "__main__":
    main()
