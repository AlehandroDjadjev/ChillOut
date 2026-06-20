from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_rl.actions import CREATE, NOOP, REMOVE, rasterize_actions
from cloud_rl.dataset import CloudFolderDataset, collate_cloud_batch, compute_stats
from cloud_rl.first_model_reward import CloudTempCheckpointReward, resolve_first_model_checkpoint
from cloud_rl.models import CloudActorCritic
from cloud_rl.ppo import RolloutBatch, ppo_update
from cloud_rl.rewards import DummyMaxReward
from cloud_rl.targeting import augment_target_temperature
from cloud_rl.utils import load_config, resolve_existing_path, save_json, seed_everything, to_device


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def random_actions(batch_size: int, max_actions: int, device: torch.device) -> Dict[str, torch.Tensor]:
    op_choice = torch.randint(0, 3, (batch_size, max_actions), device=device)
    op = torch.where(op_choice == 2, torch.full_like(op_choice, CREATE), op_choice)
    return {
        "op": op,
        "params": torch.rand(batch_size, max_actions, 8, device=device) * 2.0 - 1.0,
    }


def forced_actions(batch_size: int, max_actions: int, op_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
    op = torch.full((batch_size, max_actions), op_id, dtype=torch.long, device=device)
    params = torch.rand(batch_size, max_actions, 8, device=device) * 2.0 - 1.0
    params[..., 2] = 1.0
    return {"op": op, "params": params}


def noop_actions(batch_size: int, max_actions: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "op": torch.full((batch_size, max_actions), NOOP, dtype=torch.long, device=device),
        "params": torch.zeros(batch_size, max_actions, 8, device=device),
    }


def resolve_rl_resume_checkpoint(source: str, out_dir: Path) -> Path | None:
    if source in {"", "none", None}:
        return None

    if source == "auto":
        candidates = [out_dir / "policy_latest.pt"]
        candidates.extend(sorted(out_dir.glob("policy_update_*.pt"), reverse=True))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    path = Path(source)
    if path.is_dir():
        candidates = [path / "policy_latest.pt"]
        candidates.extend(sorted(path.glob("policy_update_*.pt"), reverse=True))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No policy checkpoints found in {path}")

    if path.exists():
        return path

    raise FileNotFoundError(f"Resume checkpoint not found: {path}")


def load_history_tail(out_dir: Path) -> List[Dict]:
    history_path = out_dir / "history.tail.json"
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(payload.get("history", []))


def select_guided_actions(
    model,
    reward_fn,
    batch: Dict[str, torch.Tensor],
    policy_sample: Dict[str, torch.Tensor],
    max_actions: int,
    random_candidates: int,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    b = batch["original_mask"].shape[0]
    candidates = [
        {"op": policy_sample["op"], "params": policy_sample["params"]},
        noop_actions(b, max_actions, device),
        forced_actions(b, max_actions, CREATE, device),
        forced_actions(b, max_actions, REMOVE, device),
    ]
    candidates.extend(random_actions(b, max_actions, device) for _ in range(max(0, random_candidates)))

    rewards = []
    info_by_key: Dict[str, List[torch.Tensor]] = {}
    for candidate in candidates:
        rast = rasterize_actions(batch["original_mask"], candidate["op"], candidate["params"])
        reward, info = reward_fn(
            batch["original_mask"],
            rast["generated_mask"],
            batch["raw_features"],
            batch["target_temp"],
            rast["property_maps"],
        )
        rewards.append(reward)
        for key, value in info.items():
            info_by_key.setdefault(key, []).append(value)

    reward_stack = torch.stack(rewards, dim=0)
    best_idx = reward_stack.squeeze(-1).argmax(dim=0)
    row_idx = torch.arange(b, device=device)
    op_stack = torch.stack([candidate["op"] for candidate in candidates], dim=0)
    param_stack = torch.stack([candidate["params"] for candidate in candidates], dim=0)
    selected = {
        "op": op_stack[best_idx, row_idx],
        "params": param_stack[best_idx, row_idx],
    }
    selected_reward = reward_stack[best_idx, row_idx]
    selected_info = {
        key: torch.stack(values, dim=0)[best_idx, row_idx]
        for key, values in info_by_key.items()
    }
    log_prob, entropy, value = model.evaluate_actions(
        batch["obs_map"],
        batch["features"],
        batch["target_temp_norm"],
        selected["op"],
        selected["params"],
    )
    selected.update({"log_prob": log_prob, "entropy": entropy, "value": value})
    return selected, selected_reward, selected_info


def collect_rollout(
    model,
    reward_fn,
    loader_iter,
    steps: int,
    device: torch.device,
    cfg: Dict,
    target_std: float,
) -> tuple[RolloutBatch, Dict[str, float]]:
    chunks: Dict[str, List[torch.Tensor]] = {
        "obs_map": [], "features": [], "target_temp_norm": [], "original_mask": [],
        "raw_features": [], "target_temp": [], "op": [], "params": [],
        "old_log_prob": [], "returns": [], "advantages": [],
    }
    debug_chunks: Dict[str, List[torch.Tensor]] = {
        "reward": [],
        "generated_temp_err": [],
        "original_temp_err": [],
        "temp_improvement": [],
        "target_temp": [],
    }
    model.eval()
    with torch.inference_mode():
        for _ in range(steps):
            batch = to_device(next(loader_iter), device, non_blocking=(device.type == "cuda"))
            batch = augment_target_temperature(batch, cfg, target_std=target_std, device=device)
            if device.type == "cuda":
                batch["obs_map"] = batch["obs_map"].contiguous(memory_format=torch.channels_last)
                batch["original_mask"] = batch["original_mask"].contiguous(memory_format=torch.channels_last)
            sampled = model.sample(batch["obs_map"], batch["features"], batch["target_temp_norm"])
            guided_candidates = int((cfg.get("policy") or {}).get("guided_random_candidates", 0))
            if guided_candidates > 0:
                sampled, reward, reward_info = select_guided_actions(
                    model,
                    reward_fn,
                    batch,
                    sampled,
                    max_actions=int((cfg.get("policy") or {}).get("max_actions", 3)),
                    random_candidates=guided_candidates,
                    device=device,
                )
            else:
                rast = rasterize_actions(batch["original_mask"], sampled["op"], sampled["params"])
                reward, reward_info = reward_fn(
                    batch["original_mask"],
                    rast["generated_mask"],
                    batch["raw_features"],
                    batch["target_temp"],
                    rast["property_maps"],
                )
            returns = reward.view(-1, 1)
            advantages = returns - sampled["value"]
            if "temp_error_c" in reward_info:
                debug_chunks["generated_temp_err"].append(reward_info["temp_error_c"].detach().cpu())
            if "original_temp_error_c" in reward_info:
                debug_chunks["original_temp_err"].append(reward_info["original_temp_error_c"].detach().cpu())
                if "temp_error_c" in reward_info:
                    debug_chunks["temp_improvement"].append(
                        (reward_info["original_temp_error_c"] - reward_info["temp_error_c"]).detach().cpu()
                    )
            debug_chunks["reward"].append(reward.detach().cpu())
            debug_chunks["target_temp"].append(batch["target_temp"].detach().cpu())

            for k in ["obs_map", "features", "target_temp_norm", "original_mask", "raw_features", "target_temp"]:
                chunks[k].append(batch[k].detach().cpu())
            chunks["op"].append(sampled["op"].detach().cpu())
            chunks["params"].append(sampled["params"].detach().cpu())
            chunks["old_log_prob"].append(sampled["log_prob"].detach().cpu())
            chunks["returns"].append(returns.detach().cpu())
            chunks["advantages"].append(advantages.detach().cpu())

    cat = {k: torch.cat(v, dim=0).to(device) for k, v in chunks.items()}
    debug = {}
    for key, values in debug_chunks.items():
        if values:
            debug[key] = float(torch.cat(values, dim=0).mean().item())
    return RolloutBatch(**cat), debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="train", help="Dataset split to use: train, val, test, or all.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", default="auto", help="Resume from policy_latest.pt, a checkpoint path, or none.")
    parser.add_argument("--write-stats", action="store_true", help="Compute data_root/stats.json before training.")
    parser.add_argument(
        "--reward-checkpoint",
        default="auto",
        help="First-model checkpoint path, reward run dir, auto, or none for the dummy reward.",
    )
    parser.add_argument("--reward-run-dir", default="runs/cloud_temp", help="Directory used when reward-checkpoint=auto.")
    parser.add_argument("--reward-scale", type=float, default=5.0, help="Temperature error scale for the first-model reward.")
    parser.add_argument("--reward-improvement-gain", type=float, default=100.0)
    parser.add_argument("--reward-absolute-error-weight", type=float, default=0.0)
    parser.add_argument("--reward-budget-penalty", type=float, default=0.0)
    parser.add_argument("--dummy-budget-penalty", type=float, default=0.0, help="Penalty used only with the dummy reward.")
    args = parser.parse_args()

    config_path = resolve_existing_path(args.config)
    cfg = load_config(config_path)
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir

    seed_everything(int(cfg.get("seed", 42)))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.resolved.json")

    data_root = resolve_existing_path(cfg["data_root"])

    if args.write_stats:
        stats = compute_stats(data_root, split=None if args.split == "all" else args.split)
        (data_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats to {data_root / 'stats.json'}")

    image_size = (int(cfg["image_height"]), int(cfg["image_width"]))
    num_workers = max(0, min(int(cfg.get("num_workers", 0)), 1))
    dataset = CloudFolderDataset(data_root, image_size=image_size, split=args.split)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(cfg["batch_size"]),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_cloud_batch,
        "drop_last": True if len(dataset) >= int(cfg["batch_size"]) else False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(**loader_kwargs)
    loader_iter = infinite_loader(loader)

    obs_channels = 1 + 14 + 1
    pcfg = cfg["policy"]
    model = CloudActorCritic(
        obs_channels=obs_channels,
        feature_dim=14,
        max_actions=int(pcfg["max_actions"]),
        hidden_dim=int(pcfg["hidden_dim"]),
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(pcfg["lr"]), weight_decay=1e-4)

    reward_fn = None
    if args.reward_checkpoint == "none":
        reward_fn = DummyMaxReward(max_score=1.0, optional_budget_penalty=args.dummy_budget_penalty).to(device)
        print("Using dummy reward.")
    else:
        reward_source = args.reward_checkpoint
        if reward_source == "auto":
            reward_source = str(resolve_existing_path(args.reward_run_dir))
        reward_ckpt = resolve_first_model_checkpoint(reward_source, prefer_best=True)
        reward_fn = CloudTempCheckpointReward(
            reward_ckpt,
            reward_scale_c=args.reward_scale,
            improvement_gain=args.reward_improvement_gain,
            absolute_error_weight=args.reward_absolute_error_weight,
            optional_budget_penalty=args.reward_budget_penalty,
        ).to(device)
        if device.type == "cuda":
            reward_fn = reward_fn.to(memory_format=torch.channels_last)
        print(f"Using first-model reward checkpoint: {reward_fn.checkpoint_path}")

    resume_ckpt = resolve_rl_resume_checkpoint(args.resume, out_dir)
    start_update = 1
    history: List[Dict] = load_history_tail(out_dir)
    if resume_ckpt is not None:
        state = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_update = int(state.get("update", 0)) + 1
        print(f"Resumed RL training from {resume_ckpt} at update {start_update}.")
    else:
        print("Starting RL training from scratch.")

    train_cfg = cfg["training"]
    total_updates = int(train_cfg["updates"])
    if start_update > total_updates:
        print(
            f"Checkpoint already reached update {start_update - 1}, which is beyond the configured total {total_updates}."
        )
        return

    pbar = tqdm(range(start_update, total_updates + 1), desc="PPO updates")
    for update in pbar:
        rollout, rollout_debug = collect_rollout(
            model,
            reward_fn,
            loader_iter,
            steps=int(train_cfg["steps_per_update"]),
            device=device,
            cfg=cfg,
            target_std=float(dataset.target_std),
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
                bc_coef=float(pcfg.get("bc_coef", 0.0)),
                grad_clip=float(pcfg["grad_clip"]),
            )
        logs["update"] = update
        logs.update(rollout_debug)
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
