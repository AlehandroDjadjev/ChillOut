from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_rl.actions import actions_to_jsonable, rasterize_actions
from cloud_rl.dataset import CloudFolderDataset, collate_cloud_batch
from cloud_rl.first_model_reward import CloudTempCheckpointReward, resolve_first_model_checkpoint
from cloud_rl.models import CloudActorCritic
from cloud_rl.ppo import ppo_update
from cloud_rl.rewards import DummyMaxReward
from cloud_rl.targeting import augment_target_goal
from cloud_rl.utils import load_config, resolve_existing_path, save_json, seed_everything, to_device
from train_rl import collect_rollout, infinite_loader, resolve_auto_reward_source, resolve_rl_resume_checkpoint


PROPERTY_MAP_NAMES = [
    "cloud_probability",
    "aot_norm",
    "cloud_layer",
    "texture_norm",
    "cirrus_proxy",
]


def save_mask(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().float().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    x = tensor.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "min": float(x.min()) if x.numel() else 0.0,
        "max": float(x.max()) if x.numel() else 0.0,
        "mean": float(x.mean()) if x.numel() else 0.0,
        "std": float(x.std(unbiased=False)) if x.numel() else 0.0,
        "finite": bool(torch.isfinite(x).all().item()) if x.numel() else True,
    }


def small_vector(tensor: torch.Tensor, limit: int = 32) -> List[float]:
    flat = tensor.detach().float().cpu().flatten()
    return [float(x) for x in flat[:limit]]


def load_reward(args: argparse.Namespace, device: torch.device):
    if args.reward_checkpoint == "none":
        print("Using dummy reward.")
        return DummyMaxReward(max_score=1.0, optional_budget_penalty=args.dummy_budget_penalty).to(device)

    reward_source = args.reward_checkpoint
    if reward_source == "auto":
        reward_source = str(resolve_auto_reward_source(args.reward_run_dir))
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
    return reward_fn


def reward_call(reward_fn, batch: Dict[str, torch.Tensor], generated_mask: torch.Tensor, property_maps: torch.Tensor):
    return reward_fn(
        batch["original_mask"],
        generated_mask,
        batch["raw_features"],
        batch["target_temp"],
        property_maps,
        original_mask_sequence=batch.get("original_mask_sequence"),
        cloud_tensor_sequence=batch.get("cloud_tensor_sequence"),
        raw_feature_sequence=batch.get("raw_feature_sequence"),
        trend_features=batch.get("trend_features"),
        current_temperature=batch.get("current_temp"),
        radiation_context_features=batch.get("radiation_context_features"),
        radiation_clear_wm2=batch.get("radiation_clear_wm2"),
        current_radiation_loss_wm2=batch.get("current_radiation_loss_wm2"),
    )


@torch.no_grad()
def export_io_examples(
    model: CloudActorCritic,
    reward_fn,
    loader: DataLoader,
    cfg: Dict[str, Any],
    dataset: CloudFolderDataset,
    out_dir: Path,
    device: torch.device,
    num_samples: int,
    stochastic: bool,
) -> None:
    model.eval()
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    written = 0
    for batch in loader:
        batch = to_device(batch, device, non_blocking=(device.type == "cuda"))
        batch = augment_target_goal(batch, cfg, target_std=float(dataset.target_std), device=device)
        if device.type == "cuda":
            batch["obs_map"] = batch["obs_map"].contiguous(memory_format=torch.channels_last)
            batch["original_mask"] = batch["original_mask"].contiguous(memory_format=torch.channels_last)

        policy_out = model(batch["obs_map"], batch["features"], batch["target_temp_norm"])
        sampled = (
            model.sample(batch["obs_map"], batch["features"], batch["target_temp_norm"])
            if stochastic
            else model.deterministic(batch["obs_map"], batch["features"], batch["target_temp_norm"])
        )
        rast = rasterize_actions(batch["original_mask"], sampled["op"], sampled["params"])
        reward, info = reward_call(reward_fn, batch, rast["generated_mask"], rast["property_maps"])

        batch_size = batch["original_mask"].shape[0]
        for row in range(batch_size):
            if written >= num_samples:
                break
            sample_id = str(batch["sample_id"][row])
            prefix = f"{written:03d}_{sample_id}"
            original_name = f"{prefix}_input_mask.png"
            generated_name = f"{prefix}_generated_mask.png"
            save_mask(batch["original_mask"][row, 0], examples_dir / original_name)
            save_mask(rast["generated_mask"][row, 0], examples_dir / generated_name)

            property_outputs: Dict[str, str] = {}
            for idx, name in enumerate(PROPERTY_MAP_NAMES):
                file_name = f"{prefix}_{name}.png"
                save_mask(rast["property_maps"][row, idx], examples_dir / file_name)
                property_outputs[name] = str(Path("examples") / file_name)

            input_record = {
                "sample_id": sample_id,
                "target_temperature_c": float(batch["target_temp"][row, 0].detach().cpu()),
                "target_temp_norm": float(batch["target_temp_norm"][row, 0].detach().cpu()),
                "current_temperature_c": float(batch.get("current_temp", torch.zeros_like(batch["target_temp"]))[row, 0].detach().cpu()),
                "current_radiation_loss_wm2": float(batch.get("current_radiation_loss_wm2", torch.zeros_like(batch["target_temp"]))[row, 0].detach().cpu()),
                "target_radiation_loss_wm2": float(batch.get("target_radiation_loss_wm2", batch["target_temp"])[row, 0].detach().cpu()),
                "radiation_clear_wm2": float(batch.get("radiation_clear_wm2", torch.zeros_like(batch["target_temp"]))[row, 0].detach().cpu()),
                "obs_map": tensor_stats(batch["obs_map"][row]),
                "original_mask": tensor_stats(batch["original_mask"][row]),
                "features": {
                    "shape": list(batch["features"][row].shape),
                    "first_values": small_vector(batch["features"][row]),
                },
                "raw_features": {
                    "shape": list(batch["raw_features"][row].shape),
                    "first_values": small_vector(batch["raw_features"][row]),
                },
                "trend_features": tensor_stats(batch.get("trend_features", torch.empty(0, device=device))[row])
                if "trend_features" in batch else None,
                "mask_sequence": tensor_stats(batch.get("original_mask_sequence", torch.empty(0, device=device))[row])
                if "original_mask_sequence" in batch else None,
            }

            output_record = {
                "policy": {
                    "op_logits": policy_out.op_logits[row].detach().float().cpu().tolist(),
                    "param_mu": policy_out.param_mu[row].detach().float().cpu().tolist(),
                    "param_std": policy_out.param_std[row].detach().float().cpu().tolist(),
                    "value": float(policy_out.value[row, 0].detach().cpu()),
                    "sampled_log_prob": float(sampled["log_prob"][row, 0].detach().cpu()),
                    "sampled_entropy": float(sampled["entropy"][row, 0].detach().cpu()),
                },
                "actions": actions_to_jsonable(sampled["op"][row].detach().cpu(), sampled["params"][row].detach().cpu()),
                "generated_mask": tensor_stats(rast["generated_mask"][row]),
                "property_maps": tensor_stats(rast["property_maps"][row]),
                "reward": float(reward[row, 0].detach().cpu()),
                "reward_info": {
                    key: float(value[row, 0].detach().cpu())
                    for key, value in info.items()
                    if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] > row
                },
            }

            records.append(
                {
                    "index": written,
                    "input": input_record,
                    "output": output_record,
                    "files": {
                        "input_mask": str(Path("examples") / original_name),
                        "generated_mask": str(Path("examples") / generated_name),
                        **property_outputs,
                    },
                    "meta": batch["meta"][row],
                }
            )
            written += 1
        if written >= num_samples:
            break

    save_json({"examples": records}, out_dir / "rl_io_debug.json")
    print(f"Wrote {len(records)} RL input/output examples to {out_dir / 'rl_io_debug.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local short-run RL trainer with input/output debug exports.")
    parser.add_argument("--config", default="cloud_rl/configs/debug_local.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--out-dir", default="runs/local_rl_debug")
    parser.add_argument("--resume", default="none")
    parser.add_argument("--reward-checkpoint", default="none", help="auto, none, run dir, or checkpoint path.")
    parser.add_argument("--reward-run-dir", default="auto")
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--reward-improvement-gain", type=float, default=100.0)
    parser.add_argument("--reward-absolute-error-weight", type=float, default=0.0)
    parser.add_argument("--reward-budget-penalty", type=float, default=0.0)
    parser.add_argument("--dummy-budget-penalty", type=float, default=0.02)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--steps-per-update", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--stochastic-eval", action="store_true")
    parser.add_argument("--write-stats", action="store_true")
    args = parser.parse_args()

    cfg = load_config(resolve_existing_path(args.config))
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    cfg["out_dir"] = args.out_dir
    cfg["batch_size"] = args.batch_size
    cfg.setdefault("training", {})
    cfg["training"]["updates"] = args.updates
    cfg["training"]["steps_per_update"] = args.steps_per_update
    cfg["training"]["save_every"] = max(1, min(args.updates, int(cfg["training"].get("save_every", 1))))
    cfg.setdefault("policy", {})
    cfg["policy"]["ppo_epochs"] = int(cfg["policy"].get("ppo_epochs", 1))
    cfg["num_workers"] = 0

    seed_everything(int(cfg.get("seed", 42)))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    data_root = resolve_existing_path(cfg["data_root"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.resolved.json")

    image_size = (int(cfg["image_height"]), int(cfg["image_width"]))
    train_dataset = CloudFolderDataset(
        data_root,
        image_size=image_size,
        split=args.split,
        lookback=int(cfg.get("lookback", 4)),
        max_gap_days=float(cfg.get("max_gap_days", 12.0)),
    )
    if args.write_stats:
        stats = {
            "feature_keys": list(train_dataset.feature_keys),
            "feature_mean": train_dataset.feature_mean.tolist(),
            "feature_std": train_dataset.feature_std.tolist(),
            "target_temp_mean": [float(train_dataset.target_mean)],
            "target_temp_std": [float(train_dataset.target_std)],
        }
        (data_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats to {data_root / 'stats.json'}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_cloud_batch,
        drop_last=len(train_dataset) >= int(cfg["batch_size"]),
    )
    train_iter = infinite_loader(train_loader)

    feature_dim = int(train_dataset.feature_dim)
    model = CloudActorCritic(
        obs_channels=1 + feature_dim + 1,
        feature_dim=feature_dim,
        max_actions=int(cfg["policy"]["max_actions"]),
        hidden_dim=int(cfg["policy"]["hidden_dim"]),
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["policy"].get("lr", 1e-4)),
        weight_decay=float(cfg["policy"].get("weight_decay", 1e-4)),
    )
    reward_fn = load_reward(args, device)
    cfg["reward_target_kind"] = getattr(reward_fn, "reward_target_kind", "temperature")

    resume_ckpt = resolve_rl_resume_checkpoint(args.resume, out_dir)
    start_update = 1
    history: List[Dict[str, Any]] = []
    if resume_ckpt is not None:
        state = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_update = int(state.get("update", 0)) + 1
        print(f"Resumed from {resume_ckpt} at update {start_update}.")
    else:
        print("Starting local RL debug training from scratch.")

    for update in tqdm(range(start_update, args.updates + 1), desc="local PPO"):
        rollout, debug = collect_rollout(
            model,
            reward_fn,
            train_iter,
            steps=int(cfg["training"]["steps_per_update"]),
            device=device,
            cfg=cfg,
            target_std=float(train_dataset.target_std),
        )
        logs: Dict[str, Any] = {}
        for _ in range(int(cfg["policy"].get("ppo_epochs", 1))):
            logs = ppo_update(
                model,
                rollout,
                optimizer,
                clip_eps=float(cfg["policy"].get("clip_eps", 0.2)),
                value_coef=float(cfg["policy"].get("value_coef", 0.5)),
                entropy_coef=float(cfg["policy"].get("entropy_coef", 0.001)),
                bc_coef=float(cfg["policy"].get("bc_coef", 0.0)),
                grad_clip=float(cfg["policy"].get("grad_clip", 0.25)),
            )
        logs["update"] = update
        logs.update(debug)
        history.append(logs)
        save_json({"history": history}, out_dir / "history.local_debug.json")

    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg, "update": args.updates}
    torch.save(ckpt, out_dir / "policy_latest.pt")
    print(f"Wrote checkpoint to {out_dir / 'policy_latest.pt'}")

    eval_split = args.eval_split
    try:
        eval_dataset = CloudFolderDataset(
            data_root,
            image_size=image_size,
            split=eval_split,
            lookback=int(cfg.get("lookback", 4)),
            max_gap_days=float(cfg.get("max_gap_days", 12.0)),
        )
    except Exception as exc:
        print(f"[WARN] Could not load eval split {eval_split!r}: {exc}. Falling back to training split.")
        eval_dataset = train_dataset
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_cloud_batch)
    export_io_examples(
        model,
        reward_fn,
        eval_loader,
        cfg,
        eval_dataset,
        out_dir,
        device,
        num_samples=int(args.eval_samples),
        stochastic=bool(args.stochastic_eval),
    )


if __name__ == "__main__":
    main()
