#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader




def enable_frozen_lstm_input_grads(model):
    """Keep frozen reward model deterministic, but allow CuDNN LSTM backward.

    CuDNN RNN backward requires nn.LSTM modules to be in training mode even when
    weights are frozen and gradients are only needed w.r.t. inputs.
    BatchNorm/Dropout stay eval; only nn.LSTM modules are flipped.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            m.train()

def import_bottom_model():
    last = None
    for name in [
        "train_cloud_temp_cloudforced_radiation_v6",
        "train_cloud_temp_cloudforced_radiation_v6_FIXED2",
        "train_cloud_temp_hybrid_convlstm_v6",
    ]:
        try:
            return __import__(name, fromlist=["dummy"])
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Could not import v6 trainer module. Last error: {last}")


def strip_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


class CloudEditPolicy(nn.Module):
    def __init__(self, num_cloud_features: int, world_dim: int, trend_dim: int, hidden: int = 128, max_mask_delta: float = 3.0, max_cloud_delta_z: float = 1.5):
        super().__init__()
        self.max_mask_delta = float(max_mask_delta)
        self.max_cloud_delta_z = float(max_cloud_delta_z)
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 24, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(24, 32, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(32, 24, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(24, 1, 3, padding=1),
        )
        cond_in = num_cloud_features + world_dim + trend_dim + 3
        self.cond = nn.Sequential(
            nn.Linear(cond_in, hidden), nn.LayerNorm(hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(inplace=True),
        )
        self.cloud_delta = nn.Linear(hidden, num_cloud_features)
        self.cond_to_mask = nn.Linear(hidden, 1)

    def forward(self, mask, cloud, world, trend, target_norm, actual_norm):
        b, t, _, h, w = mask.shape
        cloud_last = cloud[:, -1]
        world_last = world[:, -1]
        trend_last = trend[:, -1]
        target_delta = target_norm - actual_norm
        cond_vec = torch.cat([cloud_last, world_last, trend_last, target_norm, actual_norm, target_delta], dim=-1)
        z = self.cond(cond_vec)
        cloud_step = self.max_cloud_delta_z * torch.tanh(self.cloud_delta(z)).view(b, 1, -1)
        cloud_out = cloud + cloud_step.expand(-1, t, -1)
        c = torch.tanh(self.cond_to_mask(z)).view(b, 1, 1, 1, 1).expand(-1, t, -1, h, w)
        x = torch.cat([mask, c], dim=2).reshape(b * t, 2, h, w)
        delta_logits = self.max_mask_delta * torch.tanh(self.mask_net(x)).view(b, t, 1, h, w)
        eps = 1e-4
        m = mask.clamp(eps, 1 - eps)
        base_logits = torch.log(m / (1 - m))
        mask_out = torch.sigmoid(base_logits + delta_logits)
        return mask_out, cloud_out


def total_variation(mask):
    return (mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]).abs().mean() + (mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]).abs().mean()


def save_strip(seq, path):
    arr = seq.detach().float().cpu().clamp(0, 1).numpy()
    imgs = []
    for i in range(arr.shape[0]):
        im = Image.fromarray((arr[i, 0] * 255).astype(np.uint8), mode="L")
        im = ImageOps.expand(im, border=2, fill=128)
        imgs.append(im)
    out = Image.new("L", (sum(i.width for i in imgs), max(i.height for i in imgs)), 0)
    x = 0
    for im in imgs:
        out.paste(im, (x, 0)); x += im.width
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def reward_forward(reward_model, mask, cloud, world, trend, context_scale):
    if mask.is_cuda:
        with torch.backends.cudnn.flags(enabled=False):
            return reward_model(mask, cloud, world, trend, context_scale=context_scale)
    return reward_model(mask, cloud, world, trend, context_scale=context_scale)


PATCHED_CUDNN_LSTM_INPUT_GRADS = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="runs/cloud_radiation_policy")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--target-jitter-wm2", type=float, default=120.0)
    ap.add_argument("--lambda-anchor-mask", type=float, default=0.05)
    ap.add_argument("--lambda-anchor-cloud", type=float, default=0.05)
    ap.add_argument("--lambda-tv", type=float, default=0.04)
    ap.add_argument("--lambda-range", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    mod = import_bottom_model()
    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = mod.load_metadata(root)
    raw = ckpt.get("raw_feature_names", meta["raw_feature_names"])
    cloud_names = ckpt.get("cloud_feature_names", meta["cloud_feature_names"])
    world_names = ckpt.get("world_feature_names", meta["world_feature_names"])
    x_norm = mod.Normalizer.from_state_dict(ckpt["normalizer"])
    delta_norm = mod.TargetNormalizer.from_state_dict(ckpt.get("delta_normalizer", ckpt["target_normalizer"]))
    h = int(ckpt.get("image_height", 160)); w = int(ckpt.get("image_width", 160))
    common = dict(root=root, raw_names=raw, cloud_names=cloud_names, world_names=world_names, x_norm=x_norm, delta_norm=delta_norm, image_height=h, image_width=w, lookback=int(ckpt.get("lookback", 4)), max_gap_days=float(ckpt.get("args", {}).get("max_gap_days", 12.0)), augment=False, cache_images=False)
    train_ds = mod.CloudTempResidualSequenceDataset(records=mod.load_records(root, "train"), **common)
    val_ds = mod.CloudTempResidualSequenceDataset(records=mod.load_records(root, "val"), **common)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = mod.ResidualTrendSplitConvLSTM(**ckpt["model_kwargs"]).to(device)
    reward_model.load_state_dict(strip_prefix(ckpt["model_state"]))
    enable_frozen_lstm_input_grads(reward_model)
    for p in reward_model.parameters(): p.requires_grad_(False)
    rad_scale = float(ckpt.get("args", {}).get("radiation_loss_scale", 300.0))
    context_scale = float(ckpt.get("args", {}).get("context_scale", 0.15))
    policy = CloudEditPolicy(len(cloud_names), len(world_names), 4, hidden=128).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        enable_frozen_lstm_input_grads(reward_model)
        policy.train(); total_loss = 0.0; total_mae = 0.0; total_n = 0
        for batch in train_loader:
            mask = batch["mask"].to(device); cloud = batch["cloud_features"].to(device); world = batch["world_features"].to(device); trend = batch["trend_features"].to(device)
            actual = batch["radiation_loss_raw"].to(device).float(); clear = batch["radiation_clear_wm2"].to(device).float().clamp_min(0.0)
            jitter = (torch.rand_like(actual) * 2.0 - 1.0) * args.target_jitter_wm2
            target = torch.minimum((actual + jitter).clamp(min=-100.0), clear)
            opt.zero_grad(set_to_none=True)
            mask_out, cloud_out = policy(mask, cloud, world, trend, target / rad_scale, actual / rad_scale)
            out = reward_forward(reward_model, mask_out, cloud_out, world, trend, context_scale)
            pred = out["cloud_radiation"] * rad_scale
            loss_target = F.smooth_l1_loss(pred, target, beta=50.0)
            loss = loss_target + args.lambda_anchor_mask * F.mse_loss(mask_out, mask) + args.lambda_anchor_cloud * F.mse_loss(cloud_out, cloud) + args.lambda_tv * total_variation(mask_out) + args.lambda_range * (F.relu(-cloud_out).pow(2).mean() + F.relu(cloud_out.abs() - 4.0).pow(2).mean())
            loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 3.0); opt.step()
            bs = mask.size(0); total_loss += float(loss.detach()) * bs; total_mae += float((pred.detach() - target).abs().mean()) * bs; total_n += bs
        policy.eval(); reward_model.eval(); val_mae = 0.0; val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                mask = batch["mask"].to(device); cloud = batch["cloud_features"].to(device); world = batch["world_features"].to(device); trend = batch["trend_features"].to(device)
                actual = batch["radiation_loss_raw"].to(device).float(); clear = batch["radiation_clear_wm2"].to(device).float().clamp_min(0.0)
                target = torch.minimum((actual + args.target_jitter_wm2 * 0.5).clamp(min=-100.0), clear)
                mask_out, cloud_out = policy(mask, cloud, world, trend, target / rad_scale, actual / rad_scale)
                pred = reward_forward(reward_model, mask_out, cloud_out, world, trend, context_scale)["cloud_radiation"] * rad_scale
                val_mae += float((pred - target).abs().mean()) * mask.size(0); val_n += mask.size(0)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_n), "train_target_mae_wm2": total_mae / max(1, total_n), "val_target_mae_wm2": val_mae / max(1, val_n)}
        history.append(row); print(f"epoch={epoch:03d} train_target_mae={row['train_target_mae_wm2']:.2f}W/m2 val_target_mae={row['val_target_mae_wm2']:.2f}W/m2")
        torch.save({"policy_state": policy.state_dict(), "args": vars(args), "cloud_names": cloud_names, "world_names": world_names}, out_dir / "last_policy.pt")
        if row["val_target_mae_wm2"] <= min(hh["val_target_mae_wm2"] for hh in history): torch.save({"policy_state": policy.state_dict(), "args": vars(args), "cloud_names": cloud_names, "world_names": world_names}, out_dir / "best_policy.pt")
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys())); writer.writeheader(); writer.writerows(history)
    batch = next(iter(val_loader))
    mask = batch["mask"].to(device); cloud = batch["cloud_features"].to(device); world = batch["world_features"].to(device); trend = batch["trend_features"].to(device)
    actual = batch["radiation_loss_raw"].to(device).float(); clear = batch["radiation_clear_wm2"].to(device).float().clamp_min(0.0)
    target = torch.minimum((actual + args.target_jitter_wm2 * 0.5).clamp(min=-100.0), clear)
    reward_model.eval()
    with torch.no_grad():
        mask_out, cloud_out = policy(mask, cloud, world, trend, target / rad_scale, actual / rad_scale)
        pred = reward_forward(reward_model, mask_out, cloud_out, world, trend, context_scale)["cloud_radiation"] * rad_scale
    save_strip(mask[0], out_dir / "example_original_sequence.png"); save_strip(mask_out[0], out_dir / "example_generated_sequence.png")
    summary = {"example_actual_loss_wm2": float(actual[0].item()), "example_target_loss_wm2": float(target[0].item()), "example_predicted_loss_wm2": float(pred[0].item()), "history_csv": str(out_dir / "history.csv"), "example_original_sequence": str(out_dir / "example_original_sequence.png"), "example_generated_sequence": str(out_dir / "example_generated_sequence.png")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE"); print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
