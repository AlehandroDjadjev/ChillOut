#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def import_v8_reward():
    names = [
        "train_cloud_radiation_bottom_v8_CLEAN_DIRECT",
        "train_cloud_radiation_bottom_v8_clean_direct",
    ]
    last = None
    for name in names:
        try:
            return __import__(name, fromlist=["dummy"])
        except Exception as exc:
            last = exc
    raise RuntimeError(
        "Could not import v8 bottom model. Put this file next to "
        "train_cloud_radiation_bottom_v8_CLEAN_DIRECT.py. Last error: "
        f"{last}"
    )


def strip_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def enable_frozen_lstm_input_grads(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            m.train()


def reward_forward(reward_model, image, cloud, context, clear):
    if image.is_cuda:
        with torch.backends.cudnn.flags(enabled=False):
            return reward_model(image, cloud, context, clear)
    return reward_model(image, cloud, context, clear)


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_sequence_strip(seq: torch.Tensor, path: Path, channel: int = 0) -> None:
    arr = seq.detach().float().cpu().clamp(0, 1).numpy()
    channel = min(channel, arr.shape[1] - 1)
    imgs = []
    for i in range(arr.shape[0]):
        im = Image.fromarray((arr[i, channel] * 255).astype(np.uint8), mode="L")
        im = ImageOps.expand(im, border=2, fill=128)
        imgs.append(im)
    out = Image.new("L", (sum(i.width for i in imgs), max(i.height for i in imgs)), 0)
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def make_clean_args(ckargs: Dict[str, Any], override_min_clear: float | None = None):
    class A:
        pass
    a = A()
    a.min_clear_wm2 = float(override_min_clear if override_min_clear is not None else ckargs.get("min_clear_wm2", 120.0))
    a.clean_drop_invalid = bool(ckargs.get("clean_drop_invalid", True))
    a.clean_drop_negative = bool(ckargs.get("clean_drop_negative", True))
    a.clean_drop_high_cloud_low_loss = bool(ckargs.get("clean_drop_high_cloud_low_loss", True))
    a.clean_drop_low_cloud_high_loss = bool(ckargs.get("clean_drop_low_cloud_high_loss", False))
    a.high_cloud_thresh = float(ckargs.get("high_cloud_thresh", 0.65))
    a.low_cloud_thresh = float(ckargs.get("low_cloud_thresh", 0.05))
    a.low_loss_thresh_wm2 = float(ckargs.get("low_loss_thresh_wm2", 25.0))
    a.high_loss_thresh_wm2 = float(ckargs.get("high_loss_thresh_wm2", 280.0))
    a.min_loss_wm2 = float(ckargs.get("min_loss_wm2", 0.0))
    a.max_loss_wm2 = float(ckargs.get("max_loss_wm2", 900.0))
    return a


def get_cloud_raw_stats(v8, x_norm, raw_names: List[str], cloud_names: List[str], device):
    raw_to_idx = {n: i for i, n in enumerate(raw_names)}
    idx = [raw_to_idx[n] for n in cloud_names]
    mean = torch.tensor(x_norm.mean[idx], dtype=torch.float32, device=device).view(1, 1, -1)
    std = torch.tensor(x_norm.std[idx], dtype=torch.float32, device=device).view(1, 1, -1).clamp_min(1e-6)
    return mean, std


def target_schedule(actual, clear, args, epoch: int):
    """Wide target curriculum for generation from zero/messed-up inputs."""
    progress = min(1.0, max(0.0, epoch / max(1, args.curriculum_epochs)))
    jitter = args.target_jitter_start_wm2 + progress * (args.target_jitter_wm2 - args.target_jitter_start_wm2)
    raw = (torch.rand_like(actual) * 2.0 - 1.0) * jitter
    target = actual + raw

    if args.prefer_positive_targets > 0:
        pos = (torch.rand_like(actual) < args.prefer_positive_targets).float()
        target = pos * (actual + torch.rand_like(actual) * jitter) + (1.0 - pos) * target

    if args.extra_random_absolute_targets > 0:
        use_abs = (torch.rand_like(actual) < args.extra_random_absolute_targets).float()
        abs_target = torch.rand_like(actual) * clear.clamp_min(1.0)
        target = use_abs * abs_target + (1.0 - use_abs) * target

    return torch.minimum(target.clamp(min=args.min_target_wm2), clear.clamp_min(0.0))


def fixed_val_target(actual, clear, args):
    target = actual + args.val_target_delta_wm2
    return torch.minimum(target.clamp(min=args.min_target_wm2), clear.clamp_min(0.0))


def corruption_keep_ratio(args, epoch: int) -> float:
    progress = min(1.0, max(0.0, epoch / max(1, args.corruption_curriculum_epochs)))
    return args.keep_ratio_start + progress * (args.keep_ratio_end - args.keep_ratio_start)


def make_block_keep_mask(b: int, t: int, h: int, w: int, keep_ratio: float, block: int, device) -> torch.Tensor:
    block = max(1, int(block))
    gh = int(math.ceil(h / block))
    gw = int(math.ceil(w / block))
    small = (torch.rand(b, t, 1, gh, gw, device=device) < keep_ratio).float()
    mask = F.interpolate(small.reshape(b * t, 1, gh, gw), size=(h, w), mode="nearest")
    return mask.view(b, t, 1, h, w)


def corrupt_cloud_input(image: torch.Tensor, args, epoch: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Make model train from zero / mostly-zero / mixed image inputs."""
    b, t, c, h, w = image.shape
    device = image.device
    keep_ratio = corruption_keep_ratio(args, epoch)

    keep = make_block_keep_mask(b, t, h, w, keep_ratio, args.corrupt_block_size, device)
    if args.degrade_frames == "last":
        frame_gate = torch.zeros(b, t, 1, h, w, device=device)
        frame_gate[:, -1] = 1.0
        keep = keep * frame_gate + (1.0 - frame_gate)
    elif args.degrade_frames == "random_last_or_all":
        if random.random() < 0.5:
            frame_gate = torch.zeros(b, t, 1, h, w, device=device)
            frame_gate[:, -1] = 1.0
            keep = keep * frame_gate + (1.0 - frame_gate)

    noise = torch.rand_like(image) * args.noise_fill_ratio
    black = torch.full_like(image, args.black_value)

    # Forced randomness: sometimes start from pure black/noise even where keep says original.
    mode_rand = random.random()
    if mode_rand < args.zero_start_prob:
        keep = torch.zeros_like(keep)
        fill = black + noise
        mode = "zero_noise"
    elif mode_rand < args.zero_start_prob + args.noise_start_prob:
        keep = torch.zeros_like(keep)
        fill = noise
        mode = "noise_only"
    else:
        fill = black + noise
        mode = "mixed_blocks"

    mixed = image * keep.expand_as(image) + fill * (1.0 - keep.expand_as(image))
    return mixed.clamp(0.0, 1.0), keep, {"keep_ratio": float(keep_ratio), "corrupt_mode": mode}


def random_blob_latent(
    b: int,
    t: int,
    h: int,
    w: int,
    channels: int,
    num_blobs: int,
    min_radius: float,
    max_radius: float,
    device,
) -> torch.Tensor:
    """Soft random blob maps used as spatial latent. This pushes clustered structures."""
    if channels <= 0:
        return torch.zeros(b, t, 0, h, w, device=device)

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, h, device=device),
        torch.linspace(0.0, 1.0, w, device=device),
        indexing="ij",
    )
    xx = xx.view(1, 1, 1, h, w)
    yy = yy.view(1, 1, 1, h, w)

    outs = []
    for _ in range(channels):
        acc = torch.zeros(b, t, 1, h, w, device=device)
        for _j in range(num_blobs):
            cx = torch.rand(b, t, 1, 1, 1, device=device)
            cy = torch.rand(b, t, 1, 1, 1, device=device)
            r = min_radius + (max_radius - min_radius) * torch.rand(b, t, 1, 1, 1, device=device)
            amp = torch.rand(b, t, 1, 1, 1, device=device)
            d2 = (xx - cx).pow(2) + (yy - cy).pow(2)
            acc = acc + amp * torch.exp(-d2 / (2.0 * r.pow(2).clamp_min(1e-4)))
        acc = acc / acc.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        outs.append(acc)
    return torch.cat(outs, dim=2).clamp(0.0, 1.0)


class ConvBlock(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(min(8, b), b), nn.SiLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(min(8, b), b), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CloudGeneratorV9ZeroBlobs(nn.Module):
    """Full 4-frame cloud generator.

    Unlike the old controller, this outputs the full T*C image tensor from a
    corrupted/zero/mixed image input. It is conditioned on target radiation and
    fixed context. Blob latent maps give it clustered spatial randomness.
    """

    def __init__(
        self,
        frames: int,
        in_channels: int,
        context_dim: int = 8,
        blob_channels: int = 3,
        base: int = 48,
        cond_dim: int = 32,
        loss_scale: float = 300.0,
    ):
        super().__init__()
        self.frames = int(frames)
        self.in_channels = int(in_channels)
        self.blob_channels = int(blob_channels)
        self.loss_scale = float(loss_scale)

        self.cond = nn.Sequential(
            nn.Linear(context_dim + 4, 128), nn.LayerNorm(128), nn.SiLU(inplace=True),
            nn.Linear(128, cond_dim), nn.SiLU(inplace=True),
        )

        input_ch = frames * in_channels + frames * blob_channels + cond_dim
        out_ch = frames * in_channels

        self.e1 = ConvBlock(input_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 4)
        self.u2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.u1 = ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.out.weight, 0.0, 1e-3)

    def forward(self, corrupted, context, clear_wm2, target_wm2, actual_wm2, blob_latent):
        b, t, c, h, w = corrupted.shape
        x_img = corrupted.reshape(b, t * c, h, w)
        x_blob = blob_latent.reshape(b, t * self.blob_channels, h, w) if self.blob_channels > 0 else corrupted.new_zeros(b, 0, h, w)

        target_norm = target_wm2 / self.loss_scale
        actual_norm = actual_wm2 / self.loss_scale
        clear_norm = clear_wm2 / self.loss_scale
        delta_norm = (target_wm2 - actual_wm2) / self.loss_scale
        cond_vec = torch.cat([context, target_norm, actual_norm, delta_norm, clear_norm], dim=-1)
        z = self.cond(cond_vec).view(b, -1, 1, 1).expand(-1, -1, h, w)

        x = torch.cat([x_img, x_blob, z], dim=1)
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        m = self.mid(e3)
        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.u1(torch.cat([u1, e1], dim=1))

        logits = self.out(u1).view(b, t, c, h, w)
        generated = torch.sigmoid(logits)
        return generated


def edge_density(x: torch.Tensor) -> torch.Tensor:
    # x [B,T,H,W]
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(-1, -2))
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(-1, -2))
    return (dx + dy).clamp(0.0, 1.0)


def derive_cloud_features_from_image(
    image: torch.Tensor,
    cloud_base: torch.Tensor,
    cloud_names: List[str],
    raw_mean: torch.Tensor,
    raw_std: torch.Tensor,
) -> torch.Tensor:
    """Derive editable cloud scalar features from generated image.

    This prevents the policy from setting impossible scalar values independently
    of the generated cloud tensor.
    """
    b, t, c, h, w = image.shape
    raw = cloud_base * raw_std + raw_mean
    name_to_i = {n: i for i, n in enumerate(cloud_names)}

    ch0 = image[:, :, 0]
    ch1 = image[:, :, min(1, c - 1)]
    ch2 = image[:, :, min(2, c - 1)]
    ch3 = image[:, :, min(3, c - 1)]
    ch4 = image[:, :, min(4, c - 1)]
    ch5 = image[:, :, min(5, c - 1)]
    ch6 = image[:, :, min(6, c - 1)]
    ch7 = image[:, :, min(7, c - 1)]

    flat1 = ch1.flatten(2)
    quant90 = torch.quantile(flat1, 0.90, dim=2)

    vals = {
        "cloud_s2_fraction": ch0.mean(dim=(-1, -2)),
        "cloud_s2_prob_mean": ch1.mean(dim=(-1, -2)),
        "cloud_s2_prob_std": ch1.std(dim=(-1, -2)).clamp(0, 1),
        "cloud_s2_prob_p90": quant90.clamp(0, 1),
        "cloud_s2_cirrus_fraction": ch3.mean(dim=(-1, -2)),
        "cloud_s2_high_fraction": ch4.mean(dim=(-1, -2)),
        "cloud_s2_medium_fraction": ch5.mean(dim=(-1, -2)),
        "cloud_s2_aot_mean": ch6.mean(dim=(-1, -2)),
        "cloud_s2_texture_std": ch7.std(dim=(-1, -2)).clamp(0, 1),
        "cloud_s2_edge_density": edge_density(ch0),
    }

    for name, value in vals.items():
        if name in name_to_i:
            raw[:, :, name_to_i[name]] = value.clamp(0.0, 1.0)

    return (raw - raw_mean) / raw_std


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (
        (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().mean()
        + (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().mean()
    )


def temporal_smoothness(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) <= 1:
        return x.new_tensor(0.0)
    return (x[:, 1:] - x[:, :-1]).abs().mean()


def highfreq_loss(x: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    kernel = max(3, int(kernel) | 1)
    b, t, c, h, w = x.shape
    y = x.reshape(b * t * c, 1, h, w)
    smooth = F.avg_pool2d(y, kernel_size=kernel, stride=1, padding=kernel // 2)
    return (y - smooth).abs().mean()


def channel_consistency_loss(x: torch.Tensor) -> torch.Tensor:
    # Encourage cloud mask/prob/white/high/medium channels to be mutually coherent.
    if x.size(2) < 2:
        return x.new_tensor(0.0)
    prob = x[:, :, 1]
    mask = x[:, :, 0]
    loss = F.smooth_l1_loss(mask, prob, beta=0.1)
    if x.size(2) > 2:
        # subtype channels should usually not exceed probability too much
        for ch in [2, 3, 4, 5]:
            if ch < x.size(2):
                loss = loss + F.relu(x[:, :, ch] - prob).mean()
    return loss


def evaluate_generator(
    reward_model,
    generator,
    loader,
    device,
    args,
    epoch,
    loss_scale,
    raw_mean,
    raw_std,
    cloud_names,
):
    generator.eval()
    enable_frozen_lstm_input_grads(reward_model)
    total = {"mae": 0.0, "base_mae": 0.0, "mixed_mae": 0.0, "recon_l1": 0.0, "n": 0}
    rows = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            cloud = batch["cloud_features"].to(device)
            context = batch["context_features"].to(device)
            clear = batch["clear_wm2"].to(device).float()
            actual = batch["target_loss_wm2"].to(device).float()
            target = fixed_val_target(actual, clear, args)

            corrupted, _keep, _info = corrupt_cloud_input(image, args, epoch)
            blob = random_blob_latent(
                image.size(0), image.size(1), image.size(-2), image.size(-1),
                args.blob_channels, args.num_blobs, args.min_blob_radius, args.max_blob_radius,
                device,
            )
            generated = generator(corrupted, context, clear, target, actual, blob)
            cloud_gen = derive_cloud_features_from_image(generated, cloud, cloud_names, raw_mean, raw_std)

            base_pred = reward_forward(reward_model, image, cloud, context, clear)["loss_wm2"]
            mixed_pred = reward_forward(reward_model, corrupted, cloud, context, clear)["loss_wm2"]
            pred = reward_forward(reward_model, generated, cloud_gen, context, clear)["loss_wm2"]

            bs = image.size(0)
            total["mae"] += float((pred - target).abs().mean()) * bs
            total["base_mae"] += float((base_pred - target).abs().mean()) * bs
            total["mixed_mae"] += float((mixed_pred - target).abs().mean()) * bs
            total["recon_l1"] += float((generated - image).abs().mean()) * bs
            total["n"] += bs

            for i in range(bs):
                rows.append({
                    "sample_id": batch["sample_id"][i],
                    "location": batch["location"][i],
                    "anchor": batch["anchor"][i],
                    "actual_loss_wm2": float(actual[i].item()),
                    "target_loss_wm2": float(target[i].item()),
                    "base_pred_loss_wm2": float(base_pred[i].item()),
                    "mixed_pred_loss_wm2": float(mixed_pred[i].item()),
                    "generated_pred_loss_wm2": float(pred[i].item()),
                    "abs_error_wm2": abs(float(pred[i].item() - target[i].item())),
                })

    n = max(1, total["n"])
    return {
        "val_target_mae_wm2": total["mae"] / n,
        "val_base_mae_wm2": total["base_mae"] / n,
        "val_mixed_mae_wm2": total["mixed_mae"] / n,
        "val_improvement_vs_mixed_wm2": total["mixed_mae"] / n - total["mae"] / n,
        "val_recon_l1": total["recon_l1"] / n,
    }, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True, help="v8 reward checkpoint")
    ap.add_argument("--out-dir", default="runs/cloud_radiation_generator_v9_zero_blobs")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=8e-4)
    ap.add_argument("--base-channels", type=int, default=48)
    ap.add_argument("--cond-dim", type=int, default=32)

    # Corruption / from-zero training.
    ap.add_argument("--keep-ratio-start", type=float, default=0.60)
    ap.add_argument("--keep-ratio-end", type=float, default=0.10)
    ap.add_argument("--corruption-curriculum-epochs", type=int, default=35)
    ap.add_argument("--corrupt-block-size", type=int, default=14)
    ap.add_argument("--noise-fill-ratio", type=float, default=0.08)
    ap.add_argument("--black-value", type=float, default=0.0)
    ap.add_argument("--zero-start-prob", type=float, default=0.35)
    ap.add_argument("--noise-start-prob", type=float, default=0.20)
    ap.add_argument("--degrade-frames", choices=["last", "all", "random_last_or_all"], default="all")

    # Random blob latent.
    ap.add_argument("--blob-channels", type=int, default=3)
    ap.add_argument("--num-blobs", type=int, default=9)
    ap.add_argument("--min-blob-radius", type=float, default=0.04)
    ap.add_argument("--max-blob-radius", type=float, default=0.18)

    # Target curriculum.
    ap.add_argument("--target-jitter-start-wm2", type=float, default=80.0)
    ap.add_argument("--target-jitter-wm2", type=float, default=320.0)
    ap.add_argument("--curriculum-epochs", type=int, default=35)
    ap.add_argument("--val-target-delta-wm2", type=float, default=180.0)
    ap.add_argument("--prefer-positive-targets", type=float, default=0.70)
    ap.add_argument("--extra-random-absolute-targets", type=float, default=0.30)
    ap.add_argument("--min-target-wm2", type=float, default=0.0)
    ap.add_argument("--beta-wm2", type=float, default=45.0)

    # Losses.
    ap.add_argument("--lambda-recon-start", type=float, default=2.0)
    ap.add_argument("--lambda-recon-floor", type=float, default=0.25)
    ap.add_argument("--recon-decay-epochs", type=int, default=45)
    ap.add_argument("--lambda-reward", type=float, default=1.0)
    ap.add_argument("--lambda-tv", type=float, default=0.12)
    ap.add_argument("--lambda-temporal", type=float, default=0.20)
    ap.add_argument("--lambda-highfreq", type=float, default=0.18)
    ap.add_argument("--highfreq-kernel", type=int, default=11)
    ap.add_argument("--lambda-channel-consistency", type=float, default=0.40)

    ap.add_argument("--min-clear-wm2", type=float, default=None)
    ap.add_argument("--early-stop-patience", type=int, default=16)
    ap.add_argument("--save-examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    v8 = import_v8_reward()
    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if ckpt.get("architecture") != "CloudRadiationV8CleanDirect":
        print(f"WARNING checkpoint architecture={ckpt.get('architecture')} expected CloudRadiationV8CleanDirect")

    raw_names = list(ckpt["raw_feature_names"])
    cloud_names = list(ckpt["cloud_feature_names"])
    x_norm = v8.base.Normalizer.from_state_dict(ckpt["normalizer"])
    ckargs = ckpt.get("args", {})

    cleaner = make_clean_args(ckargs, override_min_clear=args.min_clear_wm2)
    train_records = v8.clean_radiation_records(v8.base.load_records(root, "train"), cleaner, "train")
    val_records = v8.clean_radiation_records(v8.base.load_records(root, "val"), cleaner, "val")

    ds_kwargs = dict(
        root=root,
        raw_names=raw_names,
        cloud_names=cloud_names,
        x_norm=x_norm,
        image_height=int(ckpt.get("image_height", 96)),
        image_width=int(ckpt.get("image_width", 96)),
        lookback=int(ckpt.get("lookback", 4)),
        max_gap_days=float(ckargs.get("max_gap_days", 12.0)),
        use_cloud_tensor=bool(ckargs.get("use_cloud_tensor", True)),
        augment=False,
        min_clear_wm2=float(cleaner.min_clear_wm2),
    )
    train_ds = v8.RadiationSequenceDataset(records=train_records, **ds_kwargs)
    val_ds = v8.RadiationSequenceDataset(records=val_records, **ds_kwargs)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    reward_model = v8.CloudRadiationV8CleanDirect(**ckpt["model_kwargs"]).to(device)
    reward_model.load_state_dict(strip_prefix(ckpt["model_state"]))
    enable_frozen_lstm_input_grads(reward_model)

    sample = train_ds[0]
    frames = int(sample["image"].shape[0])
    in_channels = int(sample["image"].shape[1])
    loss_scale = float(ckpt.get("model_kwargs", {}).get("loss_scale", ckargs.get("loss_scale", 300.0)))

    generator = CloudGeneratorV9ZeroBlobs(
        frames=frames,
        in_channels=in_channels,
        context_dim=8,
        blob_channels=args.blob_channels,
        base=args.base_channels,
        cond_dim=args.cond_dim,
        loss_scale=loss_scale,
    ).to(device)

    opt = torch.optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=6)

    raw_mean, raw_std = get_cloud_raw_stats(v8, x_norm, raw_names, cloud_names, device)

    config = {
        "trainer": "CloudGeneratorV9ZeroBlobs",
        "reward_checkpoint": str(args.checkpoint),
        "device": str(device),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "frames": frames,
        "in_channels": in_channels,
        "zero_start_prob": args.zero_start_prob,
        "noise_start_prob": args.noise_start_prob,
        "keep_ratio_start": args.keep_ratio_start,
        "keep_ratio_end": args.keep_ratio_end,
        "blob_channels": args.blob_channels,
        "num_blobs": args.num_blobs,
        "min_blob_radius": args.min_blob_radius,
        "max_blob_radius": args.max_blob_radius,
        "target_jitter_wm2": args.target_jitter_wm2,
        "scalar_mode": "derived_from_generated_image",
    }
    print(json.dumps(config, indent=2))

    history = []
    best = float("inf")
    bad = 0

    for epoch in range(1, args.epochs + 1):
        enable_frozen_lstm_input_grads(reward_model)
        generator.train()
        totals = {"loss": 0.0, "mae": 0.0, "mixed_mae": 0.0, "recon": 0.0, "n": 0}

        recon_progress = min(1.0, epoch / max(1, args.recon_decay_epochs))
        lambda_recon = args.lambda_recon_floor + (1.0 - recon_progress) * (args.lambda_recon_start - args.lambda_recon_floor)

        for batch in train_loader:
            image = batch["image"].to(device)
            cloud = batch["cloud_features"].to(device)
            context = batch["context_features"].to(device)
            clear = batch["clear_wm2"].to(device).float()
            actual = batch["target_loss_wm2"].to(device).float()
            target = target_schedule(actual, clear, args, epoch)

            corrupted, _keep, _corr_info = corrupt_cloud_input(image, args, epoch)
            blob = random_blob_latent(
                image.size(0), image.size(1), image.size(-2), image.size(-1),
                args.blob_channels, args.num_blobs, args.min_blob_radius, args.max_blob_radius,
                device,
            )

            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                mixed_pred = reward_forward(reward_model, corrupted, cloud, context, clear)["loss_wm2"]

            generated = generator(corrupted, context, clear, target, actual, blob)
            cloud_gen = derive_cloud_features_from_image(generated, cloud, cloud_names, raw_mean, raw_std)
            pred = reward_forward(reward_model, generated, cloud_gen, context, clear)["loss_wm2"]

            reward_loss = F.smooth_l1_loss(pred, target, beta=args.beta_wm2)
            recon_loss = F.smooth_l1_loss(generated, image, beta=0.08)
            tv = total_variation(generated)
            temp = temporal_smoothness(generated)
            hf = highfreq_loss(generated, args.highfreq_kernel)
            ch_cons = channel_consistency_loss(generated)

            loss = (
                args.lambda_reward * reward_loss
                + lambda_recon * recon_loss
                + args.lambda_tv * tv
                + args.lambda_temporal * temp
                + args.lambda_highfreq * hf
                + args.lambda_channel_consistency * ch_cons
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 2.0)
            opt.step()

            bs = image.size(0)
            totals["loss"] += float(loss.detach()) * bs
            totals["mae"] += float((pred.detach() - target).abs().mean()) * bs
            totals["mixed_mae"] += float((mixed_pred.detach() - target).abs().mean()) * bs
            totals["recon"] += float((generated.detach() - image).abs().mean()) * bs
            totals["n"] += bs

        val, val_rows = evaluate_generator(reward_model, generator, val_loader, device, args, epoch, loss_scale, raw_mean, raw_std, cloud_names)
        scheduler.step(val["val_target_mae_wm2"])

        n = max(1, totals["n"])
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / n,
            "train_target_mae_wm2": totals["mae"] / n,
            "train_mixed_mae_wm2": totals["mixed_mae"] / n,
            "train_improvement_vs_mixed_wm2": totals["mixed_mae"] / n - totals["mae"] / n,
            "train_recon_l1": totals["recon"] / n,
            "lambda_recon": lambda_recon,
            "keep_ratio": corruption_keep_ratio(args, epoch),
            **val,
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_mae={row['train_target_mae_wm2']:.2f} "
            f"train_mixed={row['train_mixed_mae_wm2']:.2f} "
            f"val_mae={row['val_target_mae_wm2']:.2f} "
            f"val_mixed={row['val_mixed_mae_wm2']:.2f} "
            f"val_improve={row['val_improvement_vs_mixed_wm2']:.2f} "
            f"recon={row['val_recon_l1']:.4f} "
            f"keep={row['keep_ratio']:.2f} "
            f"lambda_recon={row['lambda_recon']:.2f} "
            f"lr={row['lr']:.2e}"
        )

        write_csv(out_dir / "history.csv", history)
        write_csv(out_dir / "val_predictions_latest.csv", val_rows)

        payload = {
            "architecture": "CloudGeneratorV9ZeroBlobs",
            "model_state": generator.state_dict(),
            "args": vars(args),
            "reward_checkpoint": str(args.checkpoint),
            "cloud_names": cloud_names,
            "raw_feature_names": raw_names,
            "frames": frames,
            "in_channels": in_channels,
            "loss_scale": loss_scale,
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(payload, out_dir / "last_generator.pt")

        if row["val_target_mae_wm2"] < best:
            best = row["val_target_mae_wm2"]
            bad = 0
            torch.save(payload, out_dir / "best_generator.pt")
            write_csv(out_dir / "val_predictions_best.csv", val_rows)
            print(f"saved best_generator.pt val_mae={best:.2f}W/m2")
        else:
            bad += 1

        if args.early_stop_patience > 0 and bad >= args.early_stop_patience:
            print(f"early stop: {bad} epochs without val improvement")
            break

    # Save visual examples.
    generator.eval()
    batch = next(iter(val_loader))
    image = batch["image"].to(device)
    cloud = batch["cloud_features"].to(device)
    context = batch["context_features"].to(device)
    clear = batch["clear_wm2"].to(device).float()
    actual = batch["target_loss_wm2"].to(device).float()
    target = fixed_val_target(actual, clear, args)

    with torch.no_grad():
        corrupted, _keep, _info = corrupt_cloud_input(image, args, args.epochs)
        blob = random_blob_latent(
            image.size(0), image.size(1), image.size(-2), image.size(-1),
            args.blob_channels, args.num_blobs, args.min_blob_radius, args.max_blob_radius,
            device,
        )
        generated = generator(corrupted, context, clear, target, actual, blob)
        cloud_gen = derive_cloud_features_from_image(generated, cloud, cloud_names, raw_mean, raw_std)
        base_pred = reward_forward(reward_model, image, cloud, context, clear)["loss_wm2"]
        mixed_pred = reward_forward(reward_model, corrupted, cloud, context, clear)["loss_wm2"]
        pred = reward_forward(reward_model, generated, cloud_gen, context, clear)["loss_wm2"]

    examples = []
    for i in range(min(args.save_examples, image.size(0))):
        prefix = out_dir / "examples" / f"sample_{i:02d}"
        for ch in [0, 1, 2, 4, 5, 7]:
            if ch < image.size(2):
                save_sequence_strip(image[i], prefix.with_name(prefix.name + f"_original_ch{ch}.png"), ch)
                save_sequence_strip(corrupted[i], prefix.with_name(prefix.name + f"_corrupted_ch{ch}.png"), ch)
                save_sequence_strip(generated[i], prefix.with_name(prefix.name + f"_generated_ch{ch}.png"), ch)
        examples.append({
            "i": i,
            "sample_id": batch["sample_id"][i],
            "location": batch["location"][i],
            "anchor": batch["anchor"][i],
            "actual_loss_wm2": float(actual[i].item()),
            "target_loss_wm2": float(target[i].item()),
            "base_pred_wm2": float(base_pred[i].item()),
            "mixed_pred_wm2": float(mixed_pred[i].item()),
            "generated_pred_wm2": float(pred[i].item()),
        })

    summary = {
        "best_val_target_mae_wm2": best,
        "history_csv": str(out_dir / "history.csv"),
        "best_generator": str(out_dir / "best_generator.pt"),
        "last_generator": str(out_dir / "last_generator.pt"),
        "examples": examples,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
