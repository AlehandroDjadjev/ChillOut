#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


MODE_NAMES = [
    "keep",
    "replace",
    "alpha_0p50",
    "alpha_0p75",
    "max_add_cloud",
    "min_remove_cloud",
    "soft_add_cloud",
]


# ---------------------------------------------------------------------
# Imports only v8 bottom reward. This is intentionally not based on v9/v11.
# ---------------------------------------------------------------------

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
        "Could not import train_cloud_radiation_bottom_v8_CLEAN_DIRECT.py. "
        "Put this file next to the v8 bottom model script. Last error: "
        f"{last}"
    )


def strip_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def enable_frozen_reward_grads(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # cuDNN RNN backward through frozen LSTM can fail unless LSTM is in train mode.
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            m.train()


def reward_forward(reward_model, image, cloud, context, clear):
    if image.is_cuda:
        with torch.backends.cudnn.flags(enabled=False):
            return reward_model(image, cloud, context, clear)
    return reward_model(image, cloud, context, clear)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def parse_channels(s: str, max_c: int) -> List[int]:
    if str(s).strip().lower() == "all":
        return list(range(max_c))
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        i = int(x)
        if 0 <= i < max_c:
            out.append(i)
    return out


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


# ---------------------------------------------------------------------
# v8-compatible dataset / cloud feature helpers.
# ---------------------------------------------------------------------

def make_clean_args(ckargs: Dict[str, Any], override_min_clear: Optional[float] = None):
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


def edge_density(x: torch.Tensor) -> torch.Tensor:
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
    """Derive cloud scalar features from output image.

    This prevents scalar cheating. The selector can only affect cloud scalars
    through the chosen/generated image tensor.
    """
    b, t, c, h, w = image.shape
    raw = cloud_base * raw_std + raw_mean
    name_to_i = {n: i for i, n in enumerate(cloud_names)}

    ch0 = image[:, :, 0]
    ch1 = image[:, :, min(1, c - 1)]
    ch3 = image[:, :, min(3, c - 1)]
    ch4 = image[:, :, min(4, c - 1)]
    ch5 = image[:, :, min(5, c - 1)]
    ch6 = image[:, :, min(6, c - 1)]
    ch7 = image[:, :, min(7, c - 1)]

    flat1 = ch1.flatten(2)
    try:
        q90 = torch.quantile(flat1, 0.90, dim=2)
    except Exception:
        q90 = flat1.topk(max(1, int(flat1.size(2) * 0.10)), dim=2).values[:, :, -1]

    vals = {
        "cloud_s2_fraction": ch0.mean(dim=(-1, -2)),
        "cloud_s2_prob_mean": ch1.mean(dim=(-1, -2)),
        "cloud_s2_prob_std": ch1.std(dim=(-1, -2)).clamp(0, 1),
        "cloud_s2_prob_p90": q90.clamp(0, 1),
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


# ---------------------------------------------------------------------
# Targets and input corruption.
# ---------------------------------------------------------------------

def target_schedule(actual: torch.Tensor, clear: torch.Tensor, args, epoch: int) -> torch.Tensor:
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


def fixed_val_target(actual: torch.Tensor, clear: torch.Tensor, args) -> torch.Tensor:
    target = actual + args.val_target_delta_wm2
    return torch.minimum(target.clamp(min=args.min_target_wm2), clear.clamp_min(0.0))


def white_dropout_input(image: torch.Tensor, args):
    """Only remove existing white/cloud pixels. No zero/noise starts."""
    out = image.clone()
    b, t, c, h, w = image.shape
    chans = parse_channels(args.white_drop_channels, c)

    total_white = 0.0
    total_dropped = 0.0
    dropped_mask = torch.zeros(b, t, 1, h, w, device=image.device)

    for ch in chans:
        white = out[:, :, ch:ch+1] >= args.white_threshold
        drop = (torch.rand_like(out[:, :, ch:ch+1]) < args.white_drop_prob) & white
        out[:, :, ch:ch+1] = torch.where(drop, torch.zeros_like(out[:, :, ch:ch+1]), out[:, :, ch:ch+1])
        dropped_mask = torch.maximum(dropped_mask, drop.float())
        total_white += float(white.float().sum().detach().cpu())
        total_dropped += float(drop.float().sum().detach().cpu())

    return out.clamp(0.0, 1.0), dropped_mask, {
        "mode": "white_dropout_only",
        "actual_drop_rate_on_white": float(total_dropped / max(1.0, total_white)),
    }


def maybe_make_input(image: torch.Tensor, args):
    if random.random() < args.full_input_prob:
        dropped = torch.zeros(image.size(0), image.size(1), 1, image.size(-2), image.size(-1), device=image.device)
        return image.clone(), dropped, {"mode": "full_input"}
    return white_dropout_input(image, args)


# ---------------------------------------------------------------------
# Codebook.
# ---------------------------------------------------------------------

def coverage_of_image(x: torch.Tensor, channels: List[int]) -> float:
    vals = []
    for ch in channels:
        if ch < x.size(1):
            vals.append(float(x[:, ch].mean().item()))
    return float(np.mean(vals)) if vals else 0.0


def build_codebook(train_ds, args, device) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Build a real-cloud template codebook from training samples.

    Codebook shape: [K,T,C,H,W].
    Stratified by cloud coverage so the selector has clear, partial, and cloudy options.
    """
    cov_channels = None
    images = []
    metas = []

    for i in range(len(train_ds)):
        item = train_ds[i]
        img = item["image"].float()  # [T,C,H,W]
        if cov_channels is None:
            cov_channels = parse_channels(args.codebook_coverage_channels, img.size(1))
        cov = coverage_of_image(img, cov_channels)
        if cov < args.codebook_min_coverage or cov > args.codebook_max_coverage:
            continue
        images.append(img)
        metas.append({
            "dataset_index": i,
            "sample_id": str(item["sample_id"]),
            "location": str(item["location"]),
            "anchor": str(item["anchor"]),
            "coverage": cov,
        })

    if not images:
        raise RuntimeError("No images available for codebook after coverage filtering.")

    # Stratified sample by coverage bins.
    bins = [[] for _ in range(args.codebook_bins)]
    for idx, meta in enumerate(metas):
        b = min(args.codebook_bins - 1, max(0, int(meta["coverage"] * args.codebook_bins)))
        bins[b].append(idx)

    rng = random.Random(args.seed)
    selected_idx = []
    per_bin = max(1, math.ceil(args.codebook_size / args.codebook_bins))
    for b in bins:
        rng.shuffle(b)
        selected_idx.extend(b[:per_bin])

    if len(selected_idx) < args.codebook_size:
        remaining = [i for i in range(len(images)) if i not in set(selected_idx)]
        rng.shuffle(remaining)
        selected_idx.extend(remaining[:args.codebook_size - len(selected_idx)])

    selected_idx = selected_idx[:args.codebook_size]
    selected_images = torch.stack([images[i] for i in selected_idx], 0).to(device)
    selected_metas = [metas[i] for i in selected_idx]

    return selected_images.contiguous(), selected_metas


# ---------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------

class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, base: int = 32, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(min(8, base), base),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, base * 2), base * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, base * 4), base * 4),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(base * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = image.shape
        x = image.reshape(b, t * c, h, w)
        y = self.net(x).flatten(1)
        return self.proj(y)


class CloudTemplateSelectorV13(nn.Module):
    """Simple trainable controller.

    It does not paint pixels.
    It selects one real cloud template from a codebook and one edit mode.

    Output is built from:
      input image + selected real cloud template + selected mode
    """

    def __init__(
        self,
        frames: int,
        in_channels: int,
        codebook_size: int,
        num_modes: int,
        context_dim: int = 8,
        base: int = 32,
        hidden: int = 192,
        loss_scale: float = 300.0,
    ):
        super().__init__()
        self.loss_scale = float(loss_scale)
        self.encoder = ConvEncoder(frames * in_channels, base=base, out_dim=hidden)
        self.cond = nn.Sequential(
            nn.Linear(context_dim + 8, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
        )
        self.template_head = nn.Linear(hidden, codebook_size)
        self.mode_head = nn.Linear(hidden, num_modes)

    def forward(
        self,
        image_in: torch.Tensor,
        context: torch.Tensor,
        clear_wm2: torch.Tensor,
        actual_wm2: torch.Tensor,
        target_wm2: torch.Tensor,
        input_pred_wm2: torch.Tensor,
        input_coverage: torch.Tensor,
    ):
        target_norm = target_wm2 / self.loss_scale
        actual_norm = actual_wm2 / self.loss_scale
        input_pred_norm = input_pred_wm2 / self.loss_scale
        clear_norm = clear_wm2 / self.loss_scale
        delta_actual = (target_wm2 - actual_wm2) / self.loss_scale
        delta_input = (target_wm2 - input_pred_wm2) / self.loss_scale
        abs_delta_input = delta_input.abs()
        cond = torch.cat([
            context,
            target_norm,
            actual_norm,
            input_pred_norm,
            clear_norm,
            delta_actual,
            delta_input,
            abs_delta_input,
            input_coverage,
        ], dim=-1)

        z_img = self.encoder(image_in)
        z_cond = self.cond(cond)
        z = self.fuse(torch.cat([z_img, z_cond], dim=-1))
        return self.template_head(z), self.mode_head(z)


def st_sample(logits: torch.Tensor, tau: float, hard: bool, training: bool, use_gumbel: bool):
    if training and use_gumbel:
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
    probs = F.softmax(logits / max(1e-6, tau), dim=-1)
    if hard:
        idx = probs.argmax(dim=-1)
        one = F.one_hot(idx, probs.size(-1)).float()
        return one.detach() - probs.detach() + probs
    return probs


def apply_modes(input_image: torch.Tensor, selected: torch.Tensor, mode_w: torch.Tensor, edit_channels: List[int]) -> torch.Tensor:
    b, t, c, h, w = input_image.shape
    actions = []

    keep = input_image
    replace = selected
    alpha50 = 0.50 * input_image + 0.50 * selected
    alpha75 = 0.25 * input_image + 0.75 * selected
    max_add = torch.maximum(input_image, selected)
    min_remove = torch.minimum(input_image, selected)
    soft_add = (input_image + 0.50 * selected * (1.0 - input_image)).clamp(0.0, 1.0)

    raw_actions = [keep, replace, alpha50, alpha75, max_add, min_remove, soft_add]

    mask = torch.zeros_like(input_image)
    for ch in edit_channels:
        if 0 <= ch < c:
            mask[:, :, ch:ch+1] = 1.0

    for a in raw_actions:
        # Non-edit channels stay original.
        actions.append(a * mask + input_image * (1.0 - mask))

    stack = torch.stack(actions, dim=1)  # [B,M,T,C,H,W]
    out = (mode_w.view(b, len(actions), 1, 1, 1, 1) * stack).sum(dim=1)
    return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------
# Metrics and losses.
# ---------------------------------------------------------------------

def batch_coverage(image: torch.Tensor, channels: List[int]) -> torch.Tensor:
    vals = []
    for ch in channels:
        if 0 <= ch < image.size(2):
            vals.append(image[:, :, ch].mean(dim=(-1, -2)))
    if not vals:
        return image.new_zeros(image.size(0), 1)
    cov = torch.stack(vals, dim=0).mean(dim=0)  # [B,T]
    return cov.mean(dim=1, keepdim=True)


def coverage_loss(output: torch.Tensor, input_image: torch.Tensor, actual, target, clear, channels: List[int], args):
    cov = batch_coverage(output, channels)
    base_cov = batch_coverage(input_image, channels)
    delta = ((target - actual) / clear.clamp_min(1.0)).clamp(-1.0, 1.0)
    desired = (base_cov + args.coverage_delta_gain * delta).clamp(args.min_coverage, args.max_coverage)
    loss = F.smooth_l1_loss(cov, desired, beta=0.04)
    loss = loss + args.lambda_overfill_internal * F.relu(cov - args.max_coverage).pow(2).mean()
    loss = loss + args.lambda_underfill_internal * F.relu(args.min_coverage - cov).pow(2).mean()
    return loss, {
        "coverage_mean": float(cov.mean().detach().cpu()),
        "coverage_desired": float(desired.mean().detach().cpu()),
    }


def edge_density_scalar(x: torch.Tensor) -> torch.Tensor:
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(-1, -2))
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(-1, -2))
    return dx + dy


def style_stats_loss(output: torch.Tensor, selected: torch.Tensor, channels: List[int]) -> torch.Tensor:
    if not channels:
        return output.new_tensor(0.0)
    loss = output.new_tensor(0.0)
    for ch in channels:
        if ch >= output.size(2):
            continue
        o = output[:, :, ch]
        s = selected[:, :, ch]
        loss = loss + F.smooth_l1_loss(o.mean(dim=(-1, -2)), s.mean(dim=(-1, -2)), beta=0.04)
        loss = loss + 0.5 * F.smooth_l1_loss(o.std(dim=(-1, -2)), s.std(dim=(-1, -2)), beta=0.04)
        loss = loss + 0.5 * F.smooth_l1_loss(edge_density_scalar(o), edge_density_scalar(s), beta=0.04)
    return loss / float(max(1, len(channels)))


def entropy_loss(w: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -(w.clamp_min(eps) * w.clamp_min(eps).log()).sum(dim=-1).mean()


def mode_cost_loss(mode_w: torch.Tensor, device) -> torch.Tensor:
    # keep, replace, alpha50, alpha75, max, min, soft_add
    costs = torch.tensor([0.00, 0.70, 0.22, 0.35, 0.28, 0.32, 0.30], dtype=torch.float32, device=device)
    return (mode_w * costs.view(1, -1)).sum(dim=-1).mean()


def image_l1(output: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
    return (output - input_image).abs().mean()


# ---------------------------------------------------------------------
# Training / evaluation.
# ---------------------------------------------------------------------

def evaluate(model, reward_model, loader, codebook, codebook_meta, device, args, raw_mean, raw_std, cloud_names, cov_channels, edit_channels, style_channels):
    model.eval()
    enable_frozen_reward_grads(reward_model)

    rows = []
    totals = {
        "mae": 0.0,
        "input_mae": 0.0,
        "base_mae": 0.0,
        "l1": 0.0,
        "coverage": 0.0,
        "n": 0,
    }

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            cloud = batch["cloud_features"].to(device)
            context = batch["context_features"].to(device)
            clear = batch["clear_wm2"].to(device).float()
            actual = batch["target_loss_wm2"].to(device).float()
            target = fixed_val_target(actual, clear, args)

            input_image, _drop, _info = white_dropout_input(image, args)
            cloud_input = derive_cloud_features_from_image(input_image, cloud, cloud_names, raw_mean, raw_std)
            input_pred = reward_forward(reward_model, input_image, cloud_input, context, clear)["loss_wm2"]
            base_pred = reward_forward(reward_model, image, cloud, context, clear)["loss_wm2"]

            inp_cov = batch_coverage(input_image, cov_channels)
            template_logits, mode_logits = model(input_image, context, clear, actual, target, input_pred, inp_cov)
            tw = st_sample(template_logits, args.eval_tau, hard=True, training=False, use_gumbel=False)
            mw = st_sample(mode_logits, args.eval_tau, hard=True, training=False, use_gumbel=False)

            selected = torch.einsum("bk,ktchw->btchw", tw, codebook)
            output = apply_modes(input_image, selected, mw, edit_channels)

            cloud_out = derive_cloud_features_from_image(output, cloud, cloud_names, raw_mean, raw_std)
            pred = reward_forward(reward_model, output, cloud_out, context, clear)["loss_wm2"]

            cov = batch_coverage(output, cov_channels)
            bs = image.size(0)
            totals["mae"] += float((pred - target).abs().mean()) * bs
            totals["input_mae"] += float((input_pred - target).abs().mean()) * bs
            totals["base_mae"] += float((base_pred - target).abs().mean()) * bs
            totals["l1"] += float((output - input_image).abs().mean()) * bs
            totals["coverage"] += float(cov.mean()) * bs
            totals["n"] += bs

            template_idx = tw.argmax(dim=-1).detach().cpu().tolist()
            mode_idx = mw.argmax(dim=-1).detach().cpu().tolist()
            for i in range(bs):
                ti = int(template_idx[i])
                mi = int(mode_idx[i])
                meta = codebook_meta[ti] if ti < len(codebook_meta) else {}
                rows.append({
                    "sample_id": batch["sample_id"][i],
                    "location": batch["location"][i],
                    "anchor": batch["anchor"][i],
                    "actual_loss_wm2": float(actual[i].item()),
                    "target_loss_wm2": float(target[i].item()),
                    "base_pred_loss_wm2": float(base_pred[i].item()),
                    "input_pred_loss_wm2": float(input_pred[i].item()),
                    "generated_pred_loss_wm2": float(pred[i].item()),
                    "abs_error_wm2": abs(float(pred[i].item() - target[i].item())),
                    "template_index": ti,
                    "template_sample_id": meta.get("sample_id", ""),
                    "template_coverage": meta.get("coverage", float("nan")),
                    "mode_index": mi,
                    "mode": MODE_NAMES[mi] if mi < len(MODE_NAMES) else str(mi),
                    "output_coverage": float(cov[i].item()),
                    "image_l1_vs_input": float((output[i] - input_image[i]).abs().mean().item()),
                })

    n = max(1, totals["n"])
    return {
        "val_target_mae_wm2": totals["mae"] / n,
        "val_input_mae_wm2": totals["input_mae"] / n,
        "val_base_mae_wm2": totals["base_mae"] / n,
        "val_improvement_vs_input_wm2": totals["input_mae"] / n - totals["mae"] / n,
        "val_improvement_vs_base_wm2": totals["base_mae"] / n - totals["mae"] / n,
        "val_l1_vs_input": totals["l1"] / n,
        "val_coverage_mean": totals["coverage"] / n,
    }, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True, help="v8 reward checkpoint")
    ap.add_argument("--out-dir", default="runs/cloud_template_selector_v13")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=192)

    # Codebook.
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--codebook-bins", type=int, default=10)
    ap.add_argument("--codebook-min-coverage", type=float, default=0.01)
    ap.add_argument("--codebook-max-coverage", type=float, default=0.85)
    ap.add_argument("--codebook-coverage-channels", default="0,1,2")

    # Input.
    ap.add_argument("--full-input-prob", type=float, default=0.35)
    ap.add_argument("--white-drop-prob", type=float, default=0.50)
    ap.add_argument("--white-threshold", type=float, default=0.35)
    ap.add_argument("--white-drop-channels", default="0,1,2,4,5,7")

    # Selection temperature.
    ap.add_argument("--tau-start", type=float, default=1.2)
    ap.add_argument("--tau-end", type=float, default=0.35)
    ap.add_argument("--tau-decay-epochs", type=int, default=45)
    ap.add_argument("--eval-tau", type=float, default=0.20)
    ap.add_argument("--hard-train", action="store_true", default=True)
    ap.add_argument("--gumbel-train", action="store_true", default=True)

    # Targets.
    ap.add_argument("--target-jitter-start-wm2", type=float, default=40.0)
    ap.add_argument("--target-jitter-wm2", type=float, default=200.0)
    ap.add_argument("--curriculum-epochs", type=int, default=35)
    ap.add_argument("--val-target-delta-wm2", type=float, default=120.0)
    ap.add_argument("--prefer-positive-targets", type=float, default=0.60)
    ap.add_argument("--extra-random-absolute-targets", type=float, default=0.10)
    ap.add_argument("--min-target-wm2", type=float, default=0.0)
    ap.add_argument("--beta-wm2", type=float, default=45.0)

    # Output/action.
    ap.add_argument("--edit-channels", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--coverage-channels", default="0,1,2")
    ap.add_argument("--style-channels", default="0,1,2,4,5,7")
    ap.add_argument("--coverage-delta-gain", type=float, default=0.35)
    ap.add_argument("--min-coverage", type=float, default=0.01)
    ap.add_argument("--max-coverage", type=float, default=0.70)
    ap.add_argument("--lambda-overfill-internal", type=float, default=8.0)
    ap.add_argument("--lambda-underfill-internal", type=float, default=1.0)

    # Loss weights.
    ap.add_argument("--lambda-reward", type=float, default=1.00)
    ap.add_argument("--lambda-coverage", type=float, default=1.25)
    ap.add_argument("--lambda-style", type=float, default=0.50)
    ap.add_argument("--lambda-l1", type=float, default=0.25)
    ap.add_argument("--lambda-mode-cost", type=float, default=0.08)
    ap.add_argument("--lambda-template-entropy", type=float, default=0.02)
    ap.add_argument("--lambda-mode-entropy", type=float, default=0.01)

    ap.add_argument("--min-clear-wm2", type=float, default=None)
    ap.add_argument("--early-stop-patience", type=int, default=14)
    ap.add_argument("--save-examples", type=int, default=8)
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

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Empty train or val dataset after cleaning.")

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    reward_model = v8.CloudRadiationV8CleanDirect(**ckpt["model_kwargs"]).to(device)
    reward_model.load_state_dict(strip_prefix(ckpt["model_state"]))
    enable_frozen_reward_grads(reward_model)

    sample = train_ds[0]
    frames = int(sample["image"].shape[0])
    in_channels = int(sample["image"].shape[1])
    loss_scale = float(ckpt.get("model_kwargs", {}).get("loss_scale", ckargs.get("loss_scale", 300.0)))

    cov_channels = parse_channels(args.coverage_channels, in_channels)
    edit_channels = parse_channels(args.edit_channels, in_channels)
    style_channels = parse_channels(args.style_channels, in_channels)

    codebook, codebook_meta = build_codebook(train_ds, args, device)
    write_csv(out_dir / "codebook_meta.csv", codebook_meta)

    model = CloudTemplateSelectorV13(
        frames=frames,
        in_channels=in_channels,
        codebook_size=codebook.size(0),
        num_modes=len(MODE_NAMES),
        context_dim=8,
        base=args.base_channels,
        hidden=args.hidden_dim,
        loss_scale=loss_scale,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=5)
    raw_mean, raw_std = get_cloud_raw_stats(v8, x_norm, raw_names, cloud_names, device)

    print(json.dumps({
        "trainer": "CloudTemplateSelectorV13",
        "idea": "scene + target -> select real cloud template + edit mode",
        "data_root": str(root),
        "reward_checkpoint": str(args.checkpoint),
        "device": str(device),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "frames": frames,
        "in_channels": in_channels,
        "codebook_size": int(codebook.size(0)),
        "modes": MODE_NAMES,
        "no_free_pixel_generation": True,
    }, indent=2))

    history = []
    best = float("inf")
    bad = 0

    for epoch in range(1, args.epochs + 1):
        enable_frozen_reward_grads(reward_model)
        model.train()

        progress = min(1.0, epoch / max(1, args.tau_decay_epochs))
        tau = args.tau_start + progress * (args.tau_end - args.tau_start)

        totals = {
            "loss": 0.0,
            "mae": 0.0,
            "input_mae": 0.0,
            "base_mae": 0.0,
            "l1": 0.0,
            "cov": 0.0,
            "n": 0,
        }

        for batch in train_loader:
            image = batch["image"].to(device)
            cloud = batch["cloud_features"].to(device)
            context = batch["context_features"].to(device)
            clear = batch["clear_wm2"].to(device).float()
            actual = batch["target_loss_wm2"].to(device).float()
            target = target_schedule(actual, clear, args, epoch)

            input_image, _drop, _info = maybe_make_input(image, args)

            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                cloud_input = derive_cloud_features_from_image(input_image, cloud, cloud_names, raw_mean, raw_std)
                input_pred = reward_forward(reward_model, input_image, cloud_input, context, clear)["loss_wm2"]
                base_pred = reward_forward(reward_model, image, cloud, context, clear)["loss_wm2"]

            inp_cov = batch_coverage(input_image, cov_channels)
            template_logits, mode_logits = model(input_image, context, clear, actual, target, input_pred, inp_cov)

            template_w = st_sample(template_logits, tau=tau, hard=args.hard_train, training=True, use_gumbel=args.gumbel_train)
            mode_w = st_sample(mode_logits, tau=tau, hard=args.hard_train, training=True, use_gumbel=args.gumbel_train)

            selected = torch.einsum("bk,ktchw->btchw", template_w, codebook)
            output = apply_modes(input_image, selected, mode_w, edit_channels)

            cloud_out = derive_cloud_features_from_image(output, cloud, cloud_names, raw_mean, raw_std)
            pred = reward_forward(reward_model, output, cloud_out, context, clear)["loss_wm2"]

            reward_loss = F.smooth_l1_loss(pred, target, beta=args.beta_wm2)
            cov_loss, cov_info = coverage_loss(output, input_image, actual, target, clear, cov_channels, args)
            st_loss = style_stats_loss(output, selected, style_channels)
            l1 = image_l1(output, input_image)
            m_cost = mode_cost_loss(mode_w, device)
            t_ent = entropy_loss(F.softmax(template_logits / max(1e-6, tau), dim=-1))
            m_ent = entropy_loss(F.softmax(mode_logits / max(1e-6, tau), dim=-1))

            # Entropy is subtracted to prevent early collapse.
            loss = (
                args.lambda_reward * reward_loss
                + args.lambda_coverage * cov_loss
                + args.lambda_style * st_loss
                + args.lambda_l1 * l1
                + args.lambda_mode_cost * m_cost
                - args.lambda_template_entropy * t_ent
                - args.lambda_mode_entropy * m_ent
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            opt.step()

            bs = image.size(0)
            totals["loss"] += float(loss.detach()) * bs
            totals["mae"] += float((pred.detach() - target).abs().mean()) * bs
            totals["input_mae"] += float((input_pred.detach() - target).abs().mean()) * bs
            totals["base_mae"] += float((base_pred.detach() - target).abs().mean()) * bs
            totals["l1"] += float(l1.detach()) * bs
            totals["cov"] += cov_info["coverage_mean"] * bs
            totals["n"] += bs

        val, val_rows = evaluate(model, reward_model, val_loader, codebook, codebook_meta, device, args, raw_mean, raw_std, cloud_names, cov_channels, edit_channels, style_channels)
        scheduler.step(val["val_target_mae_wm2"])

        n = max(1, totals["n"])
        row = {
            "epoch": epoch,
            "tau": tau,
            "train_loss": totals["loss"] / n,
            "train_target_mae_wm2": totals["mae"] / n,
            "train_input_mae_wm2": totals["input_mae"] / n,
            "train_base_mae_wm2": totals["base_mae"] / n,
            "train_improvement_vs_input_wm2": totals["input_mae"] / n - totals["mae"] / n,
            "train_improvement_vs_base_wm2": totals["base_mae"] / n - totals["mae"] / n,
            "train_l1_vs_input": totals["l1"] / n,
            "train_coverage_mean": totals["cov"] / n,
            **val,
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_mae={row['train_target_mae_wm2']:.2f} "
            f"input={row['train_input_mae_wm2']:.2f} "
            f"val_mae={row['val_target_mae_wm2']:.2f} "
            f"val_input={row['val_input_mae_wm2']:.2f} "
            f"val_improve={row['val_improvement_vs_input_wm2']:.2f} "
            f"cov={row['val_coverage_mean']:.3f} "
            f"l1={row['val_l1_vs_input']:.4f} "
            f"tau={tau:.3f} "
            f"lr={row['lr']:.2e}"
        )

        write_csv(out_dir / "history.csv", history)
        write_csv(out_dir / "val_predictions_latest.csv", val_rows)

        payload = {
            "architecture": "CloudTemplateSelectorV13",
            "model_state": model.state_dict(),
            "args": vars(args),
            "reward_checkpoint": str(args.checkpoint),
            "raw_feature_names": raw_names,
            "cloud_names": cloud_names,
            "frames": frames,
            "in_channels": in_channels,
            "loss_scale": loss_scale,
            "codebook": codebook.detach().cpu(),
            "codebook_meta": codebook_meta,
            "mode_names": MODE_NAMES,
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(payload, out_dir / "last_selector.pt")

        if row["val_target_mae_wm2"] < best:
            best = row["val_target_mae_wm2"]
            bad = 0
            torch.save(payload, out_dir / "best_selector.pt")
            write_csv(out_dir / "val_predictions_best.csv", val_rows)
            print(f"saved best_selector.pt val_mae={best:.2f}W/m2")
        else:
            bad += 1

        if args.early_stop_patience > 0 and bad >= args.early_stop_patience:
            print(f"early stop: {bad} epochs without val improvement")
            break

    # Save visual examples from best/latest model.
    model.eval()
    batch = next(iter(val_loader))
    image = batch["image"].to(device)
    cloud = batch["cloud_features"].to(device)
    context = batch["context_features"].to(device)
    clear = batch["clear_wm2"].to(device).float()
    actual = batch["target_loss_wm2"].to(device).float()
    target = fixed_val_target(actual, clear, args)

    with torch.no_grad():
        input_image, _drop, _info = white_dropout_input(image, args)
        cloud_input = derive_cloud_features_from_image(input_image, cloud, cloud_names, raw_mean, raw_std)
        input_pred = reward_forward(reward_model, input_image, cloud_input, context, clear)["loss_wm2"]
        base_pred = reward_forward(reward_model, image, cloud, context, clear)["loss_wm2"]
        inp_cov = batch_coverage(input_image, cov_channels)
        template_logits, mode_logits = model(input_image, context, clear, actual, target, input_pred, inp_cov)
        tw = st_sample(template_logits, args.eval_tau, hard=True, training=False, use_gumbel=False)
        mw = st_sample(mode_logits, args.eval_tau, hard=True, training=False, use_gumbel=False)
        selected = torch.einsum("bk,ktchw->btchw", tw, codebook)
        output = apply_modes(input_image, selected, mw, edit_channels)
        cloud_out = derive_cloud_features_from_image(output, cloud, cloud_names, raw_mean, raw_std)
        pred = reward_forward(reward_model, output, cloud_out, context, clear)["loss_wm2"]

    examples = []
    template_idx = tw.argmax(dim=-1).detach().cpu().tolist()
    mode_idx = mw.argmax(dim=-1).detach().cpu().tolist()

    for i in range(min(args.save_examples, image.size(0))):
        prefix = out_dir / "examples" / f"sample_{i:02d}"
        for ch in [0, 1, 2, 4, 5, 7]:
            if ch < image.size(2):
                save_sequence_strip(image[i], prefix.with_name(prefix.name + f"_original_ch{ch}.png"), ch)
                save_sequence_strip(input_image[i], prefix.with_name(prefix.name + f"_input_ch{ch}.png"), ch)
                save_sequence_strip(selected[i], prefix.with_name(prefix.name + f"_selected_template_ch{ch}.png"), ch)
                save_sequence_strip(output[i], prefix.with_name(prefix.name + f"_output_ch{ch}.png"), ch)
        ti = int(template_idx[i])
        mi = int(mode_idx[i])
        examples.append({
            "i": i,
            "sample_id": batch["sample_id"][i],
            "location": batch["location"][i],
            "anchor": batch["anchor"][i],
            "actual_loss_wm2": float(actual[i].item()),
            "target_loss_wm2": float(target[i].item()),
            "base_pred_wm2": float(base_pred[i].item()),
            "input_pred_wm2": float(input_pred[i].item()),
            "output_pred_wm2": float(pred[i].item()),
            "template_index": ti,
            "template_meta": codebook_meta[ti] if ti < len(codebook_meta) else {},
            "mode": MODE_NAMES[mi] if mi < len(MODE_NAMES) else str(mi),
        })

    summary = {
        "best_val_target_mae_wm2": best,
        "history_csv": str(out_dir / "history.csv"),
        "best_selector": str(out_dir / "best_selector.pt"),
        "last_selector": str(out_dir / "last_selector.pt"),
        "codebook_meta_csv": str(out_dir / "codebook_meta.csv"),
        "examples": examples,
    }
    write_json(out_dir / "summary.json", summary)
    print("DONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
