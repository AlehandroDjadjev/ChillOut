#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F




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
    raise RuntimeError(f"Could not import v6 trainer module from current folder. Last error: {last}")


def strip_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def tensor_to_png(t: torch.Tensor, path: Path) -> None:
    arr = t.detach().float().cpu().clamp(0, 1).numpy()
    if arr.ndim == 3:
        arr = arr[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(path)


def save_diff_png(before: torch.Tensor, after: torch.Tensor, path: Path) -> None:
    b = before.detach().float().cpu().squeeze().numpy()
    a = after.detach().float().cpu().squeeze().numpy()
    d = np.clip((a - b) * 0.5 + 0.5, 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((d * 255.0).astype(np.uint8), mode="L").save(path)


def save_sequence_strip(seq: torch.Tensor, path: Path) -> None:
    arr = seq.detach().float().cpu().clamp(0, 1).numpy()
    imgs = []
    for i in range(arr.shape[0]):
        im = Image.fromarray((arr[i, 0] * 255).astype(np.uint8), mode="L")
        im = ImageOps.expand(im, border=2, fill=128)
        imgs.append(im)
    out = Image.new("L", (sum(im.width for im in imgs), max(im.height for im in imgs)), 0)
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def denorm_cloud(cloud_norm: torch.Tensor, ckpt: Dict[str, Any], raw_names: List[str], cloud_names: List[str]) -> torch.Tensor:
    normalizer = ckpt["normalizer"]
    mean_all = torch.tensor(normalizer["mean"], dtype=cloud_norm.dtype, device=cloud_norm.device)
    std_all = torch.tensor(normalizer["std"], dtype=cloud_norm.dtype, device=cloud_norm.device)
    raw_to_idx = {n: i for i, n in enumerate(raw_names)}
    idx = torch.tensor([raw_to_idx[n] for n in cloud_names], dtype=torch.long, device=cloud_norm.device)
    return cloud_norm * std_all[idx].view(1, 1, -1) + mean_all[idx].view(1, 1, -1)


def raw_feature(raw_cloud: torch.Tensor, cloud_names: List[str], name: str) -> torch.Tensor:
    if name not in cloud_names:
        return torch.zeros(raw_cloud.shape[:2] + (1,), dtype=raw_cloud.dtype, device=raw_cloud.device)
    return raw_cloud[:, :, cloud_names.index(name):cloud_names.index(name)+1]


def mask_scalar_consistency_loss(mask: torch.Tensor, cloud_norm: torch.Tensor, ckpt: Dict[str, Any], raw_names: List[str], cloud_names: List[str]) -> torch.Tensor:
    raw = denorm_cloud(cloud_norm, ckpt, raw_names, cloud_names)
    frac_feat = raw_feature(raw, cloud_names, "cloud_s2_fraction").clamp(0, 1)
    prob_feat = raw_feature(raw, cloud_names, "cloud_s2_prob_mean").clamp(0, 1)
    edge_feat = raw_feature(raw, cloud_names, "cloud_s2_edge_density").clamp(0, 1)

    frac_img = mask.mean(dim=(-1, -2))
    prob_img = mask.sum(dim=(-1, -2)) / mask.gt(0.05).float().sum(dim=(-1, -2)).clamp_min(1.0)
    dx = (mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]).abs().mean(dim=(-1, -2))
    dy = (mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]).abs().mean(dim=(-1, -2))
    edge_img = 0.5 * (dx + dy)
    return F.mse_loss(frac_img, frac_feat) + 0.5 * F.mse_loss(prob_img, prob_feat) + 0.25 * F.mse_loss(edge_img, edge_feat)


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    return (mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]).abs().mean() + (mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]).abs().mean()


def clamp_target_for_scene(target: float, clear_wm2: float, min_loss: float = -100.0) -> float:
    return float(max(min_loss, min(max(0.0, clear_wm2), target)))

PATCHED_CUDNN_LSTM_INPUT_GRADS = True

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--sample-id", default=None)
    tg = ap.add_mutually_exclusive_group()
    tg.add_argument("--target-loss-wm2", type=float, default=None)
    tg.add_argument("--target-delta-wm2", type=float, default=75.0)
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--lr-mask", type=float, default=0.07)
    ap.add_argument("--lr-cloud", type=float, default=0.035)
    ap.add_argument("--max-cloud-delta-z", type=float, default=1.75)
    ap.add_argument("--optimize-frames", choices=["last", "all"], default="last")
    ap.add_argument("--target-head", choices=["cloud", "image", "both"], default="cloud")
    ap.add_argument("--lambda-tv", type=float, default=0.04)
    ap.add_argument("--lambda-anchor-mask", type=float, default=0.06)
    ap.add_argument("--lambda-anchor-cloud", type=float, default=0.04)
    ap.add_argument("--lambda-consistency", type=float, default=0.75)
    ap.add_argument("--lambda-bounds", type=float, default=0.15)
    ap.add_argument("--out-dir", default="rl_cloud_radiation_single_scene")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    mod = import_bottom_model()
    root = Path(args.data_root).resolve()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = mod.load_metadata(root)
    raw_names = ckpt.get("raw_feature_names", meta["raw_feature_names"])
    cloud_names = ckpt.get("cloud_feature_names", meta["cloud_feature_names"])
    world_names = ckpt.get("world_feature_names", meta["world_feature_names"])
    x_norm = mod.Normalizer.from_state_dict(ckpt["normalizer"])
    delta_norm = mod.TargetNormalizer.from_state_dict(ckpt.get("delta_normalizer", ckpt["target_normalizer"]))

    records = mod.load_records(root, args.split)
    ds = mod.CloudTempResidualSequenceDataset(
        root, records, raw_names, cloud_names, world_names, x_norm, delta_norm,
        int(ckpt.get("image_height", 160)), int(ckpt.get("image_width", 160)),
        int(ckpt.get("lookback", 4)), float(ckpt.get("args", {}).get("max_gap_days", 12.0)),
        augment=False, cache_images=False,
    )

    if args.sample_id:
        idx = None
        for i, w in enumerate(ds.windows):
            if str(w.records[-1].get("sample_id", "")) == args.sample_id:
                idx = i
                break
        if idx is None:
            raise SystemExit(f"sample_id not found: {args.sample_id}")
    else:
        idx = int(args.sample_index)

    item = ds[idx]
    device = torch.device(args.device)
    model = mod.ResidualTrendSplitConvLSTM(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(strip_prefix(ckpt["model_state"]))
    enable_frozen_lstm_input_grads(model)

    radiation_scale = float(ckpt.get("args", {}).get("radiation_loss_scale", 300.0))
    context_scale = float(ckpt.get("args", {}).get("context_scale", 0.15))

    world = item["world_features"].unsqueeze(0).to(device)
    trend = item["trend_features"].unsqueeze(0).to(device)
    base_mask = item["mask"].unsqueeze(0).to(device)
    base_cloud = item["cloud_features"].unsqueeze(0).to(device)

    actual_loss = float(item["radiation_loss_raw"].item())
    clear_wm2 = float(item["radiation_clear_wm2"].item())
    observed_wm2 = float(item["radiation_observed_wm2"].item())
    valid = float(item["radiation_valid"].item())
    target_loss = float(args.target_loss_wm2) if args.target_loss_wm2 is not None else actual_loss + float(args.target_delta_wm2)
    target_loss = clamp_target_for_scene(target_loss, clear_wm2)

    eps = 1e-4
    base_mask_clamped = base_mask.clamp(eps, 1.0 - eps)
    mask_logits = torch.log(base_mask_clamped / (1.0 - base_mask_clamped)).detach().clone().requires_grad_(True)
    cloud_delta_raw = torch.zeros_like(base_cloud, requires_grad=True)
    opt = torch.optim.AdamW([{"params": [mask_logits], "lr": args.lr_mask}, {"params": [cloud_delta_raw], "lr": args.lr_cloud}], weight_decay=0.0)

    frame_mask = torch.zeros_like(base_mask)
    if args.optimize_frames == "last":
        frame_mask[:, -1:] = 1.0
    else:
        frame_mask[:] = 1.0

    target_t = torch.tensor([[target_loss]], dtype=torch.float32, device=device)
    history = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        base_out = model(base_mask, base_cloud, world, trend, context_scale=context_scale)
        base_image_pred = float((base_out["image_radiation"] * radiation_scale).detach().cpu().item())
        base_cloud_pred = float((base_out["cloud_radiation"] * radiation_scale).detach().cpu().item())

    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        proposed_mask_all = torch.sigmoid(mask_logits)
        mask = base_mask * (1.0 - frame_mask) + proposed_mask_all * frame_mask
        cloud_delta = args.max_cloud_delta_z * torch.tanh(cloud_delta_raw)
        cloud = base_cloud + cloud_delta
        out = model(mask, cloud, world, trend, context_scale=context_scale)
        pred_image = out["image_radiation"] * radiation_scale
        pred_cloud = out["cloud_radiation"] * radiation_scale

        if args.target_head == "cloud":
            target_loss_term = F.smooth_l1_loss(pred_cloud, target_t, beta=50.0)
        elif args.target_head == "image":
            target_loss_term = F.smooth_l1_loss(pred_image, target_t, beta=50.0)
        else:
            target_loss_term = 0.5 * (F.smooth_l1_loss(pred_cloud, target_t, beta=50.0) + F.smooth_l1_loss(pred_image, target_t, beta=50.0))

        tv = total_variation(mask)
        anchor_mask = F.mse_loss(mask, base_mask)
        anchor_cloud = F.mse_loss(cloud_delta, torch.zeros_like(cloud_delta))
        consistency = mask_scalar_consistency_loss(mask, cloud, ckpt, raw_names, cloud_names)
        raw_cloud = denorm_cloud(cloud, ckpt, raw_names, cloud_names)
        bounds = F.relu(-raw_cloud).pow(2).mean() + F.relu(raw_cloud - 1.5).pow(2).mean()
        loss = target_loss_term + args.lambda_tv * tv + args.lambda_anchor_mask * anchor_mask + args.lambda_anchor_cloud * anchor_cloud + args.lambda_consistency * consistency + args.lambda_bounds * bounds

        if step < args.steps:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([mask_logits, cloud_delta_raw], 5.0)
            opt.step()

        if step % max(1, args.steps // 50) == 0 or step == args.steps:
            rec = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "target_loss_wm2": target_loss,
                "actual_loss_wm2": actual_loss,
                "pred_image_loss_wm2": float(pred_image.detach().cpu().item()),
                "pred_cloud_loss_wm2": float(pred_cloud.detach().cpu().item()),
                "abs_error_cloud_wm2": abs(float(pred_cloud.detach().cpu().item()) - target_loss),
                "abs_error_image_wm2": abs(float(pred_image.detach().cpu().item()) - target_loss),
                "tv": float(tv.detach().cpu()),
                "anchor_mask": float(anchor_mask.detach().cpu()),
                "anchor_cloud": float(anchor_cloud.detach().cpu()),
                "consistency": float(consistency.detach().cpu()),
                "bounds": float(bounds.detach().cpu()),
            }
            history.append(rec)
            print(f"step={step:04d} target={target_loss:.1f} cloud_pred={rec['pred_cloud_loss_wm2']:.1f} image_pred={rec['pred_image_loss_wm2']:.1f} cloud_err={rec['abs_error_cloud_wm2']:.1f} loss={rec['loss']:.4f}", flush=True)

    with torch.no_grad():
        final_mask_all = torch.sigmoid(mask_logits)
        final_mask = base_mask * (1.0 - frame_mask) + final_mask_all * frame_mask
        final_cloud = base_cloud + args.max_cloud_delta_z * torch.tanh(cloud_delta_raw)
        final_out = model(final_mask, final_cloud, world, trend, context_scale=context_scale)
        final_image_pred = float((final_out["image_radiation"] * radiation_scale).detach().cpu().item())
        final_cloud_pred = float((final_out["cloud_radiation"] * radiation_scale).detach().cpu().item())

    save_sequence_strip(base_mask[0], out_dir / "sequence_original.png")
    save_sequence_strip(final_mask[0], out_dir / "sequence_optimized.png")
    tensor_to_png(base_mask[0, -1], out_dir / "original_last.png")
    tensor_to_png(final_mask[0, -1], out_dir / "optimized_last.png")
    save_diff_png(base_mask[0, -1], final_mask[0, -1], out_dir / "difference.png")
    np.save(out_dir / "optimized_mask_sequence.npy", final_mask[0].detach().cpu().numpy())
    np.save(out_dir / "optimized_cloud_features_norm.npy", final_cloud[0].detach().cpu().numpy())

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)

    final = {
        "sample_index": idx,
        "sample_id": item["sample_id"],
        "location": item["location"],
        "anchor": item["anchor"],
        "radiation_daylight_valid": valid,
        "clear_sky_proxy_wm2": clear_wm2,
        "observed_shortwave_wm2": observed_wm2,
        "actual_cloud_loss_wm2": actual_loss,
        "target_cloud_loss_wm2": target_loss,
        "base_image_pred_wm2": base_image_pred,
        "base_cloud_pred_wm2": base_cloud_pred,
        "final_image_pred_wm2": final_image_pred,
        "final_cloud_pred_wm2": final_cloud_pred,
        "final_cloud_abs_error_wm2": abs(final_cloud_pred - target_loss),
        "final_image_abs_error_wm2": abs(final_image_pred - target_loss),
        "optimized_files": {
            "original_last": str(out_dir / "original_last.png"),
            "optimized_last": str(out_dir / "optimized_last.png"),
            "difference": str(out_dir / "difference.png"),
            "sequence_original": str(out_dir / "sequence_original.png"),
            "sequence_optimized": str(out_dir / "sequence_optimized.png"),
            "history": str(out_dir / "history.csv"),
        },
        "notes": [
            "Only cloud mask and cloud scalar features were optimized.",
            "World features, trend features, current temp, location, and timestamp stayed fixed.",
            "This is differentiable reward optimization through the frozen v6 bottom model, not full PPO.",
        ],
    }
    (out_dir / "final_state.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\nDONE")
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
