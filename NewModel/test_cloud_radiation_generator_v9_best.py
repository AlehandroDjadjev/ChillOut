#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import torch


CHANNEL_NAMES_8 = [
    "cloud_mask_or_fraction",
    "cloud_probability",
    "white_cloud",
    "cirrus",
    "high_cloud",
    "medium_cloud",
    "aot_on_cloud",
    "visible_gray",
]


def import_v9():
    names = [
        "train_cloud_radiation_generator_v9_ZERO_BLOBS",
        "train_cloud_radiation_generator_v9_zero_blobs",
    ]
    last = None
    for name in names:
        try:
            return __import__(name, fromlist=["dummy"])
        except Exception as exc:
            last = exc
    raise RuntimeError(
        "Could not import train_cloud_radiation_generator_v9_ZERO_BLOBS.py. "
        "Put this test script in the same NewModel folder as the v9 training script. "
        f"Last error: {last}"
    )


def strip_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def to_gray_img(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")


def signed_diff_img(before: np.ndarray, after: np.ndarray, scale: float = 0.35) -> Image.Image:
    d = np.asarray(after, dtype=np.float32) - np.asarray(before, dtype=np.float32)
    d = np.nan_to_num(d, nan=0.0)
    pos = np.clip(d / scale, 0.0, 1.0)
    neg = np.clip(-d / scale, 0.0, 1.0)
    rgb = np.zeros((*d.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (pos * 255).astype(np.uint8)
    rgb[..., 2] = (neg * 255).astype(np.uint8)
    rgb[..., 1] = (np.clip(1.0 - np.maximum(pos, neg), 0.0, 1.0) * 40).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str) -> None:
    draw.text(xy, text, fill=(255, 255, 255))


def make_channel_sheet(
    original: np.ndarray,
    corrupted: np.ndarray,
    generated: np.ndarray,
    ch: int,
    cname: str,
    out_path: Path,
) -> None:
    """Arrays are [T,C,H,W]. Creates rows with 4 frames each."""
    t, c, h, w = original.shape
    label_w = 135
    top_h = 34
    gap = 8
    row_gap = 18

    rows = [
        ("ORIGINAL", [to_gray_img(original[i, ch]) for i in range(t)]),
        ("INPUT", [to_gray_img(corrupted[i, ch]) for i in range(t)]),
        ("OUTPUT", [to_gray_img(generated[i, ch]) for i in range(t)]),
        ("OUT-INPUT", [signed_diff_img(corrupted[i, ch], generated[i, ch]) for i in range(t)]),
        ("OUT-ORIG", [signed_diff_img(original[i, ch], generated[i, ch]) for i in range(t)]),
    ]

    sheet_w = label_w + t * w + (t - 1) * gap
    sheet_h = top_h + len(rows) * h + (len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (8, 8), f"channel {ch}: {cname} | red=increase blue=decrease")

    y = top_h
    for label, frames in rows:
        draw_text(draw, (8, y + h // 2 - 8), label)
        x = label_w
        for i, im in enumerate(frames):
            canvas.paste(im.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(150, 150, 150), width=1)
            draw_text(draw, (x + 4, y + 4), f"t{i}")
            x += w + gap
        y += h + row_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def image_stats(name: str, arr: np.ndarray, channel_names: List[str]) -> List[Dict[str, Any]]:
    rows = []
    t, c, h, w = arr.shape
    for frame in range(t):
        for ch in range(c):
            x = arr[frame, ch]
            cname = channel_names[ch] if ch < len(channel_names) else f"channel_{ch:02d}"
            rows.append({
                "name": name,
                "frame": frame,
                "channel": ch,
                "channel_name": cname,
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "nonzero_frac_gt_0p01": float(np.mean(x > 0.01)),
                "dark_frac_lt_0p02": float(np.mean(x < 0.02)),
            })
    return rows


def diff_stats(name: str, before: np.ndarray, after: np.ndarray, channel_names: List[str]) -> List[Dict[str, Any]]:
    rows = []
    t, c, h, w = before.shape
    d = after - before
    ad = np.abs(d)
    for frame in range(t):
        for ch in range(c):
            cname = channel_names[ch] if ch < len(channel_names) else f"channel_{ch:02d}"
            x = d[frame, ch]
            a = ad[frame, ch]
            rows.append({
                "name": name,
                "frame": frame,
                "channel": ch,
                "channel_name": cname,
                "absdiff_mean": float(np.mean(a)),
                "absdiff_max": float(np.max(a)),
                "signed_mean_delta": float(np.mean(x)),
                "positive_delta_frac_gt_0p01": float(np.mean(x > 0.01)),
                "negative_delta_frac_lt_minus_0p01": float(np.mean(x < -0.01)),
            })
    return rows


def target_from_args(actual: torch.Tensor, clear: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, Any]]:
    meta: Dict[str, Any] = {}

    if args.target_loss_wm2 is not None:
        target = torch.tensor([[float(args.target_loss_wm2)]], dtype=torch.float32, device=actual.device)
        meta["target_mode"] = "absolute_loss_wm2"

    elif args.target_cooling_c is not None:
        rho = float(args.air_density_kg_m3)
        cp = float(args.air_cp_j_kg_k)
        h = float(args.mixing_layer_m)
        seconds = float(args.horizon_hours) * 3600.0
        coupling = max(1e-6, float(args.coupling_factor))
        rad_delta = float(args.target_cooling_c) * rho * cp * h / (seconds * coupling)
        target = actual + rad_delta
        meta.update({
            "target_mode": "temperature_to_extra_radiation_loss",
            "target_cooling_c": float(args.target_cooling_c),
            "computed_radiation_delta_wm2": float(rad_delta),
            "horizon_hours": float(args.horizon_hours),
            "mixing_layer_m": h,
            "coupling_factor": coupling,
        })

    else:
        target = actual + float(args.target_delta_wm2)
        meta["target_mode"] = "delta_loss_wm2"
        meta["requested_delta_wm2"] = float(args.target_delta_wm2)

    unclamped = target.clone()
    target = torch.minimum(target.clamp(min=0.0), clear.clamp_min(0.0))
    meta["target_was_clamped"] = bool((target != unclamped).any().item())
    return target, meta


def override_args_from_cli(gargs: Dict[str, Any], args) -> Dict[str, Any]:
    out = dict(gargs)

    def set_if(name: str, value):
        if value is not None:
            out[name] = value

    set_if("keep_ratio_start", args.keep_ratio)
    set_if("keep_ratio_end", args.keep_ratio)
    set_if("noise_fill_ratio", args.noise_fill_ratio)
    set_if("zero_start_prob", args.zero_start_prob)
    set_if("noise_start_prob", args.noise_start_prob)
    set_if("corrupt_block_size", args.corrupt_block_size)
    set_if("degrade_frames", args.degrade_frames)
    set_if("black_value", args.black_value)
    set_if("num_blobs", args.num_blobs)
    set_if("min_blob_radius", args.min_blob_radius)
    set_if("max_blob_radius", args.max_blob_radius)
    return out


def ns_from_dict(d: Dict[str, Any]):
    class A:
        pass
    a = A()
    for k, v in d.items():
        setattr(a, k, v)
    return a


def parse_channels(s: str, c: int) -> List[int]:
    if s.strip().lower() == "all":
        return list(range(c))
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        i = int(x)
        if 0 <= i < c:
            vals.append(i)
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--generator-checkpoint", required=True, help="runs/.../best_generator.pt")
    ap.add_argument("--reward-checkpoint", default=None, help="Optional override. If omitted, uses checkpoint metadata.")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--sample-id", default=None)

    ap.add_argument("--target-loss-wm2", type=float, default=None)
    ap.add_argument("--target-delta-wm2", type=float, default=180.0)
    ap.add_argument("--target-cooling-c", type=float, default=None)
    ap.add_argument("--horizon-hours", type=float, default=6.0)
    ap.add_argument("--mixing-layer-m", type=float, default=500.0)
    ap.add_argument("--coupling-factor", type=float, default=0.10)
    ap.add_argument("--air-density-kg-m3", type=float, default=1.2)
    ap.add_argument("--air-cp-j-kg-k", type=float, default=1005.0)

    # Optional test-time corruption overrides. If omitted, uses generator checkpoint args.
    ap.add_argument("--keep-ratio", type=float, default=None)
    ap.add_argument("--noise-fill-ratio", type=float, default=None)
    ap.add_argument("--zero-start-prob", type=float, default=None)
    ap.add_argument("--noise-start-prob", type=float, default=None)
    ap.add_argument("--corrupt-block-size", type=int, default=None)
    ap.add_argument("--degrade-frames", choices=["last", "all", "random_last_or_all"], default=None)
    ap.add_argument("--black-value", type=float, default=None)

    # Optional blob overrides.
    ap.add_argument("--num-blobs", type=int, default=None)
    ap.add_argument("--min-blob-radius", type=float, default=None)
    ap.add_argument("--max-blob-radius", type=float, default=None)

    ap.add_argument("--channels", default="0,1,2,4,5,7")
    ap.add_argument("--out-dir", default="diagnostics/generator_v9_best_test")
    ap.add_argument("--min-clear-wm2", type=float, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--save-npz", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    v9 = import_v9()
    v8 = v9.import_v8_reward()

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    gen_ckpt = torch.load(args.generator_checkpoint, map_location="cpu")
    gargs = dict(gen_ckpt.get("args", {}))
    gargs = override_args_from_cli(gargs, args)
    gns = ns_from_dict(gargs)

    reward_path = args.reward_checkpoint or gen_ckpt.get("reward_checkpoint")
    if reward_path is None:
        raise RuntimeError("No reward checkpoint found. Pass --reward-checkpoint.")
    reward_ckpt = torch.load(reward_path, map_location="cpu")

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_names = list(reward_ckpt["raw_feature_names"])
    cloud_names = list(reward_ckpt["cloud_feature_names"])
    x_norm = v8.base.Normalizer.from_state_dict(reward_ckpt["normalizer"])
    ckargs = reward_ckpt.get("args", {})
    cleaner = v9.make_clean_args(ckargs, override_min_clear=args.min_clear_wm2)

    records = v8.clean_radiation_records(v8.base.load_records(root, args.split), cleaner, args.split)
    ds = v8.RadiationSequenceDataset(
        root=root,
        records=records,
        raw_names=raw_names,
        cloud_names=cloud_names,
        x_norm=x_norm,
        image_height=int(reward_ckpt.get("image_height", 96)),
        image_width=int(reward_ckpt.get("image_width", 96)),
        lookback=int(reward_ckpt.get("lookback", 4)),
        max_gap_days=float(ckargs.get("max_gap_days", 12.0)),
        use_cloud_tensor=bool(ckargs.get("use_cloud_tensor", True)),
        augment=False,
        min_clear_wm2=float(cleaner.min_clear_wm2),
    )
    if len(ds) == 0:
        raise RuntimeError("Cleaned dataset split is empty.")

    idx = int(args.sample_index)
    if args.sample_id is not None:
        found = None
        for i in range(len(ds)):
            if str(ds[i]["sample_id"]) == str(args.sample_id):
                found = i
                break
        if found is None:
            raise RuntimeError(f"sample-id {args.sample_id!r} not found in cleaned split.")
        idx = found
    if idx < 0 or idx >= len(ds):
        raise IndexError(f"sample-index {idx} outside cleaned split length {len(ds)}")

    item = ds[idx]
    image = item["image"].unsqueeze(0).to(device)                 # [1,T,C,H,W]
    cloud = item["cloud_features"].unsqueeze(0).to(device)        # [1,T,F]
    context = item["context_features"].unsqueeze(0).to(device)    # [1,8]
    clear = item["clear_wm2"].view(1, 1).to(device).float()
    actual = item["target_loss_wm2"].view(1, 1).to(device).float()
    target, target_meta = target_from_args(actual, clear, args)

    reward_model = v8.CloudRadiationV8CleanDirect(**reward_ckpt["model_kwargs"]).to(device)
    reward_model.load_state_dict(strip_prefix(reward_ckpt["model_state"]))
    v9.enable_frozen_lstm_input_grads(reward_model)

    frames = int(gen_ckpt.get("frames", image.shape[1]))
    in_channels = int(gen_ckpt.get("in_channels", image.shape[2]))
    loss_scale = float(gen_ckpt.get("loss_scale", reward_ckpt.get("model_kwargs", {}).get("loss_scale", 300.0)))
    blob_channels = int(gargs.get("blob_channels", gen_ckpt.get("blob_channels", 3)))

    generator = v9.CloudGeneratorV9ZeroBlobs(
        frames=frames,
        in_channels=in_channels,
        context_dim=8,
        blob_channels=blob_channels,
        base=int(gargs.get("base_channels", 48)),
        cond_dim=int(gargs.get("cond_dim", 32)),
        loss_scale=loss_scale,
    ).to(device)
    generator.load_state_dict(strip_prefix(gen_ckpt["model_state"]))
    generator.eval()

    raw_mean, raw_std = v9.get_cloud_raw_stats(v8, x_norm, raw_names, cloud_names, device)

    # Use the same corruption/generation functions as training.
    with torch.no_grad():
        corrupted, keep_mask, corr_info = v9.corrupt_cloud_input(image, gns, int(gen_ckpt.get("epoch", 999999)))
        blob = v9.random_blob_latent(
            image.size(0), image.size(1), image.size(-2), image.size(-1),
            blob_channels,
            int(gargs.get("num_blobs", 9)),
            float(gargs.get("min_blob_radius", 0.04)),
            float(gargs.get("max_blob_radius", 0.18)),
            device,
        )
        generated = generator(corrupted, context, clear, target, actual, blob)
        cloud_generated = v9.derive_cloud_features_from_image(generated, cloud, cloud_names, raw_mean, raw_std)
        cloud_corrupted = v9.derive_cloud_features_from_image(corrupted, cloud, cloud_names, raw_mean, raw_std)

        original_pred = v9.reward_forward(reward_model, image, cloud, context, clear)["loss_wm2"]
        corrupted_pred_original_scalars = v9.reward_forward(reward_model, corrupted, cloud, context, clear)["loss_wm2"]
        corrupted_pred_derived_scalars = v9.reward_forward(reward_model, corrupted, cloud_corrupted, context, clear)["loss_wm2"]
        generated_pred = v9.reward_forward(reward_model, generated, cloud_generated, context, clear)["loss_wm2"]

    orig_np = image[0].detach().cpu().float().numpy()
    corr_np = corrupted[0].detach().cpu().float().numpy()
    gen_np = generated[0].detach().cpu().float().numpy()

    channel_names = CHANNEL_NAMES_8 if orig_np.shape[1] == 8 else [f"channel_{i:02d}" for i in range(orig_np.shape[1])]
    channels = parse_channels(args.channels, orig_np.shape[1])

    sheets = []
    sheets_dir = out_dir / "comparison_sheets"
    for ch in channels:
        cname = channel_names[ch] if ch < len(channel_names) else f"channel_{ch:02d}"
        p = sheets_dir / f"comparison_channel_{ch:02d}_{cname}.png"
        make_channel_sheet(orig_np, corr_np, gen_np, ch, cname, p)
        sheets.append(str(p))

    rows = []
    rows += image_stats("original", orig_np, channel_names)
    rows += image_stats("corrupted_input", corr_np, channel_names)
    rows += image_stats("generated_output", gen_np, channel_names)
    write_csv(out_dir / "image_channel_stats.csv", rows)

    drows = []
    drows += diff_stats("corrupted_minus_original", orig_np, corr_np, channel_names)
    drows += diff_stats("generated_minus_corrupted", corr_np, gen_np, channel_names)
    drows += diff_stats("generated_minus_original", orig_np, gen_np, channel_names)
    write_csv(out_dir / "image_diff_stats.csv", drows)

    # Raw derived scalar comparison.
    raw_orig = (cloud * raw_std + raw_mean)[0].detach().cpu().numpy()
    raw_corr = (cloud_corrupted * raw_std + raw_mean)[0].detach().cpu().numpy()
    raw_gen = (cloud_generated * raw_std + raw_mean)[0].detach().cpu().numpy()
    scalar_rows = []
    for frame in range(raw_orig.shape[0]):
        for j, name in enumerate(cloud_names):
            scalar_rows.append({
                "frame": frame,
                "feature": name,
                "original_raw": float(raw_orig[frame, j]),
                "corrupted_derived_raw": float(raw_corr[frame, j]),
                "generated_derived_raw": float(raw_gen[frame, j]),
                "generated_minus_original": float(raw_gen[frame, j] - raw_orig[frame, j]),
                "generated_minus_corrupted": float(raw_gen[frame, j] - raw_corr[frame, j]),
            })
    write_csv(out_dir / "derived_cloud_scalar_features.csv", scalar_rows)

    target_f = float(target.item())
    op = float(original_pred.item())
    cp_o = float(corrupted_pred_original_scalars.item())
    cp_d = float(corrupted_pred_derived_scalars.item())
    gp = float(generated_pred.item())

    result = {
        "what_this_tests": "V9 ZERO_BLOBS best_generator checkpoint: corrupted/zero cloud image + static context + target radiation -> generated 4-frame cloud tensor.",
        "split": args.split,
        "cleaned_dataset_len": len(ds),
        "sample_index": idx,
        "sample_id": str(item["sample_id"]),
        "location": str(item["location"]),
        "anchor": str(item["anchor"]),
        "actual_dataset_loss_wm2": float(actual.item()),
        "clear_sky_wm2": float(clear.item()),
        "target_loss_wm2": target_f,
        **target_meta,
        "checkpoint": {
            "generator_checkpoint": str(args.generator_checkpoint),
            "generator_epoch": int(gen_ckpt.get("epoch", -1)),
            "generator_architecture": gen_ckpt.get("architecture", "unknown"),
            "reward_checkpoint": str(reward_path),
        },
        "generation_setup": {
            "seed": int(args.seed),
            "frames": int(frames),
            "in_channels": int(in_channels),
            "blob_channels": int(blob_channels),
            "num_blobs": int(gargs.get("num_blobs", 9)),
            "min_blob_radius": float(gargs.get("min_blob_radius", 0.04)),
            "max_blob_radius": float(gargs.get("max_blob_radius", 0.18)),
            "corruption": corr_info,
            "keep_ratio_start": float(gargs.get("keep_ratio_start", -1)),
            "keep_ratio_end": float(gargs.get("keep_ratio_end", -1)),
            "zero_start_prob": float(gargs.get("zero_start_prob", -1)),
            "noise_start_prob": float(gargs.get("noise_start_prob", -1)),
            "noise_fill_ratio": float(gargs.get("noise_fill_ratio", -1)),
            "degrade_frames": str(gargs.get("degrade_frames", "unknown")),
        },
        "predictions": {
            "original_pred_loss_wm2": op,
            "corrupted_pred_original_scalars_wm2": cp_o,
            "corrupted_pred_derived_scalars_wm2": cp_d,
            "generated_pred_loss_wm2": gp,
        },
        "errors_to_target": {
            "original_abs_error_wm2": abs(op - target_f),
            "corrupted_original_scalars_abs_error_wm2": abs(cp_o - target_f),
            "corrupted_derived_scalars_abs_error_wm2": abs(cp_d - target_f),
            "generated_abs_error_wm2": abs(gp - target_f),
        },
        "improvements": {
            "generated_vs_corrupted_derived_improvement_wm2": abs(cp_d - target_f) - abs(gp - target_f),
            "generated_vs_original_improvement_wm2": abs(op - target_f) - abs(gp - target_f),
        },
        "image_l1": {
            "corrupted_vs_original_all": float(np.mean(np.abs(corr_np - orig_np))),
            "generated_vs_corrupted_all": float(np.mean(np.abs(gen_np - corr_np))),
            "generated_vs_original_all": float(np.mean(np.abs(gen_np - orig_np))),
            "generated_vs_corrupted_last": float(np.mean(np.abs(gen_np[-1] - corr_np[-1]))),
        },
        "outputs": {
            "result_json": str(out_dir / "result.json"),
            "comparison_sheets": sheets,
            "image_channel_stats_csv": str(out_dir / "image_channel_stats.csv"),
            "image_diff_stats_csv": str(out_dir / "image_diff_stats.csv"),
            "derived_cloud_scalar_features_csv": str(out_dir / "derived_cloud_scalar_features.csv"),
        },
    }

    warnings = []
    if abs(gp - target_f) > abs(cp_d - target_f):
        warnings.append("Generated output is worse than corrupted input under derived scalars.")
    if result["image_l1"]["generated_vs_corrupted_all"] < 0.01 and abs(gp - cp_d) > 10.0:
        warnings.append("Reward changed but image barely changed.")
    if np.mean(gen_np < 0.02) > 0.85:
        warnings.append("Generated output is mostly black.")
    if np.mean((gen_np > 0.35) & (gen_np < 0.65)) > 0.70:
        warnings.append("Generated output is mostly mid-gray; may be blurred/noisy.")
    result["warnings"] = warnings

    write_json(out_dir / "result.json", result)

    if args.save_npz:
        np.savez_compressed(
            out_dir / "arrays.npz",
            original=orig_np,
            corrupted=corr_np,
            generated=gen_np,
            keep_mask=keep_mask[0].detach().cpu().float().numpy(),
            blob_latent=blob[0].detach().cpu().float().numpy(),
        )

    print("DONE")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
