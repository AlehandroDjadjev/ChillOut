#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

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


def import_v12():
    names = [
        "train_cloud_radiation_generator_v12_RESIDUAL_EDITOR",
        "train_cloud_radiation_generator_v12_residual_editor",
    ]
    last = None
    for name in names:
        try:
            return __import__(name, fromlist=["dummy"])
        except Exception as exc:
            last = exc
    raise RuntimeError(
        "Could not import train_cloud_radiation_generator_v12_RESIDUAL_EDITOR.py. "
        "Put this diagnostic next to that file and train_cloud_radiation_bottom_v8_CLEAN_DIRECT.py. "
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


def parse_channels(s: str, c: int) -> List[int]:
    if str(s).strip().lower() == "all":
        return list(range(c))
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        i = int(x)
        if 0 <= i < c:
            out.append(i)
    return out


def ns_from_dict(d: Dict[str, Any]):
    class A:
        pass
    a = A()
    for k, v in d.items():
        setattr(a, k, v)
    return a


def target_from_args(actual: torch.Tensor, clear: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if args.target_loss_wm2 is not None:
        target = torch.tensor([[float(args.target_loss_wm2)]], dtype=torch.float32, device=actual.device)
        meta["target_mode"] = "absolute_loss_wm2"
    else:
        target = actual + float(args.target_delta_wm2)
        meta["target_mode"] = "delta_loss_wm2"
        meta["requested_delta_wm2"] = float(args.target_delta_wm2)

    unclamped = target.clone()
    target = torch.minimum(target.clamp(min=0.0), clear.clamp_min(0.0))
    meta["target_was_clamped"] = bool((target != unclamped).any().item())
    return target, meta


def to_gray_img(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")


def signed_img(arr: np.ndarray, scale: float = 1.0) -> Image.Image:
    x = np.asarray(arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    pos = np.clip(x / scale, 0.0, 1.0)
    neg = np.clip(-x / scale, 0.0, 1.0)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (pos * 255).astype(np.uint8)
    rgb[..., 2] = (neg * 255).astype(np.uint8)
    rgb[..., 1] = (np.clip(1.0 - np.maximum(pos, neg), 0.0, 1.0) * 40).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def signed_diff_img(before: np.ndarray, after: np.ndarray, scale: float = 0.35) -> Image.Image:
    return signed_img(np.asarray(after, dtype=np.float32) - np.asarray(before, dtype=np.float32), scale=scale)


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str) -> None:
    draw.text(xy, text, fill=(255, 255, 255))


def make_channel_sheet(
    original: np.ndarray,
    input_img: np.ndarray,
    allowed: np.ndarray,
    gate: np.ndarray,
    residual: np.ndarray,
    output: np.ndarray,
    ch: int,
    cname: str,
    out_path: Path,
) -> None:
    """Arrays are [T,C,H,W], allowed/gate are [T,1,H,W], residual is [T,C,H,W]."""
    t, c, h, w = original.shape
    label_w = 180
    top_h = 34
    gap = 8
    row_gap = 18

    rows = [
        ("ORIGINAL", [to_gray_img(original[i, ch]) for i in range(t)]),
        ("INPUT", [to_gray_img(input_img[i, ch]) for i in range(t)]),
        ("ALLOWED_MASK", [to_gray_img(allowed[i, 0]) for i in range(t)]),
        ("LEARNED_GATE", [to_gray_img(gate[i, 0]) for i in range(t)]),
        ("RESIDUAL_DIR", [signed_img(residual[i, ch], scale=1.0) for i in range(t)]),
        ("OUTPUT", [to_gray_img(output[i, ch]) for i in range(t)]),
        ("OUT-INPUT", [signed_diff_img(input_img[i, ch], output[i, ch]) for i in range(t)]),
        ("OUT-ORIG", [signed_diff_img(original[i, ch], output[i, ch]) for i in range(t)]),
    ]

    sheet_w = label_w + t * w + (t - 1) * gap
    sheet_h = top_h + len(rows) * h + (len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (8, 8), f"v12 residual editor | ch {ch}: {cname} | red=increase blue=decrease")

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
                "white_frac_ge_0p35": float(np.mean(x >= 0.35)),
                "binary_white_frac_ge_0p50": float(np.mean(x >= 0.50)),
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


def mask_stats(name: str, arr: np.ndarray) -> Dict[str, Any]:
    return {
        "name": name,
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "nonzero_frac_gt_0p01": float(np.mean(arr > 0.01)),
        "white_frac_ge_0p50": float(np.mean(arr >= 0.5)),
        "per_frame_mean": [float(arr[i, 0].mean()) for i in range(arr.shape[0])],
        "per_frame_white_frac_ge_0p50": [float((arr[i, 0] >= 0.5).mean()) for i in range(arr.shape[0])],
    }


def make_input(v12, image: torch.Tensor, args, gns):
    if args.input_mode == "dropout":
        return v12.white_dropout_input(image, gns)
    if args.input_mode == "full":
        dropped = torch.zeros(image.size(0), image.size(1), 1, image.size(-2), image.size(-1), device=image.device)
        return image.clone(), {"mode": "FULL_UNTOUCHED_IMAGE"}, dropped
    raise ValueError(args.input_mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--generator-checkpoint", required=True)
    ap.add_argument("--reward-checkpoint", default=None, help="Optional override. Defaults to checkpoint stored in generator.")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--sample-index", type=int, default=50)
    ap.add_argument("--sample-id", default=None)

    ap.add_argument("--input-mode", choices=["dropout", "full"], default="dropout")
    ap.add_argument("--target-loss-wm2", type=float, default=None)
    ap.add_argument("--target-delta-wm2", type=float, default=120.0)

    # Diagnostic overrides.
    ap.add_argument("--white-drop-prob", type=float, default=None)
    ap.add_argument("--white-threshold", type=float, default=None)
    ap.add_argument("--num-blobs", type=int, default=None)
    ap.add_argument("--min-blob-radius", type=float, default=None)
    ap.add_argument("--max-blob-radius", type=float, default=None)
    ap.add_argument("--support-dilation-kernel", type=int, default=None)
    ap.add_argument("--max-delta", type=float, default=None)

    # Optional target sweep.
    ap.add_argument("--target-losses", default=None, help="Comma-separated absolute target losses, e.g. 100,250,450,650")

    ap.add_argument("--channels", default="0,1,2,4,5,7")
    ap.add_argument("--out-dir", default="diagnostics/generator_v12_residual_editor_diag")
    ap.add_argument("--min-clear-wm2", type=float, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--save-npz", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    v12 = import_v12()
    v8 = v12.import_v8_reward()

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    gen_ckpt = torch.load(args.generator_checkpoint, map_location="cpu")
    gargs = dict(gen_ckpt.get("args", {}))

    # Apply diagnostic overrides.
    for key in [
        "white_drop_prob",
        "white_threshold",
        "num_blobs",
        "min_blob_radius",
        "max_blob_radius",
        "support_dilation_kernel",
        "max_delta",
    ]:
        val = getattr(args, key)
        if val is not None:
            gargs[key] = val
    gns = ns_from_dict(gargs)

    reward_path = args.reward_checkpoint or gen_ckpt.get("reward_checkpoint")
    if reward_path is None:
        raise RuntimeError("Need --reward-checkpoint or reward_checkpoint stored in generator checkpoint.")

    reward_ckpt = torch.load(reward_path, map_location="cpu")
    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_names = list(reward_ckpt["raw_feature_names"])
    cloud_names = list(reward_ckpt["cloud_feature_names"])
    x_norm = v8.base.Normalizer.from_state_dict(reward_ckpt["normalizer"])
    ckargs = reward_ckpt.get("args", {})
    cleaner = v12.make_clean_args(ckargs, override_min_clear=args.min_clear_wm2)

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
        raise RuntimeError("Dataset split is empty after cleaning.")

    idx = int(args.sample_index)
    if args.sample_id is not None:
        found = None
        for i in range(len(ds)):
            if str(ds[i]["sample_id"]) == str(args.sample_id):
                found = i
                break
        if found is None:
            raise RuntimeError(f"sample-id {args.sample_id!r} not found")
        idx = found
    if idx < 0 or idx >= len(ds):
        raise IndexError(f"sample-index {idx} outside split length {len(ds)}")

    item = ds[idx]
    image = item["image"].unsqueeze(0).to(device)
    cloud = item["cloud_features"].unsqueeze(0).to(device)
    context = item["context_features"].unsqueeze(0).to(device)
    clear = item["clear_wm2"].view(1, 1).to(device).float()
    actual = item["target_loss_wm2"].view(1, 1).to(device).float()
    target, target_meta = target_from_args(actual, clear, args)

    reward_model = v8.CloudRadiationV8CleanDirect(**reward_ckpt["model_kwargs"]).to(device)
    reward_model.load_state_dict(strip_prefix(reward_ckpt["model_state"]))
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    frames = int(gen_ckpt.get("frames", image.shape[1]))
    in_channels = int(gen_ckpt.get("in_channels", image.shape[2]))
    loss_scale = float(gen_ckpt.get("loss_scale", reward_ckpt.get("model_kwargs", {}).get("loss_scale", 300.0)))

    model = v12.CloudResidualEditorV12(
        frames=frames,
        in_channels=in_channels,
        context_dim=8,
        blob_channels=int(gargs.get("blob_channels", 3)),
        base=int(gargs.get("base_channels", 48)),
        cond_dim=int(gargs.get("cond_dim", 48)),
        dropout=float(gargs.get("dropout", 0.05)),
        loss_scale=loss_scale,
    ).to(device)
    model.load_state_dict(strip_prefix(gen_ckpt["model_state"]))
    model.eval()

    raw_mean, raw_std = v12.get_cloud_raw_stats(v8, x_norm, raw_names, cloud_names, device)

    with torch.no_grad():
        input_img, input_info, dropped = make_input(v12, image, args, gns)
        blob = v12.random_blob_latent(
            image.size(0),
            image.size(1),
            image.size(-2),
            image.size(-1),
            int(gargs.get("blob_channels", 3)),
            int(gargs.get("num_blobs", 4)),
            float(gargs.get("min_blob_radius", 0.035)),
            float(gargs.get("max_blob_radius", 0.105)),
            device,
        )
        allowed = v12.allowed_edit_mask(input_img, blob, gns)
        output, aux = model(input_img, context, clear, target, actual, blob, allowed, float(gargs.get("max_delta", 0.45)))

        cloud_input = v12.derive_cloud_features_from_image(input_img, cloud, cloud_names, raw_mean, raw_std)
        cloud_output = v12.derive_cloud_features_from_image(output, cloud, cloud_names, raw_mean, raw_std)

        original_pred = v12.reward_forward(reward_model, image, cloud, context, clear)["loss_wm2"]
        input_pred = v12.reward_forward(reward_model, input_img, cloud_input, context, clear)["loss_wm2"]
        output_pred = v12.reward_forward(reward_model, output, cloud_output, context, clear)["loss_wm2"]

    orig_np = image[0].detach().cpu().float().numpy()
    input_np = input_img[0].detach().cpu().float().numpy()
    allowed_np = allowed[0].detach().cpu().float().numpy()
    gate_np = aux["gate"][0].detach().cpu().float().numpy()
    residual_np = aux["residual"][0].detach().cpu().float().numpy()
    output_np = output[0].detach().cpu().float().numpy()
    dropped_np = dropped[0].detach().cpu().float().numpy()

    channel_names = CHANNEL_NAMES_8 if orig_np.shape[1] == 8 else [f"channel_{i:02d}" for i in range(orig_np.shape[1])]
    view_channels = parse_channels(args.channels, orig_np.shape[1])

    sheets = []
    for ch in view_channels:
        cname = channel_names[ch] if ch < len(channel_names) else f"channel_{ch:02d}"
        p = out_dir / "comparison_sheets" / f"comparison_channel_{ch:02d}_{cname}.png"
        make_channel_sheet(orig_np, input_np, allowed_np, gate_np, residual_np, output_np, ch, cname, p)
        sheets.append(str(p))

    rows = []
    rows += image_stats("original", orig_np, channel_names)
    rows += image_stats("input", input_np, channel_names)
    rows += image_stats("allowed_mask", np.repeat(allowed_np, orig_np.shape[1], axis=1), channel_names)
    rows += image_stats("gate", np.repeat(gate_np, orig_np.shape[1], axis=1), channel_names)
    rows += image_stats("output", output_np, channel_names)
    write_csv(out_dir / "image_channel_stats.csv", rows)

    drows = []
    drows += diff_stats("input_minus_original", orig_np, input_np, channel_names)
    drows += diff_stats("output_minus_input", input_np, output_np, channel_names)
    drows += diff_stats("output_minus_original", orig_np, output_np, channel_names)
    write_csv(out_dir / "image_diff_stats.csv", drows)

    raw_orig = (cloud * raw_std + raw_mean)[0].detach().cpu().numpy()
    raw_input = (cloud_input * raw_std + raw_mean)[0].detach().cpu().numpy()
    raw_output = (cloud_output * raw_std + raw_mean)[0].detach().cpu().numpy()

    scalar_rows = []
    for frame in range(raw_orig.shape[0]):
        for j, name in enumerate(cloud_names):
            scalar_rows.append({
                "frame": frame,
                "feature": name,
                "original_raw": float(raw_orig[frame, j]),
                "input_derived_raw": float(raw_input[frame, j]),
                "output_derived_raw": float(raw_output[frame, j]),
                "output_minus_original": float(raw_output[frame, j] - raw_orig[frame, j]),
                "output_minus_input": float(raw_output[frame, j] - raw_input[frame, j]),
            })
    write_csv(out_dir / "derived_cloud_scalar_features.csv", scalar_rows)

    target_f = float(target.item())
    op = float(original_pred.item())
    ip = float(input_pred.item())
    gp = float(output_pred.item())

    result = {
        "what_this_tests": "V12 residual editor diagnostic: original/input/allowed-mask/gate/residual/output plus frozen bottom reward scores.",
        "sample": {
            "split": args.split,
            "sample_index": idx,
            "sample_id": str(item["sample_id"]),
            "location": str(item["location"]),
            "anchor": str(item["anchor"]),
            "actual_dataset_loss_wm2": float(actual.item()),
            "clear_sky_wm2": float(clear.item()),
        },
        "target_loss_wm2": target_f,
        **target_meta,
        "checkpoint": {
            "generator_checkpoint": str(args.generator_checkpoint),
            "generator_epoch": int(gen_ckpt.get("epoch", -1)),
            "generator_architecture": str(gen_ckpt.get("architecture", "unknown")),
            "reward_checkpoint": str(reward_path),
            "max_delta": float(gargs.get("max_delta", 0.45)),
            "num_blobs": int(gargs.get("num_blobs", -1)),
            "support_dilation_kernel": int(gargs.get("support_dilation_kernel", -1)),
        },
        "input_setup": {
            "input_mode": args.input_mode,
            "input_info": input_info,
            "dropped_mask_mean": float(dropped_np.mean()),
            "dropped_mask_white_frac_ge_0p50": float((dropped_np >= 0.5).mean()),
        },
        "predictions": {
            "original_pred_loss_wm2": op,
            "input_pred_loss_wm2": ip,
            "output_pred_loss_wm2": gp,
        },
        "errors_to_target": {
            "original_abs_error_wm2": abs(op - target_f),
            "input_abs_error_wm2": abs(ip - target_f),
            "output_abs_error_wm2": abs(gp - target_f),
        },
        "improvements": {
            "output_vs_input_improvement_wm2": abs(ip - target_f) - abs(gp - target_f),
            "output_vs_original_improvement_wm2": abs(op - target_f) - abs(gp - target_f),
            "output_pred_minus_input_pred_wm2": gp - ip,
            "output_pred_minus_original_pred_wm2": gp - op,
        },
        "edit_stats": {
            "allowed_mask": mask_stats("allowed_mask", allowed_np),
            "gate": mask_stats("gate", gate_np),
            "residual_abs_mean": float(np.abs(residual_np).mean()),
            "residual_abs_max": float(np.abs(residual_np).max()),
            "image_l1_input_vs_original": float(np.abs(input_np - orig_np).mean()),
            "image_l1_output_vs_input": float(np.abs(output_np - input_np).mean()),
            "image_l1_output_vs_original": float(np.abs(output_np - orig_np).mean()),
            "output_white_frac_ge_0p50": float((output_np >= 0.5).mean()),
            "input_white_frac_ge_0p50": float((input_np >= 0.5).mean()),
            "original_white_frac_ge_0p50": float((orig_np >= 0.5).mean()),
        },
        "outputs": {
            "result_json": str(out_dir / "result.json"),
            "comparison_sheets": sheets,
            "image_channel_stats_csv": str(out_dir / "image_channel_stats.csv"),
            "image_diff_stats_csv": str(out_dir / "image_diff_stats.csv"),
            "derived_cloud_scalar_features_csv": str(out_dir / "derived_cloud_scalar_features.csv"),
        },
    }

    # Optional target sweep: same input/blob, different absolute target losses.
    if args.target_losses:
        sweep_rows = []
        for raw in str(args.target_losses).split(","):
            raw = raw.strip()
            if not raw:
                continue
            tval = float(raw)
            ttarget = torch.tensor([[tval]], dtype=torch.float32, device=device)
            ttarget = torch.minimum(ttarget.clamp(min=0.0), clear.clamp_min(0.0))
            with torch.no_grad():
                tout, taux = model(input_img, context, clear, ttarget, actual, blob, allowed, float(gargs.get("max_delta", 0.45)))
                tcloud = v12.derive_cloud_features_from_image(tout, cloud, cloud_names, raw_mean, raw_std)
                tpred = v12.reward_forward(reward_model, tout, tcloud, context, clear)["loss_wm2"]
            sweep_rows.append({
                "target_loss_wm2": float(ttarget.item()),
                "pred_loss_wm2": float(tpred.item()),
                "abs_error_wm2": abs(float(tpred.item()) - float(ttarget.item())),
                "image_l1_vs_input": float((tout - input_img).abs().mean().item()),
                "gate_mean": float(taux["gate"].mean().item()),
                "output_white_frac_ge_0p50": float((tout.detach().cpu().numpy() >= 0.5).mean()),
            })
        write_csv(out_dir / "target_sweep.csv", sweep_rows)
        result["outputs"]["target_sweep_csv"] = str(out_dir / "target_sweep.csv")
        result["target_sweep"] = sweep_rows

    warnings = []
    if result["improvements"]["output_vs_input_improvement_wm2"] < 0:
        warnings.append("Output is worse than diagnostic input.")
    if result["improvements"]["output_vs_input_improvement_wm2"] < 5:
        warnings.append("Output barely improves over input; v12 policy/reward gradient may be too weak for this sample.")
    if result["edit_stats"]["gate"]["mean"] < 0.03:
        warnings.append("Gate is nearly closed; model is mostly refusing to edit.")
    if result["edit_stats"]["gate"]["mean"] > 0.70:
        warnings.append("Gate is very open; model may be over-editing.")
    if result["edit_stats"]["allowed_mask"]["white_frac_ge_0p50"] > 0.80:
        warnings.append("Allowed mask covers most of image; edit space is too broad.")
    if result["edit_stats"]["image_l1_output_vs_input"] < 0.005:
        warnings.append("Output is almost unchanged from input.")
    if result["edit_stats"]["output_white_frac_ge_0p50"] > 0.80:
        warnings.append("Output is mostly white/overfilled.")
    result["warnings"] = warnings

    write_json(out_dir / "result.json", result)

    if args.save_npz:
        np.savez_compressed(
            out_dir / "arrays.npz",
            original=orig_np,
            input=input_np,
            allowed_mask=allowed_np,
            gate=gate_np,
            residual=residual_np,
            output=output_np,
            dropped_mask=dropped_np,
        )

    print("DONE")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
