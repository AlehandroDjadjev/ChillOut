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
import torch.nn.functional as F


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


def import_v13():
    names = [
        "train_cloud_template_selector_v13",
        "train_cloud_template_selector_V13",
    ]
    last = None
    for name in names:
        try:
            return __import__(name, fromlist=["dummy"])
        except Exception as exc:
            last = exc
    raise RuntimeError(
        "Could not import train_cloud_template_selector_v13.py. "
        "Put this diagnostic next to train_cloud_template_selector_v13.py and "
        "train_cloud_radiation_bottom_v8_CLEAN_DIRECT.py. "
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
    input_img: np.ndarray,
    selector_template: np.ndarray,
    selector_output: np.ndarray,
    oracle_template: np.ndarray,
    oracle_output: np.ndarray,
    ch: int,
    cname: str,
    out_path: Path,
) -> None:
    t, c, h, w = original.shape
    label_w = 180
    top_h = 34
    gap = 8
    row_gap = 18

    rows = [
        ("ORIGINAL", [to_gray_img(original[i, ch]) for i in range(t)]),
        ("INPUT", [to_gray_img(input_img[i, ch]) for i in range(t)]),
        ("SELECTED_TEMPLATE", [to_gray_img(selector_template[i, ch]) for i in range(t)]),
        ("SELECTOR_OUTPUT", [to_gray_img(selector_output[i, ch]) for i in range(t)]),
        ("ORACLE_TEMPLATE", [to_gray_img(oracle_template[i, ch]) for i in range(t)]),
        ("ORACLE_OUTPUT", [to_gray_img(oracle_output[i, ch]) for i in range(t)]),
        ("SELECTOR-INPUT", [signed_diff_img(input_img[i, ch], selector_output[i, ch]) for i in range(t)]),
        ("ORACLE-INPUT", [signed_diff_img(input_img[i, ch], oracle_output[i, ch]) for i in range(t)]),
        ("ORACLE-SELECTOR", [signed_diff_img(selector_output[i, ch], oracle_output[i, ch]) for i in range(t)]),
    ]

    sheet_w = label_w + t * w + (t - 1) * gap
    sheet_h = top_h + len(rows) * h + (len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (8, 8), f"v13 selector diagnostic | ch {ch}: {cname} | red=increase blue=decrease")

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


def make_input(v13, image: torch.Tensor, args, gns):
    if args.input_mode == "dropout":
        return v13.white_dropout_input(image, gns)
    if args.input_mode == "full":
        dropped = torch.zeros(image.size(0), image.size(1), 1, image.size(-2), image.size(-1), device=image.device)
        return image.clone(), dropped, {"mode": "FULL_UNTOUCHED_IMAGE"}
    raise ValueError(args.input_mode)


def score_output(v13, reward_model, output, cloud_base, cloud_names, raw_mean, raw_std, context, clear, target) -> Dict[str, Any]:
    cloud_out = v13.derive_cloud_features_from_image(output, cloud_base, cloud_names, raw_mean, raw_std)
    pred = v13.reward_forward(reward_model, output, cloud_out, context, clear)["loss_wm2"]
    return {
        "pred": pred,
        "abs_error": (pred - target).abs(),
        "cloud_features": cloud_out,
    }


def coverage(v13, image: torch.Tensor, cov_channels: List[int]) -> float:
    return float(v13.batch_coverage(image, cov_channels).mean().detach().cpu())


def brute_force_oracle(
    v13,
    reward_model,
    input_img: torch.Tensor,
    original_img: torch.Tensor,
    cloud: torch.Tensor,
    context: torch.Tensor,
    clear: torch.Tensor,
    target: torch.Tensor,
    codebook: torch.Tensor,
    codebook_meta: List[Dict[str, Any]],
    mode_names: List[str],
    raw_mean,
    raw_std,
    cloud_names,
    edit_channels: List[int],
    cov_channels: List[int],
    args,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    rows = []
    best_row = None
    best_output = None
    best_template = None

    k_total = codebook.size(0)
    modes_total = len(mode_names)

    # batch templates to avoid thousands of tiny reward calls
    for start in range(0, k_total, args.oracle_batch_size):
        end = min(k_total, start + args.oracle_batch_size)
        templates = codebook[start:end].to(input_img.device)  # [K,T,C,H,W]
        kk = templates.size(0)

        # repeat input tensors for K candidates
        inp_b = input_img.expand(kk, -1, -1, -1, -1).contiguous()
        orig_b = original_img.expand(kk, -1, -1, -1, -1).contiguous()
        cloud_b = cloud.expand(kk, -1, -1).contiguous()
        ctx_b = context.expand(kk, -1).contiguous()
        clear_b = clear.expand(kk, -1).contiguous()
        target_b = target.expand(kk, -1).contiguous()

        for mode_i, mode_name in enumerate(mode_names):
            mode_w = torch.zeros(kk, modes_total, device=input_img.device)
            mode_w[:, mode_i] = 1.0
            out = v13.apply_modes(inp_b, templates, mode_w, edit_channels)

            cloud_out = v13.derive_cloud_features_from_image(out, cloud_b, cloud_names, raw_mean, raw_std)
            pred = v13.reward_forward(reward_model, out, cloud_out, ctx_b, clear_b)["loss_wm2"]
            err = (pred - target_b).abs().view(-1)
            l1 = (out - inp_b).abs().mean(dim=(1, 2, 3, 4))
            cov = v13.batch_coverage(out, cov_channels).view(-1)
            max_coverage = 0.70 if args.max_coverage is None else float(args.max_coverage)
            cov_pen = torch.relu(cov - max_coverage)
            score = err + args.penalty_l1 * l1 + args.penalty_coverage * cov_pen

            for j in range(kk):
                global_i = start + j
                meta = codebook_meta[global_i] if global_i < len(codebook_meta) else {}
                row = {
                    "template_index": int(global_i),
                    "mode_index": int(mode_i),
                    "mode": str(mode_name),
                    "template_sample_id": str(meta.get("sample_id", "")),
                    "template_location": str(meta.get("location", "")),
                    "template_anchor": str(meta.get("anchor", "")),
                    "template_coverage": float(meta.get("coverage", float("nan"))),
                    "pred_loss_wm2": float(pred[j].item()),
                    "target_loss_wm2": float(target.item()),
                    "abs_error_wm2": float(err[j].item()),
                    "score": float(score[j].item()),
                    "image_l1_vs_input": float(l1[j].item()),
                    "coverage": float(cov[j].item()),
                    "coverage_base": coverage(v13, input_img, cov_channels),
                }
                rows.append(row)
                if best_row is None or row["score"] < best_row["score"]:
                    best_row = row
                    best_output = out[j:j+1].detach().clone()
                    best_template = templates[j:j+1].detach().clone()

    rows.sort(key=lambda r: r["score"])
    return best_row, best_output, best_template, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--selector-checkpoint", required=True)
    ap.add_argument("--reward-checkpoint", default=None, help="Optional override. Defaults to checkpoint stored in selector.")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--sample-index", type=int, default=50)
    ap.add_argument("--sample-id", default=None)

    ap.add_argument("--input-mode", choices=["dropout", "full"], default="dropout")
    ap.add_argument("--target-loss-wm2", type=float, default=None)
    ap.add_argument("--target-delta-wm2", type=float, default=120.0)

    # Runtime overrides.
    ap.add_argument("--white-drop-prob", type=float, default=None)
    ap.add_argument("--white-threshold", type=float, default=None)
    ap.add_argument("--eval-tau", type=float, default=None)

    # Oracle scoring.
    ap.add_argument("--run-oracle", action="store_true", default=True)
    ap.add_argument("--oracle-batch-size", type=int, default=32)
    ap.add_argument("--penalty-l1", type=float, default=25.0)
    ap.add_argument("--penalty-coverage", type=float, default=80.0)
    ap.add_argument("--max-coverage", type=float, default=None)

    ap.add_argument("--channels", default="0,1,2,4,5,7")
    ap.add_argument("--out-dir", default="diagnostics/cloud_template_selector_v13_diag")
    ap.add_argument("--min-clear-wm2", type=float, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--save-npz", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    v13 = import_v13()
    v8 = v13.import_v8_reward()

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    sel_ckpt = torch.load(args.selector_checkpoint, map_location="cpu")
    train_args = dict(sel_ckpt.get("args", {}))

    # Apply selected diagnostic overrides.
    if args.white_drop_prob is not None:
        train_args["white_drop_prob"] = float(args.white_drop_prob)
    if args.white_threshold is not None:
        train_args["white_threshold"] = float(args.white_threshold)
    if args.eval_tau is not None:
        train_args["eval_tau"] = float(args.eval_tau)
    if args.max_coverage is not None:
        train_args["max_coverage"] = float(args.max_coverage)
    else:
        args.max_coverage = float(train_args.get("max_coverage", 0.70))

    gns = ns_from_dict(train_args)

    reward_path = args.reward_checkpoint or sel_ckpt.get("reward_checkpoint")
    if reward_path is None:
        raise RuntimeError("Need --reward-checkpoint or reward_checkpoint stored in selector checkpoint.")

    reward_ckpt = torch.load(reward_path, map_location="cpu")
    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_names = list(reward_ckpt["raw_feature_names"])
    cloud_names = list(reward_ckpt["cloud_feature_names"])
    x_norm = v8.base.Normalizer.from_state_dict(reward_ckpt["normalizer"])
    ckargs = reward_ckpt.get("args", {})
    cleaner = v13.make_clean_args(ckargs, override_min_clear=args.min_clear_wm2)

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

    codebook = sel_ckpt["codebook"].to(device).float()
    codebook_meta = list(sel_ckpt.get("codebook_meta", []))
    mode_names = list(sel_ckpt.get("mode_names", v13.MODE_NAMES))

    frames = int(sel_ckpt.get("frames", image.shape[1]))
    in_channels = int(sel_ckpt.get("in_channels", image.shape[2]))
    loss_scale = float(sel_ckpt.get("loss_scale", reward_ckpt.get("model_kwargs", {}).get("loss_scale", 300.0)))

    selector = v13.CloudTemplateSelectorV13(
        frames=frames,
        in_channels=in_channels,
        codebook_size=codebook.size(0),
        num_modes=len(mode_names),
        context_dim=8,
        base=int(train_args.get("base_channels", 32)),
        hidden=int(train_args.get("hidden_dim", 192)),
        loss_scale=loss_scale,
    ).to(device)
    selector.load_state_dict(strip_prefix(sel_ckpt["model_state"]))
    selector.eval()

    reward_model = v8.CloudRadiationV8CleanDirect(**reward_ckpt["model_kwargs"]).to(device)
    reward_model.load_state_dict(strip_prefix(reward_ckpt["model_state"]))
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    raw_mean, raw_std = v13.get_cloud_raw_stats(v8, x_norm, raw_names, cloud_names, device)
    cov_channels = parse_channels(str(train_args.get("coverage_channels", "0,1,2")), image.size(2))
    edit_channels = parse_channels(str(train_args.get("edit_channels", "0,1,2,3,4,5,6,7")), image.size(2))
    view_channels = parse_channels(args.channels, image.size(2))

    with torch.no_grad():
        input_img, dropped, input_info = make_input(v13, image, args, gns)

        cloud_input = v13.derive_cloud_features_from_image(input_img, cloud, cloud_names, raw_mean, raw_std)
        input_pred = v13.reward_forward(reward_model, input_img, cloud_input, context, clear)["loss_wm2"]
        base_pred = v13.reward_forward(reward_model, image, cloud, context, clear)["loss_wm2"]

        inp_cov = v13.batch_coverage(input_img, cov_channels)
        template_logits, mode_logits = selector(input_img, context, clear, actual, target, input_pred, inp_cov)

        template_probs = F.softmax(template_logits / max(1e-6, float(train_args.get("eval_tau", 0.20))), dim=-1)
        mode_probs = F.softmax(mode_logits / max(1e-6, float(train_args.get("eval_tau", 0.20))), dim=-1)

        template_idx = int(template_probs.argmax(dim=-1).item())
        mode_idx = int(mode_probs.argmax(dim=-1).item())

        tw = F.one_hot(torch.tensor([template_idx], device=device), codebook.size(0)).float()
        mw = F.one_hot(torch.tensor([mode_idx], device=device), len(mode_names)).float()

        selector_template = torch.einsum("bk,ktchw->btchw", tw, codebook)
        selector_output = v13.apply_modes(input_img, selector_template, mw, edit_channels)

        selector_scored = score_output(v13, reward_model, selector_output, cloud, cloud_names, raw_mean, raw_std, context, clear, target)

    if args.run_oracle:
        oracle_row, oracle_output, oracle_template, oracle_rows = brute_force_oracle(
            v13=v13,
            reward_model=reward_model,
            input_img=input_img,
            original_img=image,
            cloud=cloud,
            context=context,
            clear=clear,
            target=target,
            codebook=codebook,
            codebook_meta=codebook_meta,
            mode_names=mode_names,
            raw_mean=raw_mean,
            raw_std=raw_std,
            cloud_names=cloud_names,
            edit_channels=edit_channels,
            cov_channels=cov_channels,
            args=args,
        )
        write_csv(out_dir / "oracle_candidate_scores.csv", oracle_rows)
    else:
        oracle_row = {
            "template_index": template_idx,
            "mode_index": mode_idx,
            "mode": mode_names[mode_idx],
            "pred_loss_wm2": float(selector_scored["pred"].item()),
            "target_loss_wm2": float(target.item()),
            "abs_error_wm2": float(selector_scored["abs_error"].item()),
            "score": float(selector_scored["abs_error"].item()),
            "image_l1_vs_input": float((selector_output - input_img).abs().mean().item()),
            "coverage": coverage(v13, selector_output, cov_channels),
        }
        oracle_output = selector_output.detach().clone()
        oracle_template = selector_template.detach().clone()
        write_csv(out_dir / "oracle_candidate_scores.csv", [oracle_row])

    orig_np = image[0].detach().cpu().float().numpy()
    input_np = input_img[0].detach().cpu().float().numpy()
    sel_template_np = selector_template[0].detach().cpu().float().numpy()
    sel_output_np = selector_output[0].detach().cpu().float().numpy()
    oracle_template_np = oracle_template[0].detach().cpu().float().numpy()
    oracle_output_np = oracle_output[0].detach().cpu().float().numpy()

    channel_names = CHANNEL_NAMES_8 if orig_np.shape[1] == 8 else [f"channel_{i:02d}" for i in range(orig_np.shape[1])]
    sheets = []
    for ch in view_channels:
        cname = channel_names[ch] if ch < len(channel_names) else f"channel_{ch:02d}"
        p = out_dir / "comparison_sheets" / f"comparison_channel_{ch:02d}_{cname}.png"
        make_channel_sheet(
            orig_np,
            input_np,
            sel_template_np,
            sel_output_np,
            oracle_template_np,
            oracle_output_np,
            ch,
            cname,
            p,
        )
        sheets.append(str(p))

    rows = []
    rows += image_stats("original", orig_np, channel_names)
    rows += image_stats("input", input_np, channel_names)
    rows += image_stats("selector_template", sel_template_np, channel_names)
    rows += image_stats("selector_output", sel_output_np, channel_names)
    rows += image_stats("oracle_template", oracle_template_np, channel_names)
    rows += image_stats("oracle_output", oracle_output_np, channel_names)
    write_csv(out_dir / "image_channel_stats.csv", rows)

    drows = []
    drows += diff_stats("input_minus_original", orig_np, input_np, channel_names)
    drows += diff_stats("selector_output_minus_input", input_np, sel_output_np, channel_names)
    drows += diff_stats("oracle_output_minus_input", input_np, oracle_output_np, channel_names)
    drows += diff_stats("oracle_minus_selector", sel_output_np, oracle_output_np, channel_names)
    write_csv(out_dir / "image_diff_stats.csv", drows)

    top_templates = []
    probs = template_probs[0].detach().cpu().numpy()
    top_idx = np.argsort(-probs)[: min(20, len(probs))]
    for ti in top_idx:
        meta = codebook_meta[int(ti)] if int(ti) < len(codebook_meta) else {}
        top_templates.append({
            "template_index": int(ti),
            "prob": float(probs[ti]),
            "sample_id": str(meta.get("sample_id", "")),
            "location": str(meta.get("location", "")),
            "anchor": str(meta.get("anchor", "")),
            "coverage": float(meta.get("coverage", float("nan"))),
        })
    write_csv(out_dir / "selector_top_templates.csv", top_templates)

    mode_prob_np = mode_probs[0].detach().cpu().numpy()
    mode_rows = []
    for mi, p in enumerate(mode_prob_np):
        mode_rows.append({
            "mode_index": int(mi),
            "mode": mode_names[mi] if mi < len(mode_names) else str(mi),
            "prob": float(p),
        })
    mode_rows.sort(key=lambda r: -r["prob"])
    write_csv(out_dir / "selector_mode_probs.csv", mode_rows)

    selector_meta = codebook_meta[template_idx] if template_idx < len(codebook_meta) else {}
    selector_pred = float(selector_scored["pred"].item())
    selector_abs = float(selector_scored["abs_error"].item())

    base_abs = abs(float(base_pred.item()) - float(target.item()))
    input_abs = abs(float(input_pred.item()) - float(target.item()))

    result = {
        "what_this_tests": "V13 template selector diagnostic with oracle brute-force comparison over the saved real-cloud codebook.",
        "sample": {
            "split": args.split,
            "sample_index": idx,
            "sample_id": str(item["sample_id"]),
            "location": str(item["location"]),
            "anchor": str(item["anchor"]),
            "actual_dataset_loss_wm2": float(actual.item()),
            "clear_sky_wm2": float(clear.item()),
        },
        "target_loss_wm2": float(target.item()),
        **target_meta,
        "input_setup": {
            "input_mode": args.input_mode,
            "input_info": input_info,
        },
        "base": {
            "original_pred_loss_wm2": float(base_pred.item()),
            "input_pred_loss_wm2": float(input_pred.item()),
            "original_abs_error_wm2": base_abs,
            "input_abs_error_wm2": input_abs,
            "input_coverage": coverage(v13, input_img, cov_channels),
            "original_coverage": coverage(v13, image, cov_channels),
        },
        "selector_choice": {
            "template_index": template_idx,
            "template_meta": selector_meta,
            "mode_index": mode_idx,
            "mode": mode_names[mode_idx] if mode_idx < len(mode_names) else str(mode_idx),
            "template_prob": float(template_probs[0, template_idx].item()),
            "mode_prob": float(mode_probs[0, mode_idx].item()),
            "pred_loss_wm2": selector_pred,
            "abs_error_wm2": selector_abs,
            "improvement_vs_input_wm2": input_abs - selector_abs,
            "improvement_vs_original_wm2": base_abs - selector_abs,
            "coverage": coverage(v13, selector_output, cov_channels),
            "image_l1_vs_input": float((selector_output - input_img).abs().mean().item()),
        },
        "oracle_best": {
            **oracle_row,
            "improvement_vs_input_wm2": input_abs - float(oracle_row["abs_error_wm2"]),
            "improvement_vs_original_wm2": base_abs - float(oracle_row["abs_error_wm2"]),
        },
        "selector_vs_oracle": {
            "selector_error_minus_oracle_error_wm2": selector_abs - float(oracle_row["abs_error_wm2"]),
            "selector_pred_minus_oracle_pred_wm2": selector_pred - float(oracle_row["pred_loss_wm2"]),
            "same_template": int(template_idx) == int(oracle_row["template_index"]),
            "same_mode": int(mode_idx) == int(oracle_row["mode_index"]),
        },
        "checkpoint": {
            "selector_checkpoint": str(args.selector_checkpoint),
            "selector_epoch": int(sel_ckpt.get("epoch", -1)),
            "selector_architecture": str(sel_ckpt.get("architecture", "unknown")),
            "reward_checkpoint": str(reward_path),
            "codebook_size": int(codebook.size(0)),
            "mode_names": mode_names,
        },
        "outputs": {
            "result_json": str(out_dir / "result.json"),
            "comparison_sheets": sheets,
            "oracle_candidate_scores_csv": str(out_dir / "oracle_candidate_scores.csv"),
            "selector_top_templates_csv": str(out_dir / "selector_top_templates.csv"),
            "selector_mode_probs_csv": str(out_dir / "selector_mode_probs.csv"),
            "image_channel_stats_csv": str(out_dir / "image_channel_stats.csv"),
            "image_diff_stats_csv": str(out_dir / "image_diff_stats.csv"),
        },
    }

    warnings = []
    if result["selector_choice"]["improvement_vs_input_wm2"] < 0:
        warnings.append("Selector output is worse than input.")
    if result["oracle_best"]["improvement_vs_input_wm2"] < 5:
        warnings.append("Oracle over saved codebook barely improves input; action space/codebook may be insufficient for this sample.")
    if result["selector_vs_oracle"]["selector_error_minus_oracle_error_wm2"] > 20:
        warnings.append("Selector is far from oracle; training/selection model is the bottleneck, not codebook/action space.")
    if result["selector_choice"]["template_prob"] < 0.15:
        warnings.append("Selector template probability is diffuse/uncertain.")
    result["warnings"] = warnings

    write_json(out_dir / "result.json", result)

    if args.save_npz:
        np.savez_compressed(
            out_dir / "arrays.npz",
            original=orig_np,
            input=input_np,
            selector_template=sel_template_np,
            selector_output=sel_output_np,
            oracle_template=oracle_template_np,
            oracle_output=oracle_output_np,
        )

    print("DONE")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
