#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_cloud_radiation_bottom_v8_CLEAN_DIRECT import (
    base, CloudRadiationV8CleanDirect, RadiationSequenceDataset, strip_prefix,
    metric, clean_radiation_records
)


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def apply_variant(batch, variant, device):
    image = batch["image"].to(device)
    cloud = batch["cloud_features"].to(device)
    ctx = batch["context_features"].to(device)
    clear = batch["clear_wm2"].to(device)
    if variant == "normal":
        return image, cloud, ctx, clear
    if variant == "no_cloud_image":
        return torch.zeros_like(image), cloud, ctx, clear
    if variant == "no_cloud_scalars":
        return image, torch.zeros_like(cloud), ctx, clear
    if variant == "no_cloud_all":
        return torch.zeros_like(image), torch.zeros_like(cloud), ctx, clear
    if variant == "cloud_shuffle":
        if image.size(0) > 1:
            p = torch.randperm(image.size(0), device=device)
            return image[p], cloud[p], ctx, clear
        return image, cloud, ctx, clear
    if variant == "no_context":
        return image, cloud, torch.zeros_like(ctx), clear
    if variant == "half_clear":
        return image, cloud, ctx, clear * 0.5
    raise ValueError(variant)


@torch.no_grad()
def evaluate_variants(model, loader, device, variants):
    rows = []
    pred_rows = []
    for variant in variants:
        preds, targets = [], []
        preds_attn, targets_attn = [], []
        from_attn = []
        ids, locs, anchors = [], [], []
        clears = []
        for batch in tqdm(loader, desc=f"eval {variant}", leave=False):
            image, cloud, ctx, clear = apply_variant(batch, variant, device)
            out = model(image, cloud, ctx, clear)
            valid = batch["valid"].view(-1) > 0.5
            y = batch["target_loss_wm2"].float().view(-1)
            ya = batch["target_attenuation"].float().view(-1)
            if valid.any():
                preds.append(out["loss_wm2"].detach().cpu().float().view(-1)[valid])
                from_attn.append(out["loss_from_attenuation"].detach().cpu().float().view(-1)[valid])
                targets.append(y[valid])
                preds_attn.append(out["attenuation"].detach().cpu().float().view(-1)[valid])
                targets_attn.append(ya[valid])
                if variant == "normal":
                    ids += [x for i,x in enumerate(batch["sample_id"]) if bool(valid[i])]
                    locs += [x for i,x in enumerate(batch["location"]) if bool(valid[i])]
                    anchors += [x for i,x in enumerate(batch["anchor"]) if bool(valid[i])]
                    clears += [float(x) for x in batch["clear_wm2"].view(-1)[valid]]
        p = torch.cat(preds)
        y = torch.cat(targets)
        pa = torch.cat(preds_attn)
        ya = torch.cat(targets_attn)
        pfa = torch.cat(from_attn)
        lm = metric(p, y)
        am = metric(pa, ya)
        fam = metric(pfa, y)
        rows.append({
            "variant": variant,
            "loss_mae_wm2": lm["mae"],
            "loss_rmse_wm2": lm["rmse"],
            "loss_bias_wm2": lm["bias"],
            "loss_corr": lm["corr"],
            "attenuation_mae": am["mae"],
            "attenuation_rmse": am["rmse"],
            "attenuation_bias": am["bias"],
            "attenuation_corr": am["corr"],
            "from_attenuation_mae_wm2": fam["mae"],
        })
        if variant == "normal":
            for i in range(p.numel()):
                pred_rows.append({
                    "sample_id": ids[i],
                    "location": locs[i],
                    "anchor": anchors[i],
                    "clear_wm2": clears[i],
                    "actual_loss_wm2": float(y[i]),
                    "pred_loss_wm2": float(p[i]),
                    "pred_from_attenuation_wm2": float(pfa[i]),
                    "actual_attenuation": float(ya[i]),
                    "pred_attenuation": float(pa[i]),
                    "abs_error_wm2": abs(float(p[i] - y[i])),
                })
    return rows, pred_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default="diagnostics_radiation_v7")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    raw = ckpt["raw_feature_names"]
    cloud_names = ckpt["cloud_feature_names"]
    x_norm = base.Normalizer.from_state_dict(ckpt["normalizer"])
    records = base.load_records(root, args.split)

    class _Args:
        pass
    _a = _Args()
    ckargs = ckpt.get("args", {})
    _a.min_clear_wm2 = float(ckargs.get("min_clear_wm2", 120.0))
    _a.clean_drop_invalid = bool(ckargs.get("clean_drop_invalid", True))
    _a.clean_drop_negative = bool(ckargs.get("clean_drop_negative", True))
    _a.clean_drop_high_cloud_low_loss = bool(ckargs.get("clean_drop_high_cloud_low_loss", True))
    _a.clean_drop_low_cloud_high_loss = bool(ckargs.get("clean_drop_low_cloud_high_loss", False))
    _a.high_cloud_thresh = float(ckargs.get("high_cloud_thresh", 0.65))
    _a.low_cloud_thresh = float(ckargs.get("low_cloud_thresh", 0.05))
    _a.low_loss_thresh_wm2 = float(ckargs.get("low_loss_thresh_wm2", 25.0))
    _a.high_loss_thresh_wm2 = float(ckargs.get("high_loss_thresh_wm2", 280.0))
    _a.min_loss_wm2 = float(ckargs.get("min_loss_wm2", 0.0))
    _a.max_loss_wm2 = float(ckargs.get("max_loss_wm2", 900.0))
    records = clean_radiation_records(records, _a, args.split)

    ds = RadiationSequenceDataset(
        root=root,
        records=records,
        raw_names=raw,
        cloud_names=cloud_names,
        x_norm=x_norm,
        image_height=int(ckpt.get("image_height", 160)),
        image_width=int(ckpt.get("image_width", 160)),
        lookback=int(ckpt.get("lookback", 4)),
        max_gap_days=float(ckpt.get("args", {}).get("max_gap_days", 12.0)),
        use_cloud_tensor=bool(ckpt.get("args", {}).get("use_cloud_tensor", False)),
        augment=False,
        min_clear_wm2=float(ckpt.get("args", {}).get("min_clear_wm2", 120.0)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CloudRadiationV8CleanDirect(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(strip_prefix(ckpt["model_state"]))
    model.eval()

    variants = ["normal", "no_cloud_image", "no_cloud_scalars", "no_cloud_all", "cloud_shuffle", "no_context", "half_clear"]
    rows, pred_rows = evaluate_variants(model, loader, device, variants)
    write_csv(out_dir / "radiation_v7_metrics_by_variant.csv", rows)
    write_csv(out_dir / "radiation_v7_predictions.csv", pred_rows)
    summary = {"architecture": ckpt.get("architecture", "CloudRadiationV8CleanDirect"), "split": args.split, "n_windows": len(ds), "metrics": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print(f"wrote {out_dir/'radiation_v7_metrics_by_variant.csv'}")
    print(f"wrote {out_dir/'radiation_v7_predictions.csv'}")
    print("\nVariant metrics:")
    for r in rows:
        print(f"{r['variant']:<18} loss_mae={r['loss_mae_wm2']:.2f}W/m2 rmse={r['loss_rmse_wm2']:.2f} corr={r['loss_corr']:.3f} attn_mae={r['attenuation_mae']:.3f}")


if __name__ == "__main__":
    main()
