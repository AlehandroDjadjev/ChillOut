#!/usr/bin/env python3
"""
diagnose_cloud_temp_interaction.py

Diagnostics for train_cloud_temp_interaction.py checkpoints.

Outputs:
  - metrics_by_variant.csv
  - component_metrics.csv
  - input_feature_importance.csv
  - worst_samples.csv
  - summary.json

Variants:
  normal, no_cloud_image, no_cloud_all, cloud_shuffle, no_world, world_shuffle, all_zero
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_cloud_temp_interaction import (
    CloudTempSequenceDataset,
    CloudWorldInteractionModel,
    Normalizer,
    TargetNormalizer,
    load_metadata,
    load_records,
    metrics_from_pred,
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def apply_variant(batch: Dict[str, Any], variant: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = batch["mask"].to(device)
    cloud = batch["cloud_features"].to(device)
    world = batch["world_features"].to(device)

    if variant == "normal":
        return mask, cloud, world

    if variant == "no_cloud_image":
        return torch.zeros_like(mask), cloud, world

    if variant == "no_cloud_all":
        return torch.zeros_like(mask), torch.zeros_like(cloud), world

    if variant == "cloud_shuffle":
        if mask.size(0) > 1:
            perm = torch.randperm(mask.size(0), device=device)
            return mask[perm], cloud[perm], world
        return mask, cloud, world

    if variant == "no_world":
        return mask, cloud, torch.zeros_like(world)

    if variant == "world_shuffle":
        if world.size(0) > 1:
            perm = torch.randperm(world.size(0), device=device)
            return mask, cloud, world[perm]
        return mask, cloud, world

    if variant == "all_zero":
        return torch.zeros_like(mask), torch.zeros_like(cloud), torch.zeros_like(world)

    raise ValueError(variant)


@torch.no_grad()
def eval_variants(model, loader, device, y_norm: TargetNormalizer, variants: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    rows = []
    component_rows = []
    worst_heap: List[Tuple[float, Dict[str, Any]]] = []

    for variant in variants:
        preds = []
        targets = []
        locations = []
        anchors = []
        sample_ids = []

        comp_preds = {"cloud": [], "world": [], "interaction": [], "final": []}

        for batch in tqdm(loader, desc=f"eval {variant}", leave=False):
            mask, cloud, world = apply_variant(batch, variant, device)
            target_raw = batch["target_raw"].to(device)

            out = model(mask, cloud, world)

            for name in comp_preds:
                comp_preds[name].append(y_norm.inverse_tensor(out[name].float()).detach().cpu())

            pred_raw = y_norm.inverse_tensor(out["final"].float())
            preds.append(pred_raw.detach().cpu())
            targets.append(target_raw.detach().cpu())
            locations += list(batch["location"])
            anchors += list(batch["anchor"])
            sample_ids += list(batch["sample_id"])

        pred_all = torch.cat(preds, dim=0)
        targ_all = torch.cat(targets, dim=0)
        m = metrics_from_pred(pred_all, targ_all)
        rows.append({"variant": variant, **m})

        if variant == "normal":
            for comp, vals in comp_preds.items():
                p = torch.cat(vals, dim=0)
                cm = metrics_from_pred(p, targ_all)
                component_rows.append({"component": comp, **cm})

            per_sample_err = (pred_all - targ_all).abs().flatten().numpy()
            for i, score in enumerate(per_sample_err):
                item = (float(score), {
                    "rank_score_mae_c": float(score),
                    "sample_id": sample_ids[i],
                    "location": locations[i],
                    "anchor": anchors[i],
                    "target_c": float(targ_all[i].item()),
                    "pred_c": float(pred_all[i].item()),
                    "error_c": float((pred_all[i] - targ_all[i]).item()),
                })
                if len(worst_heap) < 50:
                    import heapq
                    heapq.heappush(worst_heap, item)
                elif item[0] > worst_heap[0][0]:
                    import heapq
                    heapq.heapreplace(worst_heap, item)

    worst = [r for _, r in sorted(worst_heap, key=lambda x: x[0], reverse=True)]
    return rows, component_rows, worst


@torch.no_grad()
def feature_importance(model, loader, device, y_norm: TargetNormalizer, cloud_names: List[str], world_names: List[str], max_batches: int) -> List[Dict[str, Any]]:
    model.eval()

    # Cache batches for deterministic comparisons.
    cached = []
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        cached.append({
            "mask": batch["mask"].to(device),
            "cloud": batch["cloud_features"].to(device),
            "world": batch["world_features"].to(device),
            "target_raw": batch["target_raw"].to(device),
        })

    def run(zero_group: str | None = None, zero_idx: int | None = None) -> Dict[str, float]:
        preds, targets = [], []
        for b in cached:
            mask = b["mask"].clone()
            cloud = b["cloud"].clone()
            world = b["world"].clone()
            if zero_group == "cloud":
                cloud[:, :, zero_idx] = 0.0
            elif zero_group == "world":
                world[:, :, zero_idx] = 0.0
            out = model(mask, cloud, world)
            preds.append(y_norm.inverse_tensor(out["final"].float()).detach().cpu())
            targets.append(b["target_raw"].detach().cpu())
        return metrics_from_pred(torch.cat(preds, dim=0), torch.cat(targets, dim=0))

    base = run()
    rows: List[Dict[str, Any]] = []

    for i, name in enumerate(cloud_names):
        m = run("cloud", i)
        rows.append({
            "group": "cloud",
            "feature_index": i,
            "feature": name,
            "base_mae_c": base["mae_c"],
            "zeroed_mae_c": m["mae_c"],
            "mae_delta_zeroed_minus_base": m["mae_c"] - base["mae_c"],
            "base_rmse_c": base["rmse_c"],
            "zeroed_rmse_c": m["rmse_c"],
            "rmse_delta_zeroed_minus_base": m["rmse_c"] - base["rmse_c"],
        })

    for i, name in enumerate(world_names):
        m = run("world", i)
        rows.append({
            "group": "world",
            "feature_index": i,
            "feature": name,
            "base_mae_c": base["mae_c"],
            "zeroed_mae_c": m["mae_c"],
            "mae_delta_zeroed_minus_base": m["mae_c"] - base["mae_c"],
            "base_rmse_c": base["rmse_c"],
            "zeroed_rmse_c": m["rmse_c"],
            "rmse_delta_zeroed_minus_base": m["rmse_c"] - base["rmse_c"],
        })

    rows.sort(key=lambda r: r["mae_delta_zeroed_minus_base"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="diagnostics_interaction")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-importance", action="store_true")
    parser.add_argument("--feature-importance-batches", type=int, default=20)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    metadata = load_metadata(data_root)

    raw_names = ckpt.get("raw_feature_names", metadata["raw_feature_names"])
    cloud_names = ckpt.get("cloud_feature_names", metadata["cloud_feature_names"])
    world_names = ckpt.get("world_feature_names", metadata["world_feature_names"])

    x_norm = Normalizer.from_state_dict(ckpt["normalizer"])
    y_norm = TargetNormalizer.from_state_dict(ckpt["target_normalizer"])

    image_height = int(ckpt.get("image_height", 160))
    image_width = int(ckpt.get("image_width", 160))
    lookback = int(ckpt.get("lookback", 4))

    records = load_records(data_root, args.split)
    ds = CloudTempSequenceDataset(
        data_root=data_root,
        records=records,
        raw_names=raw_names,
        cloud_names=cloud_names,
        world_names=world_names,
        x_norm=x_norm,
        y_norm=y_norm,
        image_height=image_height,
        image_width=image_width,
        lookback=lookback,
        max_gap_days=float(ckpt.get("args", {}).get("max_gap_days", 12.0)),
        augment=False,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = ckpt["model_kwargs"]
    model = CloudWorldInteractionModel(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])

    variants = ["normal", "no_cloud_image", "no_cloud_all", "cloud_shuffle", "no_world", "world_shuffle", "all_zero"]
    metric_rows, component_rows, worst_rows = eval_variants(model, loader, device, y_norm, variants)

    write_csv(out_dir / "metrics_by_variant.csv", metric_rows)
    write_csv(out_dir / "component_metrics.csv", component_rows)
    write_csv(out_dir / "worst_samples.csv", worst_rows)

    summary = {
        "split": args.split,
        "num_records": len(records),
        "num_windows": len(ds),
        "variants": metric_rows,
        "components": component_rows,
        "cloud_feature_names": cloud_names,
        "world_feature_names": world_names,
    }

    if args.feature_importance:
        importance_rows = feature_importance(model, loader, device, y_norm, cloud_names, world_names, args.feature_importance_batches)
        write_csv(out_dir / "input_feature_importance.csv", importance_rows)
        summary["input_feature_importance_csv"] = str(out_dir / "input_feature_importance.csv")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print(f"wrote {out_dir / 'metrics_by_variant.csv'}")
    print(f"wrote {out_dir / 'component_metrics.csv'}")
    print(f"wrote {out_dir / 'worst_samples.csv'}")
    if args.feature_importance:
        print(f"wrote {out_dir / 'input_feature_importance.csv'}")

    print("\nVariant metrics:")
    for row in metric_rows:
        print(f"{row['variant']:<16} mae={row['mae_c']:.4f}C rmse={row['rmse_c']:.4f}C corr={row['corr']:.3f}")

    print("\nComponent metrics on normal input:")
    for row in component_rows:
        print(f"{row['component']:<12} mae={row['mae_c']:.4f}C rmse={row['rmse_c']:.4f}C corr={row['corr']:.3f}")


if __name__ == "__main__":
    main()
