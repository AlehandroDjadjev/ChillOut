#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from train_cloud_temp_cloudforced_radiation_v6 import (
        CloudTempResidualSequenceDataset, ResidualTrendSplitConvLSTM, Normalizer, TargetNormalizer,
        load_metadata, load_records, metrics_from_pred, future_from_delta, get_target_temp
    )
except ModuleNotFoundError:
    try:
        from train_cloud_temp_cloudforced_radiation_v6 import (
            CloudTempResidualSequenceDataset, ResidualTrendSplitConvLSTM, Normalizer, TargetNormalizer,
            load_metadata, load_records, metrics_from_pred, future_from_delta, get_target_temp
        )
    except ModuleNotFoundError:
        from train_cloud_temp_cloudforced_radiation_v6 import (  # type: ignore
            CloudTempResidualSequenceDataset, ResidualTrendSplitConvLSTM, Normalizer, TargetNormalizer,
            load_metadata, load_records, metrics_from_pred, future_from_delta, get_target_temp
        )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def strip_prefix(state):
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.","",1): v for k,v in state.items()}
    return state


def variant_batch(batch, variant, device):
    mask = batch["mask"].to(device)
    cloud = batch["cloud_features"].to(device)
    world = batch["world_features"].to(device)
    trend = batch["trend_features"].to(device)
    flags = {}
    if variant == "normal": return mask, cloud, world, trend, flags
    if variant == "no_cloud_image": return torch.zeros_like(mask), cloud, world, trend, flags
    if variant == "no_cloud_scalars": return mask, torch.zeros_like(cloud), world, trend, flags
    if variant == "no_cloud_all": return torch.zeros_like(mask), torch.zeros_like(cloud), world, trend, flags
    if variant == "cloud_shuffle":
        if mask.size(0) > 1:
            p = torch.randperm(mask.size(0), device=device); return mask[p], cloud[p], world, trend, flags
        return mask, cloud, world, trend, flags
    if variant == "no_world": return mask, cloud, torch.zeros_like(world), trend, flags
    if variant == "world_shuffle":
        if world.size(0) > 1:
            p = torch.randperm(world.size(0), device=device); return mask, cloud, world[p], trend, flags
        return mask, cloud, world, trend, flags
    if variant == "no_temp_trend": return mask, cloud, world, torch.zeros_like(trend), flags
    if variant == "trend_shuffle":
        if trend.size(0) > 1:
            p = torch.randperm(trend.size(0), device=device); return mask, cloud, world, trend[p], flags
        return mask, cloud, world, trend, flags
    if variant == "no_interaction": flags["disable_interaction"] = True; return mask, cloud, world, trend, flags
    if variant == "no_image_delta": flags["disable_image_delta"] = True; return mask, cloud, world, trend, flags
    if variant == "no_cloud_delta": flags["disable_cloud_delta"] = True; return mask, cloud, world, trend, flags
    if variant == "no_world_delta": flags["disable_world_delta"] = True; return mask, cloud, world, trend, flags
    if variant == "no_trend_delta": flags["disable_trend_delta"] = True; return mask, cloud, world, trend, flags
    if variant == "all_zero": return torch.zeros_like(mask), torch.zeros_like(cloud), torch.zeros_like(world), torch.zeros_like(trend), flags
    raise ValueError(variant)


@torch.no_grad()
def eval_variants(model, loader, device, delta_norm, variants, context_scale):
    rows=[]; comp_rows=[]; worst=[]
    for variant in variants:
        preds=[]; targets=[]; currents=[]; locs=[]; anchors=[]; ids=[]
        comps = {k: [] for k in ["image_delta_as_forecast","cloud_delta_as_forecast","world_delta_as_forecast","trend_delta_as_forecast","interaction_delta_as_forecast","cloud_primary_as_forecast","context_delta_as_forecast","final"]}
        gates=[]
        for batch in tqdm(loader, desc=f"eval {variant}", leave=False):
            mask, cloud, world, trend, flags = variant_batch(batch, variant, device)
            cur = batch["current_temp_raw"].to(device); targ = batch["target_raw"].to(device)
            out = model(mask, cloud, world, trend, context_scale=context_scale, **flags)
            pred = future_from_delta(out["final_delta"], cur, delta_norm)
            preds.append(pred.cpu()); targets.append(targ.cpu()); currents.append(cur.cpu())
            locs += list(batch["location"]); anchors += list(batch["anchor"]); ids += list(batch["sample_id"])
            if "gate_mean" in out: gates += [float(x) for x in out["gate_mean"].detach().cpu().flatten()]
            if variant == "normal":
                comps["image_delta_as_forecast"].append(future_from_delta(out["image_delta"], cur, delta_norm).cpu())
                comps["cloud_delta_as_forecast"].append(future_from_delta(out["cloud_delta"], cur, delta_norm).cpu())
                comps["world_delta_as_forecast"].append(future_from_delta(out["world_delta"], cur, delta_norm).cpu())
                comps["trend_delta_as_forecast"].append(future_from_delta(out["trend_delta"], cur, delta_norm).cpu())
                comps["interaction_delta_as_forecast"].append(future_from_delta(out["interaction_delta"], cur, delta_norm).cpu())
                comps["cloud_primary_as_forecast"].append(future_from_delta(out["cloud_primary_delta"], cur, delta_norm).cpu())
                comps["context_delta_as_forecast"].append(future_from_delta(out["context_delta"], cur, delta_norm).cpu())
                comps["final"].append(pred.cpu())
        p = torch.cat(preds); y = torch.cat(targets); c = torch.cat(currents)
        m = metrics_from_pred(p, y); pm = metrics_from_pred(c, y)
        m["persistence_mae_c"] = pm["mae_c"]; m["improvement_vs_persistence_c"] = pm["mae_c"] - m["mae_c"]
        if gates: m["gate_mean"] = float(np.mean(gates))
        rows.append({"variant": variant, **m})
        if variant == "normal":
            for name, vals in comps.items():
                cm = metrics_from_pred(torch.cat(vals), y)
                cm["persistence_mae_c"] = pm["mae_c"]; cm["improvement_vs_persistence_c"] = pm["mae_c"] - cm["mae_c"]
                comp_rows.append({"component": name, **cm})
            err = (p-y).abs().flatten()
            for i, e in enumerate(err.tolist()):
                worst.append({"rank_score_mae_c": float(e), "sample_id": ids[i], "location": locs[i], "anchor": anchors[i],
                              "current_c": float(c[i]), "target_c": float(y[i]), "pred_c": float(p[i]),
                              "persistence_error_c": float(c[i]-y[i]), "model_error_c": float(p[i]-y[i])})
    worst = sorted(worst, key=lambda r: r["rank_score_mae_c"], reverse=True)[:50]
    return rows, comp_rows, worst


@torch.no_grad()
def baselines_and_locations(model, loader, device, delta_norm, variants, train_target_mean, city_means, context_scale):
    base = {k: {"pred": [], "target": []} for k in ["current_temp_persistence","previous_snapshot_trend","window_linear_trend","train_mean","city_train_mean"]}
    loc_rows = {}
    for batch in tqdm(loader, desc="baselines/per-location", leave=False):
        y = batch["target_raw"].float(); c = batch["current_temp_raw"].float(); locs = [str(x) for x in batch["location"]]
        prev_pred = c + batch["previous_snapshot_trend_delta_raw"].float()
        lin_pred = c + batch["window_linear_trend_delta_raw"].float()
        tm = torch.full_like(y, float(train_target_mean))
        cm = torch.tensor([[float(city_means.get(loc, train_target_mean))] for loc in locs], dtype=torch.float32)
        for name, pred in [("current_temp_persistence",c), ("previous_snapshot_trend",prev_pred), ("window_linear_trend",lin_pred), ("train_mean",tm), ("city_train_mean",cm)]:
            base[name]["pred"] += [float(x) for x in pred.flatten()]; base[name]["target"] += [float(x) for x in y.flatten()]
        for var in variants:
            mask, cloud, world, trend, flags = variant_batch(batch, var, device)
            out = model(mask, cloud, world, trend, context_scale=context_scale, **flags)
            pred = future_from_delta(out["final_delta"], batch["current_temp_raw"].to(device), delta_norm).cpu().flatten()
            for i, loc in enumerate(locs):
                key = (var, loc); rec = loc_rows.setdefault(key, {"variant": var, "location": loc, "pred": [], "target": []})
                rec["pred"].append(float(pred[i])); rec["target"].append(float(y.flatten()[i]))
    base_rows=[]
    for name, d in base.items():
        base_rows.append({"name": name, **metrics_from_pred(torch.tensor(d["pred"]).view(-1,1), torch.tensor(d["target"]).view(-1,1))})
    per=[]
    for (var, loc), d in sorted(loc_rows.items()):
        per.append({"variant": var, "location": loc, "n": len(d["pred"]),
                    **metrics_from_pred(torch.tensor(d["pred"]).view(-1,1), torch.tensor(d["target"]).view(-1,1))})
    return base_rows, per


@torch.no_grad()
def feature_importance(model, loader, device, delta_norm, cloud_names, world_names, max_batches, context_scale):
    cache=[]
    for i,b in enumerate(loader):
        if max_batches > 0 and i >= max_batches: break
        cache.append({k: (b[k].to(device) if k in ["mask","cloud_features","world_features","trend_features","current_temp_raw","target_raw"] else b[k])
                      for k in b})
    def run(group=None, idx=None):
        preds=[]; targets=[]; currents=[]
        for b in cache:
            mask=b["mask"].clone(); cloud=b["cloud_features"].clone(); world=b["world_features"].clone(); trend=b["trend_features"].clone()
            if group == "image": mask.zero_()
            if group == "cloud": cloud[:,:,idx] = 0
            if group == "world": world[:,:,idx] = 0
            if group == "trend": trend[:,:,idx] = 0
            out = model(mask, cloud, world, trend, context_scale=context_scale)
            preds.append(future_from_delta(out["final_delta"], b["current_temp_raw"], delta_norm).cpu())
            targets.append(b["target_raw"].cpu()); currents.append(b["current_temp_raw"].cpu())
        m = metrics_from_pred(torch.cat(preds), torch.cat(targets)); pm = metrics_from_pred(torch.cat(currents), torch.cat(targets))
        m["persistence_mae_c"] = pm["mae_c"]; m["improvement_vs_persistence_c"] = pm["mae_c"] - m["mae_c"]
        return m
    base=run(); rows=[]
    trend_names = ["trend_temp_rel_current", "trend_days_rel_current", "trend_step_temp_change", "trend_step_days"]
    for group, names in [("image", ["cloud_image_sequence"]), ("cloud", cloud_names), ("world", world_names), ("trend", trend_names)]:
        for i, name in enumerate(names):
            m = run(group, 0 if group=="image" else i)
            rows.append({"group": group, "feature_index": -1 if group=="image" else i, "feature": name,
                         "base_mae_c": base["mae_c"], "zeroed_mae_c": m["mae_c"],
                         "mae_delta_zeroed_minus_base": m["mae_c"] - base["mae_c"],
                         "base_rmse_c": base["rmse_c"], "zeroed_rmse_c": m["rmse_c"],
                         "rmse_delta_zeroed_minus_base": m["rmse_c"] - base["rmse_c"]})
    return sorted(rows, key=lambda r: r["mae_delta_zeroed_minus_base"], reverse=True)



@torch.no_grad()
def radiation_metrics(model, loader, device, ckpt):
    """Evaluate image/cloud radiation heads against radiation_cloud_loss_wm2."""
    scale = float(ckpt.get("args", {}).get("radiation_loss_scale", 300.0))
    context_scale = float(ckpt.get("args", {}).get("context_scale", 0.15))
    rows = []
    preds = {"image_radiation_loss_wm2": [], "cloud_radiation_loss_wm2": []}
    target = []
    valid_count = 0
    model.eval()
    for batch in tqdm(loader, desc="radiation heads", leave=False):
        mask, cloud, world, trend, flags = variant_batch(batch, "normal", device)
        out = model(mask, cloud, world, trend, context_scale=context_scale)
        valid = batch["radiation_valid"].float().view(-1) > 0.5
        if not valid.any():
            continue
        image_pred = (out["image_radiation"].detach().cpu().float().view(-1) * scale)[valid]
        cloud_pred = (out["cloud_radiation"].detach().cpu().float().view(-1) * scale)[valid]
        targ = batch["radiation_loss_raw"].float().view(-1)[valid]
        preds["image_radiation_loss_wm2"].append(image_pred.view(-1, 1))
        preds["cloud_radiation_loss_wm2"].append(cloud_pred.view(-1, 1))
        target.append(targ.view(-1, 1))
        valid_count += int(valid.sum().item())
    if not target:
        return []
    y = torch.cat(target, 0)
    for name, vals in preds.items():
        p = torch.cat(vals, 0)
        m = metrics_from_pred(p, y)
        rows.append({"radiation_component": name, "n_daylight": valid_count, **m})
    # simple zero-loss baseline and mean target baseline
    zero = torch.zeros_like(y)
    mean = torch.full_like(y, float(y.mean()))
    rows.append({"radiation_component": "zero_cloud_loss_baseline", "n_daylight": valid_count, **metrics_from_pred(zero, y)})
    rows.append({"radiation_component": "mean_cloud_loss_baseline", "n_daylight": valid_count, **metrics_from_pred(mean, y)})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--out-dir", default="diagnostics_hybrid_v4")
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--feature-importance", action="store_true"); ap.add_argument("--feature-importance-batches", type=int, default=20)
    ap.add_argument("--benchmark-location", default=None)
    args=ap.parse_args()

    root=Path(args.data_root).resolve(); out_dir=Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt=torch.load(args.checkpoint, map_location="cpu"); meta=load_metadata(root)
    raw=ckpt.get("raw_feature_names", meta["raw_feature_names"]); cloud=ckpt.get("cloud_feature_names", meta["cloud_feature_names"]); world=ckpt.get("world_feature_names", meta["world_feature_names"])
    x_norm=Normalizer.from_state_dict(ckpt["normalizer"])
    delta_norm=TargetNormalizer.from_state_dict(ckpt.get("delta_normalizer", ckpt["target_normalizer"]))
    records=load_records(root,args.split)
    if args.benchmark_location:
        key=args.benchmark_location.lower()
        records=[r for r in records if key in str(r.get("location","")).lower() or key in str(r.get("city","")).lower()]
        if not records: raise SystemExit(f"No records matched {args.benchmark_location!r}")
    ds=CloudTempResidualSequenceDataset(root, records, raw, cloud, world, x_norm, delta_norm,
                                        int(ckpt.get("image_height",160)), int(ckpt.get("image_width",160)),
                                        int(ckpt.get("lookback",4)), float(ckpt.get("args",{}).get("max_gap_days",12.0)),
                                        augment=False, cache_images=False)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ResidualTrendSplitConvLSTM(**ckpt["model_kwargs"]).to(device); model.load_state_dict(strip_prefix(ckpt["model_state"])); model.eval()
    context_scale = float(ckpt.get("args", {}).get("context_scale", 0.15))
    variants=["normal","no_cloud_image","no_cloud_scalars","no_cloud_all","cloud_shuffle","no_world","world_shuffle","no_temp_trend","trend_shuffle","no_interaction","no_image_delta","no_cloud_delta","no_world_delta","no_trend_delta","all_zero"]
    metric_rows, comp_rows, worst_rows = eval_variants(model, loader, device, delta_norm, variants, context_scale)
    train_mean=float(ckpt.get("train_target_mean_c", 0.0))
    if train_mean == 0.0:
        train_mean=float(np.mean([get_target_temp(r) for r in load_records(root,"train")]))
    base_rows, per_rows = baselines_and_locations(model, loader, device, delta_norm, variants, train_mean, ckpt.get("train_city_means", {}), context_scale)
    rad_rows = radiation_metrics(model, loader, device, ckpt)
    write_csv(out_dir/"metrics_by_variant.csv", metric_rows); write_csv(out_dir/"component_metrics.csv", comp_rows)
    write_csv(out_dir/"baseline_metrics.csv", base_rows); write_csv(out_dir/"per_location_metrics.csv", per_rows)
    write_csv(out_dir/"worst_samples.csv", worst_rows); write_csv(out_dir/"radiation_metrics.csv", rad_rows)
    summary={"architecture": ckpt.get("architecture","ResidualTrendSplitConvLSTM_v5"), "target_is_delta": True,
             "formula": "future_temp = current_temperature_c + inverse_delta(final_delta); trend branch uses past/current temps from lookback",
             "split": args.split, "num_records": len(records), "num_windows": len(ds), "context_scale": context_scale,
             "variants": metric_rows, "components": comp_rows, "baselines": base_rows, "radiation_metrics": rad_rows,
             "cloud_feature_names": cloud, "world_feature_names": world}
    if args.feature_importance:
        rows=feature_importance(model, loader, device, delta_norm, cloud, world, args.feature_importance_batches, context_scale)
        write_csv(out_dir/"input_feature_importance.csv", rows); summary["input_feature_importance_csv"]=str(out_dir/"input_feature_importance.csv")
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("DONE")
    print(f"wrote {out_dir/'metrics_by_variant.csv'}")
    print(f"wrote {out_dir/'component_metrics.csv'}")
    print(f"wrote {out_dir/'baseline_metrics.csv'}")
    print(f"wrote {out_dir/'per_location_metrics.csv'}")
    print(f"wrote {out_dir/'worst_samples.csv'}")
    print(f"wrote {out_dir/'radiation_metrics.csv'}")
    if args.feature_importance: print(f"wrote {out_dir/'input_feature_importance.csv'}")
    print("\nVariant metrics:")
    for r in metric_rows:
        print(f"{r['variant']:<18} mae={r['mae_c']:.4f}C rmse={r['rmse_c']:.4f}C corr={r['corr']:.3f} persist={r.get('persistence_mae_c',float('nan')):.4f}C improve={r.get('improvement_vs_persistence_c',float('nan')):.4f}C")
    print("\nBaselines:")
    for r in base_rows:
        print(f"{r['name']:<24} mae={r['mae_c']:.4f}C rmse={r['rmse_c']:.4f}C corr={r['corr']:.3f}")
    print("\nRadiation head metrics against cloud radiative loss W/m2:")
    for r in rad_rows:
        print(f"{r['radiation_component']:<30} mae={r['mae_c']:.2f}W/m2 rmse={r['rmse_c']:.2f}W/m2 corr={r['corr']:.3f} n={r['n_daylight']}")
    print("\nComponent metrics as current_temp + component_delta:")
    for r in comp_rows:
        print(f"{r['component']:<30} mae={r['mae_c']:.4f}C rmse={r['rmse_c']:.4f}C corr={r['corr']:.3f} improve={r.get('improvement_vs_persistence_c',float('nan')):.4f}C")


if __name__ == "__main__":
    main()
