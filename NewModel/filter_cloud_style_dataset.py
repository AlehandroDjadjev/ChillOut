#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def tensor_path(root: Path, row: Dict[str, Any]) -> Path | None:
    p = row.get("cloud_tensor_path")
    if not p:
        return None
    return root / str(p).replace("\\", "/")


def mask_path(root: Path, row: Dict[str, Any]) -> Path | None:
    p = row.get("mask_path")
    if not p:
        return None
    return root / str(p).replace("\\", "/")


def load_cloud_tensor(root: Path, row: Dict[str, Any]) -> np.ndarray | None:
    p = tensor_path(root, row)
    if p is not None and p.exists():
        try:
            data = np.load(p)
            if "cloud_tensor" in data:
                return np.asarray(data["cloud_tensor"], dtype=np.float32)
        except Exception:
            return None

    # Fallback to mask PNG if tensor is missing.
    mp = mask_path(root, row)
    if mp is not None and mp.exists():
        try:
            from PIL import Image
            im = Image.open(mp).convert("L")
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr[..., None]
        except Exception:
            return None
    return None


def compute_style_stats(arr: np.ndarray, args) -> Dict[str, Any]:
    """arr can be [H,W,C] or [T,H,W,C] or [T,C,H,W]."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim == 3:
        # [H,W,C]
        h, w, c = x.shape
        chw = np.moveaxis(x, -1, 0)[None]  # [1,C,H,W]
    elif x.ndim == 4:
        # Decide [T,H,W,C] vs [T,C,H,W]
        if x.shape[-1] <= 16:
            chw = np.moveaxis(x, -1, 1)  # [T,C,H,W]
        else:
            chw = x
    else:
        raise ValueError(f"unexpected tensor shape {x.shape}")

    t, c, h, w = chw.shape
    mask_ch = min(args.mask_channel, c - 1)
    mask = np.clip(chw[:, mask_ch], 0.0, 1.0)

    white = mask >= args.white_threshold
    black = mask <= args.black_threshold

    stats = {
        "shape": list(chw.shape),
        "mask_channel": int(mask_ch),
        "white_frac": float(white.mean()),
        "black_frac": float(black.mean()),
        "midgray_frac": float(((mask > args.black_threshold) & (mask < args.white_threshold)).mean()),
        "mean": float(mask.mean()),
        "std": float(mask.std()),
        "min": float(mask.min()),
        "max": float(mask.max()),
        "per_frame_white_frac": [float((mask[i] >= args.white_threshold).mean()) for i in range(t)],
        "per_frame_black_frac": [float((mask[i] <= args.black_threshold).mean()) for i in range(t)],
    }

    # Any channel style stats for all selected channels.
    selected = []
    for raw in str(args.channels).split(","):
        raw = raw.strip()
        if raw == "":
            continue
        ch = int(raw)
        if 0 <= ch < c:
            selected.append(ch)

    if selected:
        vals = chw[:, selected]
        stats["selected_white_frac"] = float((vals >= args.white_threshold).mean())
        stats["selected_black_frac"] = float((vals <= args.black_threshold).mean())
        stats["selected_mean"] = float(vals.mean())
        stats["selected_std"] = float(vals.std())

    return stats


def drop_reason(stats: Dict[str, Any], args) -> str | None:
    # Main rule requested by user: remove >=95% white or >=95% black.
    if stats["white_frac"] >= args.max_white_frac:
        return f"too_white_mask_white_frac_{stats['white_frac']:.4f}"
    if stats["black_frac"] >= args.max_black_frac:
        return f"too_black_mask_black_frac_{stats['black_frac']:.4f}"

    # Optional stricter rules on selected cloud channels.
    if args.use_selected_channel_filter:
        if stats.get("selected_white_frac", 0.0) >= args.max_selected_white_frac:
            return f"too_white_selected_white_frac_{stats['selected_white_frac']:.4f}"
        if stats.get("selected_black_frac", 0.0) >= args.max_selected_black_frac:
            return f"too_black_selected_black_frac_{stats['selected_black_frac']:.4f}"

    if stats["std"] < args.min_mask_std:
        return f"too_flat_mask_std_{stats['std']:.4f}"

    return None


def copy_record_files(src_root: Path, dst_root: Path, row: Dict[str, Any], copied: set[str]) -> None:
    for key in ["cloud_tensor_path", "mask_path"]:
        rel = row.get(key)
        if not rel:
            continue
        rel = str(rel).replace("\\", "/")
        src = src_root / rel
        dst = dst_root / rel
        if not src.exists():
            continue
        marker = str(dst)
        if marker in copied:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.add(marker)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)

    ap.add_argument("--mask-channel", type=int, default=0)
    ap.add_argument("--channels", default="0,1,2,4,5,7")
    ap.add_argument("--white-threshold", type=float, default=0.95)
    ap.add_argument("--black-threshold", type=float, default=0.02)

    ap.add_argument("--max-white-frac", type=float, default=0.95)
    ap.add_argument("--max-black-frac", type=float, default=0.95)
    ap.add_argument("--min-mask-std", type=float, default=0.0)

    ap.add_argument("--use-selected-channel-filter", action="store_true")
    ap.add_argument("--max-selected-white-frac", type=float, default=0.98)
    ap.add_argument("--max-selected-black-frac", type=float, default=0.98)

    ap.add_argument("--copy-files", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.data_root).resolve()
    dst_root = Path(args.out_root).resolve()

    if not (src_root / "splits").exists():
        raise RuntimeError(f"Missing {src_root / 'splits'}")

    split_names = ["train", "val", "test"]
    reports = []
    kept_by_split: Dict[str, List[Dict[str, Any]]] = {}
    dropped_by_split: Dict[str, List[Dict[str, Any]]] = {}
    copied: set[str] = set()

    for split in split_names:
        rows = read_jsonl(src_root / "splits" / f"{split}.jsonl")
        kept = []
        dropped = []
        for row in rows:
            sample_id = str(row.get("sample_id", "unknown"))
            arr = load_cloud_tensor(src_root, row)
            if arr is None:
                reason = "missing_or_unreadable_tensor"
                stats = {}
            else:
                try:
                    stats = compute_style_stats(arr, args)
                    reason = drop_reason(stats, args)
                except Exception as exc:
                    stats = {}
                    reason = f"stats_error_{type(exc).__name__}"

            report_row = {
                "split": split,
                "sample_id": sample_id,
                "location": row.get("location"),
                "anchor": row.get("anchor"),
                "drop": reason is not None,
                "reason": reason or "",
                **{k: v for k, v in stats.items() if not isinstance(v, list)},
            }
            reports.append(report_row)

            if reason is None:
                kept.append(row)
                if args.copy_files and not args.dry_run:
                    copy_record_files(src_root, dst_root, row, copied)
            else:
                dropped.append(row)

        kept_by_split[split] = kept
        dropped_by_split[split] = dropped

    all_kept = []
    for split in split_names:
        all_kept.extend(kept_by_split[split])

    summary = {
        "source": str(src_root),
        "out": str(dst_root),
        "rules": {
            "mask_channel": args.mask_channel,
            "white_threshold": args.white_threshold,
            "black_threshold": args.black_threshold,
            "max_white_frac": args.max_white_frac,
            "max_black_frac": args.max_black_frac,
            "min_mask_std": args.min_mask_std,
            "channels": args.channels,
        },
        "splits": {
            split: {
                "kept": len(kept_by_split[split]),
                "dropped": len(dropped_by_split[split]),
                "original": len(kept_by_split[split]) + len(dropped_by_split[split]),
            }
            for split in split_names
        },
        "total_kept": len(all_kept),
        "total_dropped": sum(len(dropped_by_split[s]) for s in split_names),
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
        (dst_root / "splits").mkdir(parents=True, exist_ok=True)
        for split in split_names:
            write_jsonl(dst_root / "splits" / f"{split}.jsonl", kept_by_split[split])
        write_jsonl(dst_root / "dataset.jsonl", all_kept)

        # Copy metadata if present, with filter summary.
        meta = {}
        if (src_root / "metadata.json").exists():
            try:
                meta = json.loads((src_root / "metadata.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        meta["style_filter"] = summary
        write_json(dst_root / "metadata.json", meta)

        # Full CSV-like JSONL report.
        write_jsonl(dst_root / "style_filter_report.jsonl", reports)
        write_json(dst_root / "style_filter_summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
