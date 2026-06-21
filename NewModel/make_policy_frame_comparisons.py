#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_latest_policy_step(images_dir: Path) -> str:
    names = []
    for p in images_dir.iterdir():
        if p.is_dir():
            m = re.match(r"policy_step_(\d+)_channel_", p.name)
            if m:
                names.append((int(m.group(1)), f"policy_step_{int(m.group(1)):02d}"))
    if not names:
        raise FileNotFoundError(f"No policy_step_XX_channel_* folders found in {images_dir}")
    return sorted(names)[-1][1]


def find_channel_dirs(images_dir: Path, prefix: str) -> List[Tuple[int, str, Path]]:
    out = []
    pat = re.compile(re.escape(prefix) + r"_channel_(\d+)_(.+)$")
    for p in images_dir.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), m.group(2), p))
    return sorted(out, key=lambda x: x[0])


def load_frames(channel_dir: Path) -> List[Image.Image]:
    frames = []
    for i in range(4):
        fp = channel_dir / f"frame_{i:02d}.png"
        if not fp.exists():
            raise FileNotFoundError(fp)
        frames.append(Image.open(fp).convert("L"))
    return frames


def signed_diff_rgb(inp: Image.Image, out: Image.Image, scale: float = 0.35) -> Image.Image:
    a = np.asarray(inp).astype(np.float32) / 255.0
    b = np.asarray(out).astype(np.float32) / 255.0
    d = b - a
    pos = np.clip(d / scale, 0.0, 1.0)
    neg = np.clip(-d / scale, 0.0, 1.0)
    rgb = np.zeros((*d.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (pos * 255).astype(np.uint8)
    rgb[..., 2] = (neg * 255).astype(np.uint8)
    rgb[..., 1] = (np.clip(1.0 - np.maximum(pos, neg), 0.0, 1.0) * 40).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def to_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB")


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str):
    draw.text(xy, text, fill=(255, 255, 255))


def make_sheet(
    input_frames: List[Image.Image],
    output_frames: List[Image.Image],
    title: str,
    out_path: Path,
    cell_border: int = 3,
):
    # Rows: input, output, signed diff
    assert len(input_frames) == 4 and len(output_frames) == 4
    w, h = input_frames[0].size
    label_w = 120
    top_h = 34
    row_gap = 20
    col_gap = 8

    sheet_w = label_w + 4 * w + 3 * col_gap
    sheet_h = top_h + 3 * h + 2 * row_gap

    canvas = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (8, 8), title)

    rows = [
        ("INPUT", [to_rgb(x) for x in input_frames]),
        ("OUTPUT", [to_rgb(x) for x in output_frames]),
        ("DIFF red+ blue-", [signed_diff_rgb(i, o) for i, o in zip(input_frames, output_frames)]),
    ]

    y = top_h
    for label, frames in rows:
        draw_text(draw, (8, y + h // 2 - 8), label)
        x = label_w
        for j, im in enumerate(frames):
            canvas.paste(im, (x, y))
            draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(140, 140, 140), width=1)
            draw_text(draw, (x + 4, y + 4), f"t{j}")
            x += w + col_gap
        y += h + row_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def copy_numeric(test_dir: Path, out_dir: Path):
    keep = [
        "result.json",
        "policy_steps.csv",
        "image_channel_stats.csv",
        "image_diff_stats.csv",
        "cloud_scalar_changes.csv",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in keep:
        src = test_dir / name
        if src.exists():
            dst = out_dir / name
            dst.write_bytes(src.read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True, help="Existing diagnostics folder from mixed/forced test.")
    ap.add_argument("--input-name", default="mixed_init", help="Usually mixed_init or original.")
    ap.add_argument("--output-name", default="auto", help="Usually auto, policy_step_03, or final.")
    ap.add_argument("--channels", default="0,1,2,4,5,7", help="Comma list or 'all'.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    test_dir = Path(args.test_dir).resolve()
    images_dir = test_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    output_name = find_latest_policy_step(images_dir) if args.output_name == "auto" else args.output_name
    out_dir = Path(args.out_dir).resolve() if args.out_dir else test_dir / "comparison_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dirs = find_channel_dirs(images_dir, args.input_name)
    output_dirs = find_channel_dirs(images_dir, output_name)
    out_map = {ch: (name, path) for ch, name, path in output_dirs}

    if args.channels.strip().lower() == "all":
        wanted = [ch for ch, _, _ in input_dirs]
    else:
        wanted = [int(x.strip()) for x in args.channels.split(",") if x.strip()]

    made = []
    for ch, cname, in_dir in input_dirs:
        if ch not in wanted:
            continue
        if ch not in out_map:
            print(f"SKIP channel {ch}: no output folder")
            continue
        out_cname, out_dir_ch = out_map[ch]
        inp_frames = load_frames(in_dir)
        out_frames = load_frames(out_dir_ch)
        out_path = out_dir / f"comparison_channel_{ch:02d}_{cname}.png"
        make_sheet(inp_frames, out_frames, f"{args.input_name} -> {output_name} | channel {ch}: {cname}", out_path)
        made.append(str(out_path))

    copy_numeric(test_dir, out_dir)

    summary = {
        "test_dir": str(test_dir),
        "input_name": args.input_name,
        "output_name": output_name,
        "channels": wanted,
        "comparison_images": made,
        "numeric_files_copied": [
            name for name in [
                "result.json",
                "policy_steps.csv",
                "image_channel_stats.csv",
                "image_diff_stats.csv",
                "cloud_scalar_changes.csv",
            ] if (out_dir / name).exists()
        ],
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
