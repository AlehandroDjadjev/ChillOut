#!/usr/bin/env python3

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Iterable, List


FIRST_MODEL_FILES = ["best.pt", "last.pt", "history.json", "test_metrics.json", "run_config.json"]
RL_MODEL_FILES = ["policy_latest.pt", "history.tail.json", "config.resolved.json"]


def expand_source(source: Path, preferred_files: List[str], include_globs: List[str] | None = None) -> List[Path]:
    if source.is_file():
        return [source]

    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    files: List[Path] = []
    for name in preferred_files:
        candidate = source / name
        if candidate.exists():
            files.append(candidate)

    if include_globs:
        for pattern in include_globs:
            files.extend(sorted(source.glob(pattern)))

    return files


def write_zip(out_path: Path, sections: List[tuple[str, List[Path]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for prefix, files in sections:
            for file in files:
                arcname = f"{prefix}/{file.name}" if prefix else file.name
                archive.write(file, arcname=arcname)


def main() -> int:
    parser = argparse.ArgumentParser(description="Zip the first-model and RL-model checkpoints together.")
    parser.add_argument("--first-model", default="runs/cloud_temp", help="First-model run dir or checkpoint file.")
    parser.add_argument("--rl-model", default="runs/cloud_rl", help="RL run dir or checkpoint file.")
    parser.add_argument("--out", default="exports/model_weights.zip", help="Output zip file.")
    parser.add_argument(
        "--include-rl-snapshots",
        action="store_true",
        help="Also include policy_update_*.pt snapshots from the RL run directory.",
    )
    args = parser.parse_args()

    first_source = Path(args.first_model)
    rl_source = Path(args.rl_model)
    out_path = Path(args.out)

    first_files = expand_source(first_source, FIRST_MODEL_FILES)
    rl_files = expand_source(
        rl_source,
        RL_MODEL_FILES,
        include_globs=["policy_update_*.pt"] if args.include_rl_snapshots else None,
    )

    if not first_files:
        raise SystemExit(f"No first-model checkpoint files found at {first_source}")
    if not rl_files:
        raise SystemExit(f"No RL checkpoint files found at {rl_source}")

    sections = [
        ("first_model", first_files),
        ("rl_model", rl_files),
    ]
    write_zip(out_path, sections)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
