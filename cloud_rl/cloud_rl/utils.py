from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def resolve_existing_path(value: str | Path) -> Path:
    path = Path(value)
    candidates = [path]

    if not path.is_absolute():
        cwd = Path.cwd()
        module_dir = Path(__file__).resolve().parent
        package_root = module_dir.parent
        repo_root = package_root.parent
        candidates.extend(
            [
                cwd / path,
                module_dir / path,
                package_root / path,
                repo_root / path,
            ]
        )

    seen = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else candidate.resolve()
        marker = str(resolved)
        if marker in seen:
            continue
        seen.add(marker)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"Could not resolve existing path from {value!r}")


def to_device(batch: Dict, device: torch.device, non_blocking: bool = False) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v
    return out
