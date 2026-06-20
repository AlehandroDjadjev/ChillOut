from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from utils import load_yaml


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_windows(n_per_loc: int, locations: int, input_len: int, horizon: int):
    train, val, test = [], [], []
    for loc in range(locations):
        base = loc * n_per_loc
        train_cut = int(n_per_loc * 0.7)
        val_cut = int(n_per_loc * 0.85)
        for target in range(input_len, n_per_loc - horizon + 1):
            row = {
                "location": f"dummy_{loc}",
                "target_timestamp": str(target),
                "x_start": base + target - input_len,
                "x_end": base + target,
                "y_start": base + target,
                "y_end": base + target + horizon,
                "split": "train" if target < train_cut else ("val" if target < val_cut else "test"),
            }
            if row["split"] == "train":
                train.append(row)
            elif row["split"] == "val":
                val.append(row)
            else:
                test.append(row)
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cloud_forward_builder24_5day.yaml")
    parser.add_argument("--out-dir", default="data/processed_aligned")
    parser.add_argument("--locations", type=int, default=3)
    parser.add_argument("--states-per-location", type=int, default=120)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rng = np.random.default_rng(123)

    c_in = cfg["model"]["input_channels"]
    c_out = cfg["model"]["output_channels"]
    n = args.locations * args.states_per_location
    h, w = args.height, args.width

    x = rng.normal(size=(n, c_in, h, w)).astype("float32")
    for loc in range(args.locations):
        start = loc * args.states_per_location
        end = start + args.states_per_location
        for t in range(start + 1, end):
            x[t] = 0.90 * x[t - 1] + 0.10 * x[t]

    y = np.zeros((n, c_out, h, w), dtype="float32")
    cloud = x[:, 0] + 0.5*x[:, 1] + 0.4*x[:, 2] + 0.2*x[:, 4]
    radiation = x[:, 12] - x[:, 13] if c_in > 13 else x[:, 8]
    longwave = x[:, 14] - x[:, 15] if c_in > 15 else x[:, 10]
    humidity = x[:, 19] if c_in > 19 else x[:, -1]
    precip = x[:, 22] if c_in > 22 else x[:, -2]

    y[:, 0] = radiation - 0.2 * cloud
    y[:, 1] = longwave + 0.15 * cloud + 0.05 * humidity
    y[:, 2] = y[:, 0] + y[:, 1]
    y[:, 3] = 0.15 * y[:, 2] + 0.05 * humidity - 0.03 * precip
    y += 0.03 * rng.normal(size=y.shape).astype("float32")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "x_series.npy", x)
    np.save(out / "y_series.npy", y)

    train, val, test = make_windows(
        args.states_per_location,
        args.locations,
        cfg["window"]["input_len"],
        cfg["window"]["horizon"],
    )
    write_csv(out / "train_windows.csv", train)
    write_csv(out / "val_windows.csv", val)
    write_csv(out / "test_windows.csv", test)

    print(f"x_series={x.shape}")
    print(f"y_series={y.shape}")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
