from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import load_yaml


def make_data(n_time: int, c_in: int, c_out: int, h: int, w: int, seed: int):
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(n_time, c_in, h, w)).astype("float32")

    # Add weak temporal autocorrelation so ConvLSTM has something sequence-like.
    for t in range(1, n_time):
        x[t] = 0.85 * x[t - 1] + 0.15 * x[t]

    y = np.zeros((n_time, c_out, h, w), dtype="float32")

    # Fake relationship:
    # cloud channels suppress shortwave, cloud ice/liquid influence longwave,
    # humidity/precip affect temperature. This is not physical, only smoke-test.
    cloud_total = x[:, 0] + 0.5 * x[:, 1] + 0.3 * x[:, 2] + 0.2 * x[:, 3]
    radiation_base = x[:, 8] - x[:, 9]
    longwave_base = x[:, 10] - x[:, 11]
    humidity = x[:, 15] + 0.5 * x[:, 16]
    precip = x[:, 18]

    y[:, 0] = radiation_base - 0.25 * cloud_total
    y[:, 1] = longwave_base + 0.15 * cloud_total + 0.1 * humidity
    y[:, 2] = y[:, 0] + y[:, 1]
    y[:, 3] = 0.15 * y[:, 2] + 0.1 * humidity - 0.05 * precip

    y += 0.03 * rng.normal(size=y.shape).astype("float32")
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cloud_forward_small.yaml")
    parser.add_argument("--train-time", type=int, default=700)
    parser.add_argument("--val-time", type=int, default=250)
    parser.add_argument("--test-time", type=int, default=250)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    c_in = cfg["model"]["input_channels"]
    c_out = cfg["model"]["output_channels"]

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    train_x, train_y = make_data(args.train_time, c_in, c_out, args.height, args.width, 1)
    val_x, val_y = make_data(args.val_time, c_in, c_out, args.height, args.width, 2)
    test_x, test_y = make_data(args.test_time, c_in, c_out, args.height, args.width, 3)

    np.save(out / "train_x.npy", train_x)
    np.save(out / "train_y.npy", train_y)
    np.save(out / "val_x.npy", val_x)
    np.save(out / "val_y.npy", val_y)
    np.save(out / "test_x.npy", test_x)
    np.save(out / "test_y.npy", test_y)

    print(f"Wrote dummy data to {out.resolve()}")
    print(f"train_x={train_x.shape} train_y={train_y.shape}")


if __name__ == "__main__":
    main()
