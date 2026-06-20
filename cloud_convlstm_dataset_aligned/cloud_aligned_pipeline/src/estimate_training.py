from __future__ import annotations

import argparse

from model import CloudForwardConvLSTM
from utils import count_parameters, load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cloud_forward_small.yaml")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model = CloudForwardConvLSTM(
        input_channels=cfg["model"]["input_channels"],
        output_channels=cfg["model"]["output_channels"],
        encoder_channels=cfg["model"]["encoder_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
    )

    params = count_parameters(model)
    batch = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    input_len = cfg["window"]["input_len"]
    h = 64
    w = 64

    steps_per_epoch = max(1, args.samples // batch)
    total_steps = steps_per_epoch * epochs

    print(f"trainable parameters: {params:,}")
    print(f"samples: {args.samples:,}")
    print(f"batch size: {batch}")
    print(f"epochs: {epochs}")
    print(f"steps/epoch: ~{steps_per_epoch:,}")
    print(f"total optimizer steps: ~{total_steps:,}")
    print()
    print("Rough expectation for default 64x64/40-step config:")
    print("- 2k samples: smoke/prototype, likely minutes to a few hours on GPU")
    print("- 10k samples: first serious run, often a few hours on 16GB GPU")
    print("- 30k samples: better diversity, hours to overnight depending GPU")
    print()
    print("Major cost multipliers:")
    print("- 128x128 grid is ~4x spatial compute vs 64x64")
    print("- 80 input steps is ~2x recurrent compute vs 40")
    print("- doubling hidden channels can be >2x compute")
    print("- sequence-output targets are heavier than next-24h mean targets")


if __name__ == "__main__":
    main()
