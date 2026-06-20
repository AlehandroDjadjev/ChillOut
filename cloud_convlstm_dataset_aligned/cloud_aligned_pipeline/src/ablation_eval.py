import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import WindowedTensorDataset
from losses import WeightedChannelMSELoss
from model import ConvLSTMForecaster


@torch.no_grad()
def eval_variant(model, loader, loss_fn, device, variant, cloud_indices):
    model.eval()
    total_loss = 0.0
    total_count = 0
    mae_sum = None

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if variant == "no_cloud":
            x = x.clone()
            x[:, :, cloud_indices, :, :] = 0.0

        elif variant == "cloud_shuffle":
            x = x.clone()
            perm = torch.randperm(x.shape[0], device=x.device)
            shuffled = x[perm].clone()
            x[:, :, cloud_indices, :, :] = shuffled[:, :, cloud_indices, :, :]

        pred = model(x)
        loss = loss_fn(pred, y)

        b = x.shape[0]
        total_loss += float(loss.item()) * b
        total_count += b

        batch_mae = torch.mean(torch.abs(pred - y), dim=(0, 2, 3)).detach().cpu().numpy()
        if mae_sum is None:
            mae_sum = batch_mae * b
        else:
            mae_sum += batch_mae * b

    return total_loss / max(1, total_count), mae_sum / max(1, total_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["processed_dir"])
    manifest = yaml.safe_load((data_dir / "tensor_manifest.json").read_text(encoding="utf-8"))

    input_len = int(cfg["data"]["input_len"])
    window_csv = data_dir / f"{args.split}_windows.csv"

    ds = WindowedTensorDataset(
        x_path=data_dir / "x_series.npy",
        y_path=data_dir / "y_series.npy",
        windows_csv=window_csv,
        input_len=input_len,
    )

    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 4)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMForecaster(
        input_channels=int(cfg["model"]["input_channels"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        kernel_size=int(cfg["model"].get("kernel_size", 3)),
        output_channels=int(cfg["model"]["output_channels"]),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    output_channel_names = manifest.get("output_channel_names", [
        "shortwave_anomaly_next_5day",
        "longwave_anomaly_next_5day",
        "net_radiation_anomaly_next_5day",
        "temperature_anomaly_next_5day",
    ])

    weights = torch.tensor(cfg["loss"].get("channel_weights", [1.0] * int(cfg["model"]["output_channels"])), dtype=torch.float32)
    loss_fn = WeightedChannelMSELoss(weights=weights).to(device)

    cloud_indices = cfg.get("ablation", {}).get("cloud_channel_indices", list(range(12)))

    for variant in ["normal", "no_cloud", "cloud_shuffle"]:
        loss, mae = eval_variant(model, loader, loss_fn, device, variant, cloud_indices)
        print(f"variant={variant:<13} loss={loss:.6f}")
        for name, value in zip(output_channel_names, mae):
            print(f"  {name:<35} mae={value:.5f}")


if __name__ == "__main__":
    main()
