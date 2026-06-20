from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from channel_spec import TARGET_CHANNELS
from dataset import IndexedWeatherWindowDataset, WeatherWindowDataset, WindowConfig
from losses import WeightedSmoothL1Loss, per_channel_mae, per_channel_rmse
from model import CloudForwardConvLSTM
from utils import count_parameters, ensure_parent, get_device, load_yaml, seed_everything


def make_loader(ds, batch_size, shuffle, num_workers, device):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def train_epoch(model, loader, loss_fn, optimizer, scaler, device, cfg):
    model.train()
    total = 0.0
    use_amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    accum_steps = int(cfg["training"].get("accumulation_steps", 1))
    grad_clip = float(cfg["training"].get("grad_clip", 0.0))

    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            pred = model(x)
            loss = loss_fn(pred, y) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total += loss.item() * accum_steps * x.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    mae_sum = None
    rmse_sum = None
    seen = 0

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)
        total += loss.item() * x.size(0)

        mae = per_channel_mae(pred, y).cpu() * x.size(0)
        rmse = per_channel_rmse(pred, y).cpu() * x.size(0)
        mae_sum = mae if mae_sum is None else mae_sum + mae
        rmse_sum = rmse if rmse_sum is None else rmse_sum + rmse
        seen += x.size(0)

    return total / len(loader.dataset), mae_sum / seen, rmse_sum / seen


def save_checkpoint(path, model, optimizer, scaler, epoch, val_loss, cfg):
    ensure_parent(path)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cloud_forward_small.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    target_names = cfg.get("channels", {}).get("target", TARGET_CHANNELS)
    seed_everything(int(cfg.get("seed", 1337)))

    device = get_device(cfg["training"]["device"])
    print(f"device={device}")

    win_cfg = WindowConfig(
        input_len=int(cfg["window"]["input_len"]),
        horizon=int(cfg["window"]["horizon"]),
        target_mode=cfg["window"]["target_mode"],
    )

    # Two supported data contracts:
    # 1) Legacy contiguous series:
    #      train_x/train_y and val_x/val_y, each [N_time,C,H,W]
    # 2) Downloader-aligned indexed series:
    #      x_series/y_series + train_index/val_index CSVs.
    #    This is required when multiple locations are concatenated, because
    #    windows must not cross location boundaries.
    if "x_series" in cfg["data"]:
        train_ds = IndexedWeatherWindowDataset(
            cfg["data"]["x_series"],
            cfg["data"]["y_series"],
            cfg["data"]["train_index"],
            target_mode=cfg["window"]["target_mode"],
        )
        val_ds = IndexedWeatherWindowDataset(
            cfg["data"]["x_series"],
            cfg["data"]["y_series"],
            cfg["data"]["val_index"],
            target_mode=cfg["window"]["target_mode"],
        )
    else:
        train_ds = WeatherWindowDataset(cfg["data"]["train_x"], cfg["data"]["train_y"], win_cfg)
        val_ds = WeatherWindowDataset(cfg["data"]["val_x"], cfg["data"]["val_y"], win_cfg)

    train_loader = make_loader(
        train_ds,
        int(cfg["training"]["batch_size"]),
        True,
        int(cfg["training"]["num_workers"]),
        device,
    )
    val_loader = make_loader(
        val_ds,
        int(cfg["training"]["batch_size"]),
        False,
        int(cfg["training"]["num_workers"]),
        device,
    )

    model = CloudForwardConvLSTM(
        input_channels=int(cfg["model"]["input_channels"]),
        output_channels=int(cfg["model"]["output_channels"]),
        encoder_channels=int(cfg["model"]["encoder_channels"]),
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    print(f"trainable_params={count_parameters(model):,}")

    loss_fn = WeightedSmoothL1Loss(cfg["loss"]["channel_weights"]).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = GradScaler(enabled=(bool(cfg["training"]["amp"]) and device.type == "cuda"))

    best_val = float("inf")
    best_epoch = 0
    patience = int(cfg["training"]["patience"])

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, cfg)
        val_loss, val_mae, val_rmse = validate(model, val_loader, loss_fn, device)

        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        for name, mae, rmse in zip(target_names, val_mae.tolist(), val_rmse.tolist()):
            print(f"  {name:34s} mae={mae:.5f} rmse={rmse:.5f}")

        save_checkpoint(
            cfg["training"]["last_checkpoint_path"],
            model,
            optimizer,
            scaler,
            epoch,
            val_loss,
            cfg,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(
                cfg["training"]["checkpoint_path"],
                model,
                optimizer,
                scaler,
                epoch,
                val_loss,
                cfg,
            )
            print(f"  saved best checkpoint val_loss={val_loss:.6f}")

        if epoch - best_epoch >= patience:
            print(f"early stopping: no improvement for {patience} epochs")
            break


if __name__ == "__main__":
    main()
