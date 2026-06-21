#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader


def import_base():
    names = [
        "train_cloud_temp_cloudforced_radiation_v6",
        "train_cloud_temp_cloudforced_radiation_v6_FIXED2",
        "train_cloud_temp_cloudforced_radiation_v6_FIXED3",
        "train_cloud_temp_hybrid_convlstm_v6",
    ]
    last = None
    for n in names:
        try:
            return __import__(n, fromlist=["dummy"])
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Could not import v6 base utilities. Put this file next to train_cloud_temp_cloudforced_radiation_v6.py. Last error: {last}")


base = import_base()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def strip_prefix(state):
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def record_context_features(record: Dict[str, Any]) -> np.ndarray:
    """Static scene context that should stay fixed in the controller.

    Crucially, this does NOT include observed shortwave radiation, because
    observed shortwave is part of the target construction:
      cloud_loss = clear_sky_proxy - observed_shortwave

    We give the model clear-sky availability and solar/time geometry.
    """
    rb = base.radiation_bundle(record)
    clear = float(rb["clear_wm2"])
    valid = float(rb["valid"])
    lat = float(record.get("lat", 0.0))
    lon = float(record.get("lon", 0.0))

    raw = str(record.get("anchor") or record.get("date") or "")
    try:
        import datetime as _dt
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        dt = dt.astimezone(_dt.timezone.utc)
        doy = int(dt.strftime("%j"))
        hour = dt.hour + dt.minute / 60.0
    except Exception:
        doy = 180
        hour = 12.0

    day_ang = 2.0 * math.pi * doy / 365.0
    hour_ang = 2.0 * math.pi * hour / 24.0

    return np.asarray([
        clear / 1000.0,
        valid,
        math.sin(day_ang),
        math.cos(day_ang),
        math.sin(hour_ang),
        math.cos(hour_ang),
        lat / 90.0,
        lon / 180.0,
    ], dtype=np.float32)



def train_radiation_stats(records: List[Dict[str, Any]]) -> Dict[str, float]:
    vals_loss = []
    vals_attn = []
    vals_clear = []
    for r in records:
        rb = base.radiation_bundle(r)
        if float(rb["valid"]) > 0.5:
            vals_loss.append(float(rb["loss_wm2"]))
            vals_attn.append(float(max(0.0, min(1.2, rb["attenuation"]))))
            vals_clear.append(float(rb["clear_wm2"]))
    if not vals_loss:
        return {"mean_loss": 120.0, "mean_attn": 0.20, "mean_clear": 600.0, "n_valid": 0}
    return {
        "mean_loss": float(np.mean(vals_loss)),
        "std_loss": float(np.std(vals_loss)),
        "mean_attn": float(np.mean(vals_attn)),
        "std_attn": float(np.std(vals_attn)),
        "mean_clear": float(np.mean(vals_clear)),
        "n_valid": int(len(vals_loss)),
    }


def cloud_fraction_from_record(record: Dict[str, Any]) -> float:
    inputs = record.get("inputs") or {}
    if "cloud_s2_fraction" in inputs:
        return float(inputs.get("cloud_s2_fraction", 0.0))
    return 0.0


def clean_radiation_records(records: List[Dict[str, Any]], args, split_name: str) -> List[Dict[str, Any]]:
    """Drop obvious label contradictions before training/eval.

    These are not model errors; they are target construction/pathology cases:
    - observed shortwave greater than clear-sky proxy
    - negative cloud loss / attenuation
    - very cloudy Sentinel scene but almost no estimated radiation loss
    - very low clear-sky scenes where W/m2 target is weak/noisy
    """
    out = []
    counts = {
        "input": len(records),
        "drop_invalid": 0,
        "drop_low_clear": 0,
        "drop_negative_or_obs_gt_clear": 0,
        "drop_high_cloud_low_loss": 0,
        "drop_low_cloud_high_loss": 0,
        "drop_loss_outside_range": 0,
    }
    for r in records:
        rb = base.radiation_bundle(r)
        loss = float(rb["loss_wm2"])
        clear = float(rb["clear_wm2"])
        obs = float(rb["observed_wm2"])
        valid = float(rb["valid"])
        cf = cloud_fraction_from_record(r)

        if args.clean_drop_invalid and valid < 0.5:
            counts["drop_invalid"] += 1
            continue
        if clear < args.min_clear_wm2:
            counts["drop_low_clear"] += 1
            continue
        if args.clean_drop_negative and (loss < 0.0 or obs > clear):
            counts["drop_negative_or_obs_gt_clear"] += 1
            continue
        if args.clean_drop_high_cloud_low_loss and cf >= args.high_cloud_thresh and loss <= args.low_loss_thresh_wm2:
            counts["drop_high_cloud_low_loss"] += 1
            continue
        if args.clean_drop_low_cloud_high_loss and cf <= args.low_cloud_thresh and loss >= args.high_loss_thresh_wm2:
            counts["drop_low_cloud_high_loss"] += 1
            continue
        if loss < args.min_loss_wm2 or loss > args.max_loss_wm2:
            counts["drop_loss_outside_range"] += 1
            continue
        out.append(r)

    counts["kept"] = len(out)
    print(f"[clean {split_name}] " + " ".join(f"{k}={v}" for k, v in counts.items()))
    return out


def load_cloud_tensor_or_mask(root: Path, record: Dict[str, Any], h: int, w: int, use_cloud_tensor: bool) -> np.ndarray:
    """Return [C,H,W]. Prefer the original Sentinel tensor if present."""
    if use_cloud_tensor and record.get("cloud_tensor_path"):
        p = root / str(record["cloud_tensor_path"])
        if p.exists():
            arr = np.load(p)["cloud_tensor"].astype(np.float32)  # [H,W,8]
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr, 0.0, None)
            chans = []
            for c in range(arr.shape[-1]):
                im = Image.fromarray(np.clip(arr[..., c], 0, 1) * 255.0).convert("L")
                if im.size != (w, h):
                    im = im.resize((w, h), Image.BILINEAR)
                chans.append(np.asarray(im, dtype=np.float32) / 255.0)
            return np.stack(chans, 0).astype(np.float32)

    # Fallback: single-channel cloud mask PNG.
    img = Image.open(root / str(record["mask_path"])).convert("L")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, ...]


class RadiationSequenceDataset(Dataset):
    def __init__(self, root: Path, records: List[Dict[str, Any]], raw_names: List[str],
                 cloud_names: List[str], x_norm, image_height: int, image_width: int,
                 lookback: int, max_gap_days: float, use_cloud_tensor: bool, augment: bool,
                 min_clear_wm2: float = 120.0, target_min_attenuation: float = 0.0,
                 target_max_attenuation: float = 1.2, target_min_loss_wm2: float = -100.0,
                 target_max_loss_wm2: Optional[float] = None):
        self.root = root
        self.raw_names = raw_names
        self.cloud_names = cloud_names
        self.x_norm = x_norm
        self.image_height = image_height
        self.image_width = image_width
        self.use_cloud_tensor = use_cloud_tensor
        self.augment = augment
        self.min_clear_wm2 = float(min_clear_wm2)
        self.target_min_attenuation = float(target_min_attenuation)
        self.target_max_attenuation = float(target_max_attenuation)
        self.target_min_loss_wm2 = float(target_min_loss_wm2)
        self.target_max_loss_wm2 = None if target_max_loss_wm2 is None else float(target_max_loss_wm2)
        raw_to_idx = {n: i for i, n in enumerate(raw_names)}
        self.cloud_idx = [raw_to_idx[n] for n in cloud_names]
        self.windows = base.build_windows(records, lookback, max_gap_days)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]
        imgs, feats = [], []
        for r in win.records:
            arr = load_cloud_tensor_or_mask(self.root, r, self.image_height, self.image_width, self.use_cloud_tensor)
            if self.augment:
                if random.random() < 0.5:
                    arr = np.ascontiguousarray(arr[..., ::-1])
                if random.random() < 0.15:
                    arr = np.clip(arr + np.random.normal(0, 0.01, arr.shape).astype(np.float32), 0, 1)
            imgs.append(torch.from_numpy(arr))
            x = self.x_norm.transform(base.feature_vector(r, self.raw_names))
            feats.append(x[self.cloud_idx])
        last = win.records[-1]
        rb = base.radiation_bundle(last)
        clear = float(rb["clear_wm2"])
        loss = float(rb["loss_wm2"])
        valid = float(rb["valid"])
        if clear < self.min_clear_wm2:
            valid = 0.0
        attn = float(rb["attenuation"])
        # Robust bounded targets.
        attn = float(max(self.target_min_attenuation, min(self.target_max_attenuation, attn)))
        max_loss = max(clear, 0.0) if self.target_max_loss_wm2 is None else self.target_max_loss_wm2
        loss = float(max(self.target_min_loss_wm2, min(max_loss, loss)))
        return {
            "image": torch.stack(imgs, 0),                         # [T,C,H,W]
            "cloud_features": torch.from_numpy(np.stack(feats).astype(np.float32)),
            "context_features": torch.from_numpy(record_context_features(last)),
            "target_loss_wm2": torch.tensor([loss], dtype=torch.float32),
            "target_attenuation": torch.tensor([attn], dtype=torch.float32),
            "clear_wm2": torch.tensor([clear], dtype=torch.float32),
            "valid": torch.tensor([valid], dtype=torch.float32),
            "sample_id": str(last.get("sample_id", idx)),
            "location": str(last.get("location", last.get("city", ""))),
            "anchor": str(last.get("anchor", last.get("date", ""))),
        }


class ResBlock(nn.Module):
    def __init__(self, c: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
        )
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.net(x))


class Down(nn.Module):
    def __init__(self, a: int, b: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.SiLU(inplace=True),
            ResBlock(b, dropout),
        )
    def forward(self, x):
        return self.net(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid: int, k: int = 3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k // 2)
    def forward(self, x, state):
        if state is None:
            b, _, h, w = x.shape
            h0 = torch.zeros(b, self.hid, h, w, dtype=x.dtype, device=x.device)
            c0 = torch.zeros_like(h0)
        else:
            h0, c0 = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h0], 1)), 4, 1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c1 = f * c0 + i * g
        h1 = o * torch.tanh(c1)
        return h1, c1


class ConvLSTM(nn.Module):
    def __init__(self, in_ch: int, hidden: List[int], dropout: float):
        super().__init__()
        dims = [in_ch] + hidden
        self.cells = nn.ModuleList([ConvLSTMCell(dims[i], dims[i+1]) for i in range(len(hidden))])
        self.drop = nn.Dropout2d(dropout)
    def forward(self, x):
        layer = x
        last = None
        for li, cell in enumerate(self.cells):
            state = None
            outs = []
            for t in range(layer.size(1)):
                state = cell(layer[:, t], state)
                h, _ = state
                if li < len(self.cells) - 1:
                    h = self.drop(h)
                outs.append(h)
            layer = torch.stack(outs, 1)
            last = outs[-1]
        return last


class AttnPool(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.score = nn.Conv2d(c, 1, 1)
        self.norm = nn.LayerNorm(c)
    def forward(self, x):
        b, c, h, w = x.shape
        a = torch.softmax(self.score(x).view(b, 1, h*w), -1)
        v = x.view(b, c, h*w)
        return self.norm(torch.bmm(v, a.transpose(1, 2)).squeeze(-1))


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: List[int], out: int, dropout: float):
        super().__init__()
        layers = []
        p = inp
        for h in hidden:
            layers += [nn.Linear(p, h), nn.LayerNorm(h), nn.SiLU(inplace=True), nn.Dropout(dropout)]
            p = h
        layers += [nn.Linear(p, out), nn.LayerNorm(out), nn.SiLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class CloudRadiationV8CleanDirect(nn.Module):
    """Radiation-only bottom model.

    Design:
      cloud image/scalars -> attenuation fraction
      attenuation * clear_sky_context -> W/m2 loss

    This separates:
      cloud property = attenuation
      scene energy availability = clear sky W/m2
    """

    def __init__(self, in_channels: int, num_cloud_features: int, stem_dim: int = 64,
                 convlstm_dims: Optional[List[int]] = None, cloud_dim: int = 128,
                 context_dim: int = 64, fusion_dim: int = 192, dropout: float = 0.12,
                 loss_scale: float = 300.0, init_attn_prior: float = 0.20,
                 residual_factor: float = 0.05, init_loss_prior_wm2: float = 140.0,
                 max_direct_loss_wm2: float = 900.0, final_blend_direct: float = 1.0):
        super().__init__()
        convlstm_dims = convlstm_dims or [64, 96]
        self.loss_scale = float(loss_scale)
        self.residual_factor = float(residual_factor)
        self.max_direct_loss_wm2 = float(max_direct_loss_wm2)
        self.final_blend_direct = float(final_blend_direct)
        self.stem = nn.Sequential(
            Down(in_channels, 32, dropout * 0.25),
            Down(32, 48, dropout * 0.25),
            Down(48, stem_dim, dropout * 0.25),
        )
        self.convlstm = ConvLSTM(stem_dim, convlstm_dims, dropout * 0.5)
        self.pool = AttnPool(convlstm_dims[-1])
        self.image_proj = MLP(convlstm_dims[-1], [fusion_dim], fusion_dim, dropout)

        self.cloud_lstm = nn.LSTM(num_cloud_features, cloud_dim, batch_first=True)
        self.cloud_proj = MLP(cloud_dim, [fusion_dim], fusion_dim, dropout)

        self.context_proj = MLP(8, [context_dim], fusion_dim, dropout)

        self.fuse = MLP(fusion_dim * 5, [fusion_dim, fusion_dim], fusion_dim, dropout)
        self.attn_head = nn.Sequential(
            nn.Linear(fusion_dim, 96), nn.LayerNorm(96), nn.SiLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(96, 1)
        )
        self.loss_residual_head = nn.Sequential(
            nn.Linear(fusion_dim, 96), nn.LayerNorm(96), nn.SiLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(96, 1)
        )
        self.direct_loss_head = nn.Sequential(
            nn.Linear(fusion_dim, 96), nn.LayerNorm(96), nn.SiLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(96, 1)
        )
        # Important: attenuation = 1.2 * sigmoid(logit).
        # A zero bias means attenuation starts at 0.6, which predicts massive
        # cloud loss for every scene. Initialize to the training-set mean
        # attenuation so epoch 1 starts from a sane constant baseline.
        prior = float(max(0.02, min(1.15, init_attn_prior)))
        p = max(1e-4, min(1.0 - 1e-4, prior / 1.2))
        bias = math.log(p / (1.0 - p))
        nn.init.constant_(self.attn_head[-1].bias, bias)
        nn.init.normal_(self.attn_head[-1].weight, 0.0, 1e-4)
        nn.init.zeros_(self.loss_residual_head[-1].bias)
        nn.init.normal_(self.loss_residual_head[-1].weight, 0.0, 1e-4)

        # Direct cloud-loss head starts at train mean cloud loss.
        q = max(1e-4, min(1.0 - 1e-4, float(init_loss_prior_wm2) / max(1e-6, self.max_direct_loss_wm2)))
        direct_bias = math.log(q / (1.0 - q))
        nn.init.constant_(self.direct_loss_head[-1].bias, direct_bias)
        nn.init.normal_(self.direct_loss_head[-1].weight, 0.0, 1e-4)

    def encode_image(self, image):
        b, t, c, h, w = image.shape
        x = image.reshape(b*t, c, h, w).contiguous(memory_format=torch.channels_last)
        fmap = self.stem(x)
        _, cc, hh, ww = fmap.shape
        seq = fmap.view(b, t, cc, hh, ww)
        return self.image_proj(self.pool(self.convlstm(seq)))

    def forward(self, image, cloud_features, context_features, clear_wm2):
        image_state = self.encode_image(image)
        _, (hc, _) = self.cloud_lstm(cloud_features)
        cloud_state = self.cloud_proj(hc[-1])
        context_state = self.context_proj(context_features)

        z = self.fuse(torch.cat([
            image_state,
            cloud_state,
            context_state,
            image_state * context_state,
            cloud_state * context_state,
        ], -1))

        attenuation = 1.2 * torch.sigmoid(self.attn_head(z))
        loss_from_attenuation = attenuation * clear_wm2
        residual = self.loss_scale * torch.tanh(self.loss_residual_head(z))
        physics_loss_wm2 = loss_from_attenuation + self.residual_factor * residual
        direct_loss_wm2 = self.max_direct_loss_wm2 * torch.sigmoid(self.direct_loss_head(z))
        blend = max(0.0, min(1.0, self.final_blend_direct))
        loss_wm2 = blend * direct_loss_wm2 + (1.0 - blend) * physics_loss_wm2
        return {
            "loss_wm2": loss_wm2,
            "direct_loss_wm2": direct_loss_wm2,
            "physics_loss_wm2": physics_loss_wm2,
            "attenuation": attenuation,
            "loss_from_attenuation": loss_from_attenuation,
            "residual_wm2": residual,
        }


@torch.no_grad()
def metric(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    p = pred.float().view(-1)
    y = target.float().view(-1)
    if p.numel() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "corr": float("nan")}
    e = p - y
    out = {
        "mae": float(e.abs().mean()),
        "rmse": float(torch.sqrt((e * e).mean())),
        "bias": float(e.mean()),
    }
    if p.numel() > 1 and float(p.std()) > 1e-8 and float(y.std()) > 1e-8:
        out["corr"] = float(torch.corrcoef(torch.stack([p, y]))[0, 1])
    else:
        out["corr"] = float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    pred_loss, targ_loss = [], []
    pred_attn, targ_attn = [], []
    from_attn = []
    for batch in loader:
        image = batch["image"].to(device)
        cloud = batch["cloud_features"].to(device)
        ctx = batch["context_features"].to(device)
        clear = batch["clear_wm2"].to(device)
        y_loss = batch["target_loss_wm2"].to(device)
        y_attn = batch["target_attenuation"].to(device)
        valid = batch["valid"].to(device)
        with autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
            out = model(image, cloud, ctx, clear)
        mask = (valid.view(-1) > 0.5).detach().cpu()
        if mask.any():
            pred_loss.append(out["loss_wm2"].detach().float().view(-1).cpu()[mask])
            from_attn.append(out["loss_from_attenuation"].detach().float().view(-1).cpu()[mask])
            targ_loss.append(y_loss.detach().float().view(-1).cpu()[mask])
            pred_attn.append(out["attenuation"].detach().float().view(-1).cpu()[mask])
            targ_attn.append(y_attn.detach().float().view(-1).cpu()[mask])
    p = torch.cat(pred_loss) if pred_loss else torch.empty(0)
    y = torch.cat(targ_loss) if targ_loss else torch.empty(0)
    pa = torch.cat(pred_attn) if pred_attn else torch.empty(0)
    ya = torch.cat(targ_attn) if targ_attn else torch.empty(0)
    pfa = torch.cat(from_attn) if from_attn else torch.empty(0)
    out = {"loss": metric(p, y), "attenuation": metric(pa, ya), "loss_from_attenuation": metric(pfa, y), "n": int(p.numel())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-height", type=int, default=160)
    ap.add_argument("--image-width", type=int, default=160)
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--max-gap-days", type=float, default=12.0)
    ap.add_argument("--use-cloud-tensor", action="store_true", help="Use 8-channel Sentinel tensor if cloud_tensor_path exists.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=3e-4)
    ap.add_argument("--stem-dim", type=int, default=64)
    ap.add_argument("--convlstm-dims", default="64,96")
    ap.add_argument("--cloud-dim", type=int, default=128)
    ap.add_argument("--context-dim", type=int, default=64)
    ap.add_argument("--fusion-dim", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.12)
    ap.add_argument("--loss-scale", type=float, default=300.0)
    ap.add_argument("--min-clear-wm2", type=float, default=120.0, help="Ignore very low clear-sky scenes in radiation loss/eval because they are noisy and weakly controllable.")
    ap.add_argument("--target-min-attenuation", type=float, default=0.0)
    ap.add_argument("--target-max-attenuation", type=float, default=1.2)
    ap.add_argument("--target-min-loss-wm2", type=float, default=-100.0)
    ap.add_argument("--target-max-loss-wm2", type=float, default=None, help="Clamp target cloud loss upper bound. Default keeps old behavior: max(clear_wm2, 0).")
    ap.add_argument("--residual-factor", type=float, default=0.05, help="Auxiliary physics-head residual strength after attenuation*clear_sky.")
    ap.add_argument("--max-direct-loss-wm2", type=float, default=900.0)
    ap.add_argument("--final-blend-direct", type=float, default=1.0, help="1.0 means final prediction is direct cloud loss. 0.0 means final prediction is physics attenuation head.")
    ap.add_argument("--clean-drop-invalid", action="store_true", default=True)
    ap.add_argument("--clean-drop-negative", action="store_true", default=True)
    ap.add_argument("--clean-drop-high-cloud-low-loss", action="store_true", default=True)
    ap.add_argument("--clean-drop-low-cloud-high-loss", action="store_true", default=False)
    ap.add_argument("--high-cloud-thresh", type=float, default=0.65)
    ap.add_argument("--low-cloud-thresh", type=float, default=0.05)
    ap.add_argument("--low-loss-thresh-wm2", type=float, default=25.0)
    ap.add_argument("--high-loss-thresh-wm2", type=float, default=280.0)
    ap.add_argument("--min-loss-wm2", type=float, default=0.0)
    ap.add_argument("--max-loss-wm2", type=float, default=900.0)
    ap.add_argument("--attenuation-weight", type=float, default=0.75)
    ap.add_argument("--from-attenuation-weight", type=float, default=0.25)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=30)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = base.load_metadata(root)
    raw_names = list(meta["raw_feature_names"])
    cloud_names = list(meta["cloud_feature_names"])
    train_records = base.load_records(root, "train")
    val_records = base.load_records(root, "val")
    test_records = base.load_records(root, "test")

    train_records = clean_radiation_records(train_records, args, "train")
    val_records = clean_radiation_records(val_records, args, "val")
    test_records = clean_radiation_records(test_records, args, "test")
    rad_stats = train_radiation_stats(train_records)

    x_norm = base.Normalizer.fit(np.asarray([base.feature_vector(r, raw_names) for r in train_records], dtype=np.float32))

    ds_args = dict(
        root=root, raw_names=raw_names, cloud_names=cloud_names, x_norm=x_norm,
        image_height=args.image_height, image_width=args.image_width,
        lookback=args.lookback, max_gap_days=args.max_gap_days,
        use_cloud_tensor=args.use_cloud_tensor,
        min_clear_wm2=args.min_clear_wm2,
        target_min_attenuation=args.target_min_attenuation,
        target_max_attenuation=args.target_max_attenuation,
        target_min_loss_wm2=args.target_min_loss_wm2,
        target_max_loss_wm2=args.target_max_loss_wm2,
    )
    train_ds = RadiationSequenceDataset(records=train_records, augment=args.augment, **ds_args)
    val_ds = RadiationSequenceDataset(records=val_records, augment=False, **ds_args)
    test_ds = RadiationSequenceDataset(records=test_records, augment=False, **ds_args)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # Determine in_channels from first sample.
    in_channels = int(train_ds[0]["image"].shape[1])
    model = CloudRadiationV8CleanDirect(
        in_channels=in_channels,
        num_cloud_features=len(cloud_names),
        stem_dim=args.stem_dim,
        convlstm_dims=[int(x) for x in args.convlstm_dims.split(",") if x.strip()],
        cloud_dim=args.cloud_dim,
        context_dim=args.context_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        loss_scale=args.loss_scale,
        init_attn_prior=rad_stats["mean_attn"],
        residual_factor=args.residual_factor,
        init_loss_prior_wm2=rad_stats["mean_loss"],
        max_direct_loss_wm2=args.max_direct_loss_wm2,
        final_blend_direct=args.final_blend_direct,
    ).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.55, patience=8)
    scaler = GradScaler("cuda", enabled=(args.amp_dtype == "fp16" and device.type == "cuda"))

    print(json.dumps({
        "architecture": "CloudRadiationV8CleanDirect",
        "device": str(device),
        "in_channels": in_channels,
        "use_cloud_tensor": args.use_cloud_tensor,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "cloud_features": cloud_names,
        "context_features": ["clear_sky_proxy/1000", "daylight_valid", "sin_day", "cos_day", "sin_hour", "cos_hour", "lat/90", "lon/180"],
        "target": "radiation_cloud_loss_wm2 via attenuation * clear_sky",
        "train_radiation_stats": rad_stats,
        "initial_attenuation_prior": rad_stats["mean_attn"],
        "min_clear_wm2": args.min_clear_wm2,
        "target_clamps": {
            "min_attenuation": args.target_min_attenuation,
            "max_attenuation": args.target_max_attenuation,
            "min_loss_wm2": args.target_min_loss_wm2,
            "max_loss_wm2": args.target_max_loss_wm2,
        },
        "residual_factor": args.residual_factor,
        "max_direct_loss_wm2": args.max_direct_loss_wm2,
        "final_blend_direct": args.final_blend_direct,
        "cleaning": {
            "min_clear_wm2": args.min_clear_wm2,
            "drop_negative": args.clean_drop_negative,
            "drop_high_cloud_low_loss": args.clean_drop_high_cloud_low_loss,
            "min_loss_wm2": args.min_loss_wm2,
            "max_loss_wm2": args.max_loss_wm2
        },
    }, indent=2))

    best = float("inf")
    bad = 0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            image = batch["image"].to(device)
            cloud = batch["cloud_features"].to(device)
            ctx = batch["context_features"].to(device)
            clear = batch["clear_wm2"].to(device)
            y_loss = batch["target_loss_wm2"].to(device)
            y_attn = batch["target_attenuation"].to(device)
            valid = batch["valid"].to(device)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
                out = model(image, cloud, ctx, clear)
                v = valid.view(-1, 1)
                # Normalize W/m2 loss so attenuation and loss have comparable scale.
                loss_main = (F.smooth_l1_loss(out["loss_wm2"] / args.loss_scale, y_loss / args.loss_scale, beta=0.25, reduction="none") * v).sum() / v.sum().clamp_min(1.0)
                loss_attn = (F.smooth_l1_loss(out["attenuation"], y_attn, beta=0.08, reduction="none") * v).sum() / v.sum().clamp_min(1.0)
                loss_from_attn = (F.smooth_l1_loss(out["loss_from_attenuation"] / args.loss_scale, y_loss / args.loss_scale, beta=0.25, reduction="none") * v).sum() / v.sum().clamp_min(1.0)
                loss = loss_main + args.attenuation_weight * loss_attn + args.from_attenuation_weight * loss_from_attn

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                opt.step()

            with torch.no_grad():
                mask = valid.view(-1) > 0.5
                mae = (out["loss_wm2"].detach().float().view(-1)[mask] - y_loss.float().view(-1)[mask]).abs().mean() if mask.any() else torch.tensor(0.0)
            bs = image.size(0)
            total_loss += float(loss.detach()) * bs
            total_mae += float(mae) * bs
            n += bs
            pbar.set_postfix(loss=total_loss/max(1,n), mae_wm2=total_mae/max(1,n))

        val = evaluate(model, val_loader, device, amp_dtype)
        sched.step(val["loss"]["mae"])
        row = {"epoch": epoch, "train_loss": total_loss/max(1,n), "train_mae_wm2": total_mae/max(1,n), "val": val, "lr": opt.param_groups[0]["lr"]}
        hist.append(row)
        write_json(out_dir / "history.json", {"history": hist})
        print(f"epoch={epoch:03d} train_mae={row['train_mae_wm2']:.2f}W/m2 val_mae={val['loss']['mae']:.2f}W/m2 val_rmse={val['loss']['rmse']:.2f} corr={val['loss']['corr']:.3f} attn_mae={val['attenuation']['mae']:.3f} lr={row['lr']:.2e}")

        ckpt = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "raw_feature_names": raw_names,
            "cloud_feature_names": cloud_names,
            "normalizer": x_norm.state_dict(),
            "image_height": args.image_height,
            "image_width": args.image_width,
            "lookback": args.lookback,
            "in_channels": in_channels,
            "model_kwargs": {
                "in_channels": in_channels,
                "num_cloud_features": len(cloud_names),
                "stem_dim": args.stem_dim,
                "convlstm_dims": [int(x) for x in args.convlstm_dims.split(",") if x.strip()],
                "cloud_dim": args.cloud_dim,
                "context_dim": args.context_dim,
                "fusion_dim": args.fusion_dim,
                "dropout": args.dropout,
                "loss_scale": args.loss_scale,
                "init_attn_prior": rad_stats["mean_attn"],
                "residual_factor": args.residual_factor,
                "init_loss_prior_wm2": rad_stats["mean_loss"],
                "max_direct_loss_wm2": args.max_direct_loss_wm2,
                "final_blend_direct": args.final_blend_direct,
            },
            "train_radiation_stats": rad_stats,
            "metrics": row,
            "architecture": "CloudRadiationV8CleanDirect",
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val["loss"]["mae"] < best:
            best = val["loss"]["mae"]
            bad = 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"saved best.pt val_mae={best:.2f}W/m2")
        else:
            bad += 1
        if args.early_stop_patience > 0 and bad >= args.early_stop_patience:
            print(f"early stop: {bad} epochs without improvement")
            break

    if (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(strip_prefix(ckpt["model_state"]))
        test = evaluate(model, test_loader, device, amp_dtype)
        write_json(out_dir / "test_metrics.json", test)
        print("test:", json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
