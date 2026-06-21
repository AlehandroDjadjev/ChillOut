#!/usr/bin/env python3
from __future__ import annotations

import base64
import gc
import io
import math
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
import torch.nn.functional as F

import train_cloud_radiation_bottom_v8_CLEAN_DIRECT as v8
import train_cloud_template_selector_v13 as v13

MODULE_DIR = Path(__file__).resolve().parent


CHANNEL_NAMES_8 = [
    "mask",
    "probability",
    "probability_smooth",
    "cirrus",
    "high_cloud",
    "medium_cloud",
    "aot",
    "texture",
]


def strip_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def parse_channels(s: str, max_c: int) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if 0 <= value < max_c:
            out.append(value)
    return out


def resolve_local_path(path: Path) -> Path:
    """Resolve relative model/data paths from either cwd or this module folder."""
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = MODULE_DIR / path
    if candidate.exists():
        return candidate
    return candidate


def clamp_target_loss(target: torch.Tensor, clear: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    unclamped = target.clone()
    target = torch.minimum(target.clamp(min=0.0), clear.clamp_min(0.0))
    return target, bool((target != unclamped).any().item())


def c_delta_to_wm2_delta(delta_c: float, wm2_per_c: float) -> float:
    # More cloud-loss means less shortwave reaches the ground, so this proxy cools.
    return -float(delta_c) * float(wm2_per_c)


def wm2_delta_to_c_delta(delta_wm2: float, wm2_per_c: float) -> float:
    return -float(delta_wm2) / max(1e-6, float(wm2_per_c))


def make_clean_args(ckargs: Dict[str, Any], override_min_clear: Optional[float] = None):
    class Args:
        pass

    args = Args()
    args.min_clear_wm2 = float(override_min_clear if override_min_clear is not None else ckargs.get("min_clear_wm2", 120.0))
    args.clean_drop_invalid = bool(ckargs.get("clean_drop_invalid", True))
    args.clean_drop_negative = bool(ckargs.get("clean_drop_negative", True))
    args.clean_drop_high_cloud_low_loss = bool(ckargs.get("clean_drop_high_cloud_low_loss", True))
    args.clean_drop_low_cloud_high_loss = bool(ckargs.get("clean_drop_low_cloud_high_loss", False))
    args.high_cloud_thresh = float(ckargs.get("high_cloud_thresh", 0.65))
    args.low_cloud_thresh = float(ckargs.get("low_cloud_thresh", 0.05))
    args.low_loss_thresh_wm2 = float(ckargs.get("low_loss_thresh_wm2", 25.0))
    args.high_loss_thresh_wm2 = float(ckargs.get("high_loss_thresh_wm2", 280.0))
    args.min_loss_wm2 = float(ckargs.get("min_loss_wm2", 0.0))
    args.max_loss_wm2 = float(ckargs.get("max_loss_wm2", 900.0))
    return args


def get_cloud_raw_stats(x_norm, raw_names: List[str], cloud_names: List[str], device: torch.device):
    raw_to_idx = {name: i for i, name in enumerate(raw_names)}
    idx = [raw_to_idx[name] for name in cloud_names]
    mean = torch.tensor(x_norm.mean[idx], dtype=torch.float32, device=device).view(1, 1, -1)
    std = torch.tensor(x_norm.std[idx], dtype=torch.float32, device=device).view(1, 1, -1).clamp_min(1e-6)
    return mean, std


def white_dropout_input(image: torch.Tensor, channels: List[int], threshold: float, drop_prob: float):
    out = image.clone()
    dropped_mask = torch.zeros(image.size(0), image.size(1), 1, image.size(-2), image.size(-1), device=image.device)
    total_white = 0.0
    total_dropped = 0.0
    for ch in channels:
        white = out[:, :, ch:ch + 1] >= threshold
        drop = (torch.rand_like(out[:, :, ch:ch + 1]) < drop_prob) & white
        out[:, :, ch:ch + 1] = torch.where(drop, torch.zeros_like(out[:, :, ch:ch + 1]), out[:, :, ch:ch + 1])
        dropped_mask = torch.maximum(dropped_mask, drop.float())
        total_white += float(white.float().sum().detach().cpu())
        total_dropped += float(drop.float().sum().detach().cpu())
    return out.clamp(0.0, 1.0), dropped_mask, {
        "mode": "white_dropout",
        "actual_drop_rate_on_white": float(total_dropped / max(1.0, total_white)),
    }


def make_input(image: torch.Tensor, input_mode: str, train_args: Dict[str, Any]):
    if input_mode == "full":
        dropped = torch.zeros(image.size(0), image.size(1), 1, image.size(-2), image.size(-1), device=image.device)
        return image.clone(), dropped, {"mode": "full_original"}
    if input_mode != "dropout":
        raise ValueError(f"unknown input_mode {input_mode!r}")
    channels = parse_channels(str(train_args.get("white_drop_channels", "0,1,2,3,4,5,6,7")), image.size(2))
    threshold = float(train_args.get("white_threshold", 0.35))
    drop_prob = float(train_args.get("white_drop_prob", 0.55))
    return white_dropout_input(image, channels, threshold, drop_prob)


def reward_forward(
    reward_model,
    image: torch.Tensor,
    cloud_base: torch.Tensor,
    cloud_names: List[str],
    raw_mean: torch.Tensor,
    raw_std: torch.Tensor,
    context: torch.Tensor,
    clear: torch.Tensor,
):
    cloud = v13.derive_cloud_features_from_image(image, cloud_base, cloud_names, raw_mean, raw_std)
    return v13.reward_forward(reward_model, image, cloud, context, clear), cloud


def batch_coverage(image: torch.Tensor, channels: List[int]) -> float:
    return float(v13.batch_coverage(image, channels).mean().detach().cpu())


def tensor_to_cloud_photo(seq: torch.Tensor, frame: int = -1, size: int = 320) -> Image.Image:
    arr = seq.detach().cpu().float().numpy()
    if arr.ndim == 5:
        arr = arr[0]
    frame_arr = arr[frame]
    c, h, w = frame_arr.shape
    mask = frame_arr[0] if c > 0 else np.zeros((h, w), dtype=np.float32)
    prob = frame_arr[1] if c > 1 else mask
    high = frame_arr[4] if c > 4 else prob
    med = frame_arr[5] if c > 5 else prob
    tex = frame_arr[7] if c > 7 else prob

    cloud = np.clip(0.44 * mask + 0.30 * prob + 0.16 * high + 0.10 * med, 0.0, 1.0)
    detail = np.clip(tex - tex.mean(), -0.25, 0.25)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    sky_top = np.asarray([64, 128, 198], dtype=np.float32)
    sky_bottom = np.asarray([156, 201, 232], dtype=np.float32)
    sky = sky_top * (1.0 - yy)[..., None] + sky_bottom * yy[..., None]
    white = np.asarray([247, 250, 252], dtype=np.float32)
    shade = np.asarray([184, 197, 213], dtype=np.float32)
    cloud_rgb = white * (0.82 + 0.18 * np.clip(high, 0.0, 1.0))[..., None] + shade * (0.18 * np.clip(med, 0.0, 1.0))[..., None]
    rgb = sky * (1.0 - cloud[..., None]) + cloud_rgb * cloud[..., None]
    rgb = np.clip(rgb + detail[..., None] * 85.0, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((size, size), Image.Resampling.BILINEAR)


def tensor_to_cloud_mask(seq: torch.Tensor, frame: int = -1, size: int = 320) -> Image.Image:
    arr = seq.detach().cpu().float().numpy()
    if arr.ndim == 5:
        arr = arr[0]
    frame_arr = arr[frame]
    c, h, w = frame_arr.shape
    planes = []
    for ch in (0, 1, 4, 5):
        if ch < c:
            planes.append(frame_arr[ch])
    cloud = np.max(np.stack(planes, axis=0), axis=0) if planes else np.zeros((h, w), dtype=np.float32)
    cloud = np.clip(cloud, 0.0, 1.0)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (cloud * 235).astype(np.uint8)
    rgb[..., 1] = (np.sqrt(cloud) * 255).astype(np.uint8)
    rgb[..., 2] = (np.clip(cloud * 1.25, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.BILINEAR)


def tensor_added_delta(before: torch.Tensor, after: torch.Tensor, frame: int = -1, size: int = 320) -> Image.Image:
    a = before.detach().cpu().float()
    b = after.detach().cpu().float()
    if a.ndim == 5:
        a = a[0]
    if b.ndim == 5:
        b = b[0]
    aa = a[frame]
    bb = b[frame]
    channels = [ch for ch in (0, 1, 4, 5, 7) if ch < aa.size(0)]
    if channels:
        diff = (bb[channels] - aa[channels]).numpy()
        pos = np.clip(diff, 0.0, None).max(axis=0)
        neg = np.clip(-diff, 0.0, None).max(axis=0)
        base = np.clip(aa[channels].numpy(), 0.0, 1.0).max(axis=0)
    else:
        h, w = aa.shape[-2:]
        pos = np.zeros((h, w), dtype=np.float32)
        neg = np.zeros((h, w), dtype=np.float32)
        base = np.zeros((h, w), dtype=np.float32)

    scale = 0.35
    pos = np.clip(pos / scale, 0.0, 1.0)
    neg = np.clip(neg / scale, 0.0, 1.0)
    base = np.clip(base * 0.30, 0.0, 0.30)
    rgb = np.zeros((*base.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(base * 255 + pos * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(base * 255 + pos * 185 + neg * 65, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(base * 255 + neg * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.BILINEAR)


def tensor_to_strip(seq: torch.Tensor, size_each: int = 160) -> Image.Image:
    arr = seq.detach().cpu().float()
    if arr.ndim == 5:
        arr = arr[0]
    frames = arr.size(0)
    gap = 8
    label_h = 22
    out = Image.new("RGB", (frames * size_each + (frames - 1) * gap, size_each + label_h), (18, 24, 30))
    draw = ImageDraw.Draw(out)
    for i in range(frames):
        im = tensor_to_cloud_photo(arr[i:i + 1], frame=0, size=size_each)
        x = i * (size_each + gap)
        out.paste(im, (x, label_h))
        draw.text((x + 6, 4), f"t{i}", fill=(226, 232, 240))
    return out


def tensor_channel_grid(seq: torch.Tensor, channels: List[int], size_each: int = 96) -> Image.Image:
    arr = seq.detach().cpu().float()
    if arr.ndim == 5:
        arr = arr[0]
    last = arr[-1].numpy()
    channels = [ch for ch in channels if 0 <= ch < last.shape[0]]
    if not channels:
        channels = [0]
    label_h = 20
    gap = 8
    out = Image.new("RGB", (len(channels) * size_each + (len(channels) - 1) * gap, size_each + label_h), (18, 24, 30))
    draw = ImageDraw.Draw(out)
    for i, ch in enumerate(channels):
        x = i * (size_each + gap)
        plane = np.clip(last[ch], 0.0, 1.0)
        im = Image.fromarray((plane * 255.0).astype(np.uint8), mode="L").resize((size_each, size_each), Image.Resampling.BILINEAR).convert("RGB")
        out.paste(im, (x, label_h))
        label = CHANNEL_NAMES_8[ch] if ch < len(CHANNEL_NAMES_8) else f"ch{ch}"
        draw.text((x + 4, 4), label[:14], fill=(226, 232, 240))
    return out


def png_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def tensor_channel_data_url(seq: torch.Tensor, channel: int = 0, frame: int = -1, size: int = 320) -> str:
    arr = seq.detach().cpu().float()
    if arr.ndim == 5:
        arr = arr[0]
    plane = arr[frame, max(0, min(channel, arr.size(1) - 1))].numpy()
    im = Image.fromarray((np.clip(plane, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    return png_data_url(im.convert("RGB"))


def data_url_to_live_cloud_tensor(data_url: str, height: int, width: int, lookback: int) -> torch.Tensor:
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw = base64.b64decode(b64)
    base_img = Image.open(io.BytesIO(raw)).convert("L").resize((width, height), Image.Resampling.BILINEAR)
    prob = np.asarray(base_img, dtype=np.float32) / 255.0
    smooth_img = base_img.filter(ImageFilter.GaussianBlur(radius=max(1.0, min(height, width) / 80.0)))
    smooth = np.asarray(smooth_img, dtype=np.float32) / 255.0
    mask = (prob >= 0.35).astype(np.float32)
    gy, gx = np.gradient(smooth)
    texture = np.clip(np.sqrt(gx * gx + gy * gy) * 6.0, 0.0, 1.0).astype(np.float32)
    high = np.clip(prob * (0.60 + 0.25 * texture), 0.0, 1.0)
    medium = np.clip(smooth * 0.70, 0.0, 1.0)
    cirrus = np.clip((smooth - mask * 0.15) * 0.25, 0.0, 1.0)
    aot = np.clip(0.08 + prob * 0.22, 0.0, 1.0)
    frame = np.stack([mask, prob, smooth, cirrus, high, medium, aot, texture], axis=0).astype(np.float32)
    seq = np.stack([frame for _ in range(max(1, int(lookback)))], axis=0)
    return torch.from_numpy(seq)


def record_temperature_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    current = record.get("current_temperature_c")
    target = record.get("target_temperature_c", record.get("target"))
    try:
        current_f = None if current is None else float(current)
    except Exception:
        current_f = None
    try:
        target_f = None if target is None else float(target)
    except Exception:
        target_f = None
    return {
        "current_temperature_c": current_f,
        "dataset_future_temperature_c": target_f,
        "dataset_future_delta_c": None if current_f is None or target_f is None else target_f - current_f,
    }


@dataclass
class InferenceConfig:
    data_root: Path = Path("dataset_cloudforce_radiation_v6_big_clean")
    selector_checkpoint: Path = Path("cloud_template_selector_v13/best_selector.pt")
    reward_checkpoint: Path = Path("runs/cloud_radiation_v8_clean_direct1/best.pt")
    split: str = "val"
    force_cpu: bool = False
    min_clear_wm2: Optional[float] = None
    idle_offload: bool = True
    max_oracle_batch_size: int = 8
    load_dataset: bool = True


class CloudTemplateInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.compute_device = torch.device("cpu" if config.force_cpu or not torch.cuda.is_available() else "cuda")
        self.device = torch.device("cpu")
        self._request_lock = threading.Lock()
        if self.compute_device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self.config.data_root = resolve_local_path(self.config.data_root)
        self.config.selector_checkpoint = resolve_local_path(self.config.selector_checkpoint)
        self.config.reward_checkpoint = resolve_local_path(self.config.reward_checkpoint)

        self.selector_ckpt = torch.load(self.config.selector_checkpoint, map_location="cpu")
        self.reward_ckpt = torch.load(self.config.reward_checkpoint, map_location="cpu")
        self.train_args = dict(self.selector_ckpt.get("args", {}))
        self.raw_names = list(self.reward_ckpt["raw_feature_names"])
        self.cloud_names = list(self.reward_ckpt["cloud_feature_names"])
        self.x_norm = v8.base.Normalizer.from_state_dict(self.reward_ckpt["normalizer"])
        self.cleaner = make_clean_args(self.reward_ckpt.get("args", {}), override_min_clear=config.min_clear_wm2)

        self.records_by_split: Dict[str, List[Dict[str, Any]]] = {}
        self.datasets: Dict[str, Any] = {}
        if config.load_dataset:
            for split in ("train", "val", "test"):
                records = v8.clean_radiation_records(v8.base.load_records(config.data_root, split), self.cleaner, split)
                self.records_by_split[split] = records
                self.datasets[split] = v8.RadiationSequenceDataset(
                    root=config.data_root,
                    records=records,
                    raw_names=self.raw_names,
                    cloud_names=self.cloud_names,
                    x_norm=self.x_norm,
                    image_height=int(self.reward_ckpt.get("image_height", 160)),
                    image_width=int(self.reward_ckpt.get("image_width", 160)),
                    lookback=int(self.reward_ckpt.get("lookback", 4)),
                    max_gap_days=float(self.reward_ckpt.get("args", {}).get("max_gap_days", 12.0)),
                    use_cloud_tensor=bool(self.reward_ckpt.get("args", {}).get("use_cloud_tensor", True)),
                    augment=False,
                    min_clear_wm2=float(self.cleaner.min_clear_wm2),
                )

        self.codebook = self.selector_ckpt["codebook"].float()
        self.codebook_meta = list(self.selector_ckpt.get("codebook_meta", []))
        self.mode_names = list(self.selector_ckpt.get("mode_names", v13.MODE_NAMES))
        self.cov_channels = parse_channels(str(self.train_args.get("coverage_channels", "0,1,2")), int(self.selector_ckpt.get("in_channels", 8)))
        self.edit_channels = parse_channels(str(self.train_args.get("edit_channels", "0,1,2,3,4,5,6,7")), int(self.selector_ckpt.get("in_channels", 8)))
        self.view_channels = [0, 1, 4, 5, 7]

        self.selector = v13.CloudTemplateSelectorV13(
            frames=int(self.selector_ckpt.get("frames", 4)),
            in_channels=int(self.selector_ckpt.get("in_channels", 8)),
            codebook_size=self.codebook.size(0),
            num_modes=len(self.mode_names),
            context_dim=8,
            base=int(self.train_args.get("base_channels", 32)),
            hidden=int(self.train_args.get("hidden_dim", 192)),
            loss_scale=float(self.selector_ckpt.get("loss_scale", 300.0)),
        )
        self.selector.load_state_dict(strip_prefix(self.selector_ckpt["model_state"]))
        self.selector.eval()
        for p in self.selector.parameters():
            p.requires_grad_(False)

        self.reward_model = v8.CloudRadiationV8CleanDirect(**self.reward_ckpt["model_kwargs"])
        self.reward_model.load_state_dict(strip_prefix(self.reward_ckpt["model_state"]))
        self.reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad_(False)

        self.raw_mean, self.raw_std = get_cloud_raw_stats(self.x_norm, self.raw_names, self.cloud_names, self.device)
        if self.compute_device.type == "cuda" and not self.config.idle_offload:
            self._move_runtime_to(self.compute_device)

    def _move_runtime_to(self, device: torch.device) -> None:
        if self.device == device:
            return
        self.codebook = self.codebook.to(device).float()
        self.selector.to(device)
        self.reward_model.to(device)
        self.raw_mean = self.raw_mean.to(device)
        self.raw_std = self.raw_std.to(device)
        self.device = device

    def _release_idle_cuda(self) -> None:
        if self.compute_device.type != "cuda" or not self.config.idle_offload:
            return
        self._move_runtime_to(torch.device("cpu"))
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def split_count(self, split: str) -> int:
        return len(self.datasets[split])

    def meta(self) -> Dict[str, Any]:
        return {
            "device": str(self.compute_device) + (" (offloaded while idle)" if self.config.idle_offload and self.compute_device.type == "cuda" else ""),
            "resident_device": str(self.device),
            "idle_offload": bool(self.config.idle_offload and self.compute_device.type == "cuda"),
            "max_oracle_batch_size": int(self.config.max_oracle_batch_size),
            "splits": {split: len(ds) for split, ds in self.datasets.items()},
            "codebook_size": int(self.codebook.size(0)),
            "modes": self.mode_names,
            "default_split": self.config.split,
            "conversion": {
                "meaning": "Temperature is a local proxy: positive C means warming request, negative C means cooling request.",
                "formula": "target_loss_wm2 = actual_loss_wm2 - requested_delta_c * wm2_per_c",
            },
        }

    def _sample_record(self, split: str, index: int) -> Dict[str, Any]:
        ds = self.datasets[split]
        window = ds.windows[index]
        return window.records[-1]

    @torch.inference_mode()
    def generate(
        self,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        with self._request_lock:
            self._move_runtime_to(self.compute_device)
            try:
                return self._generate_impl(**kwargs)
            finally:
                self._release_idle_cuda()

    def _generate_impl(
        self,
        split: str,
        sample_index: int,
        target_loss_wm2: Optional[float] = None,
        target_delta_c: float = -2.0,
        wm2_per_c: float = 80.0,
        input_mode: str = "full",
        run_oracle: bool = True,
        oracle_batch_size: int = 32,
        penalty_l1: float = 25.0,
        penalty_coverage: float = 80.0,
        max_coverage: Optional[float] = None,
        seed: int = 123,
    ) -> Dict[str, Any]:
        if split not in self.datasets:
            raise ValueError(f"unknown split {split!r}")
        ds = self.datasets[split]
        if sample_index < 0 or sample_index >= len(ds):
            raise IndexError(f"sample_index {sample_index} outside split length {len(ds)}")
        oracle_batch_size = max(1, min(int(oracle_batch_size), int(self.config.max_oracle_batch_size)))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        item = ds[sample_index]
        image = item["image"].unsqueeze(0).to(self.device).float()
        cloud = item["cloud_features"].unsqueeze(0).to(self.device).float()
        context = item["context_features"].unsqueeze(0).to(self.device).float()
        clear = item["clear_wm2"].view(1, 1).to(self.device).float()
        actual = item["target_loss_wm2"].view(1, 1).to(self.device).float()

        requested_delta_wm2 = None
        if target_loss_wm2 is None:
            requested_delta_wm2 = c_delta_to_wm2_delta(target_delta_c, wm2_per_c)
            target = actual + float(requested_delta_wm2)
            target_mode = "temperature_delta_proxy"
        else:
            target = torch.tensor([[float(target_loss_wm2)]], dtype=torch.float32, device=self.device)
            target_mode = "absolute_loss_wm2"
        target, target_was_clamped = clamp_target_loss(target, clear)

        input_img, _dropped, input_info = make_input(image, input_mode, self.train_args)
        base_pred, _ = reward_forward(self.reward_model, image, cloud, self.cloud_names, self.raw_mean, self.raw_std, context, clear)
        input_pred, cloud_input = reward_forward(self.reward_model, input_img, cloud, self.cloud_names, self.raw_mean, self.raw_std, context, clear)

        inp_cov = v13.batch_coverage(input_img, self.cov_channels)
        template_logits, mode_logits = self.selector(input_img, context, clear, actual, target, input_pred["loss_wm2"], inp_cov)
        tau = max(1e-6, float(self.train_args.get("eval_tau", 0.20)))
        template_probs = F.softmax(template_logits / tau, dim=-1)
        mode_probs = F.softmax(mode_logits / tau, dim=-1)
        template_idx = int(template_probs.argmax(dim=-1).item())
        mode_idx = int(mode_probs.argmax(dim=-1).item())

        selector_template = self.codebook[template_idx:template_idx + 1]
        selector_mode = torch.zeros(1, len(self.mode_names), device=self.device)
        selector_mode[:, mode_idx] = 1.0
        selector_output = v13.apply_modes(input_img, selector_template, selector_mode, self.edit_channels)
        selector_pred, _ = reward_forward(self.reward_model, selector_output, cloud_input, self.cloud_names, self.raw_mean, self.raw_std, context, clear)

        chosen_output = selector_output
        chosen_template = selector_template
        chosen_row = self._result_row(
            "selector",
            template_idx,
            mode_idx,
            selector_pred["loss_wm2"],
            target,
            input_img,
            selector_output,
            selector_template,
        )
        oracle_row: Optional[Dict[str, Any]] = None
        if run_oracle:
            oracle_row, oracle_output, oracle_template = self._oracle_search(
                input_img=input_img,
                cloud=cloud_input,
                context=context,
                clear=clear,
                target=target,
                oracle_batch_size=max(1, int(oracle_batch_size)),
                penalty_l1=float(penalty_l1),
                penalty_coverage=float(penalty_coverage),
                max_coverage=max_coverage,
            )
            chosen_output = oracle_output
            chosen_template = oracle_template
            chosen_row = oracle_row

        final_pred, _ = reward_forward(self.reward_model, chosen_output, cloud_input, self.cloud_names, self.raw_mean, self.raw_std, context, clear)
        final_loss = float(final_pred["loss_wm2"].item())
        actual_loss = float(actual.item())
        target_loss = float(target.item())
        clear_loss = float(clear.item())
        verified_delta_c = wm2_delta_to_c_delta(final_loss - actual_loss, wm2_per_c)
        target_delta_c_proxy = wm2_delta_to_c_delta(target_loss - actual_loss, wm2_per_c)
        base_delta_c_proxy = wm2_delta_to_c_delta(float(base_pred["loss_wm2"].item()) - actual_loss, wm2_per_c)
        input_delta_c_proxy = wm2_delta_to_c_delta(float(input_pred["loss_wm2"].item()) - actual_loss, wm2_per_c)

        sample_record = self._sample_record(split, sample_index)
        top_templates = []
        probs = template_probs[0].detach().cpu().numpy()
        for ti in np.argsort(-probs)[:8]:
            meta = self.codebook_meta[int(ti)] if int(ti) < len(self.codebook_meta) else {}
            top_templates.append({
                "template_index": int(ti),
                "prob": float(probs[ti]),
                "sample_id": str(meta.get("sample_id", "")),
                "location": str(meta.get("location", "")),
                "coverage": self._finite_float(meta.get("coverage")),
            })

        mode_rows = []
        mode_np = mode_probs[0].detach().cpu().numpy()
        for mi, prob in enumerate(mode_np):
            mode_rows.append({"mode": self.mode_names[mi], "prob": float(prob)})
        mode_rows.sort(key=lambda row: -row["prob"])

        return {
            "sample": {
                "split": split,
                "sample_index": int(sample_index),
                "sample_id": str(item["sample_id"]),
                "location": str(item["location"]),
                "anchor": str(item["anchor"]),
                **record_temperature_meta(sample_record),
            },
            "target": {
                "mode": target_mode,
                "requested_delta_c": None if target_loss_wm2 is not None else float(target_delta_c),
                "target_delta_c_proxy": target_delta_c_proxy,
                "wm2_per_c": float(wm2_per_c),
                "requested_delta_wm2": requested_delta_wm2,
                "target_loss_wm2": target_loss,
                "target_was_clamped": target_was_clamped,
            },
            "verification": {
                "actual_dataset_loss_wm2": actual_loss,
                "clear_sky_wm2": clear_loss,
                "bottom_original_loss_wm2": float(base_pred["loss_wm2"].item()),
                "bottom_input_loss_wm2": float(input_pred["loss_wm2"].item()),
                "bottom_generated_loss_wm2": final_loss,
                "target_abs_error_wm2": abs(final_loss - target_loss),
                "requested_temperature_change_c": None if target_loss_wm2 is not None else float(target_delta_c),
                "verified_temperature_change_c": verified_delta_c,
                "target_temperature_change_proxy_c": target_delta_c_proxy,
                "original_temperature_change_proxy_c": base_delta_c_proxy,
                "input_temperature_change_proxy_c": input_delta_c_proxy,
                "temperature_abs_error_c": abs(verified_delta_c - target_delta_c_proxy),
                "generated_attenuation": float(final_pred["attenuation"].item()),
                "coverage": batch_coverage(chosen_output, self.cov_channels),
            },
            "selector": {
                "template_index": template_idx,
                "mode": self.mode_names[mode_idx],
                "pred_loss_wm2": float(selector_pred["loss_wm2"].item()),
                "abs_error_wm2": abs(float(selector_pred["loss_wm2"].item()) - target_loss),
                "coverage": batch_coverage(selector_output, self.cov_channels),
                "top_templates": top_templates,
                "mode_probs": mode_rows,
            },
            "chosen": chosen_row,
            "oracle": oracle_row,
            "input": input_info,
            "images": {
                "original": png_data_url(tensor_to_cloud_photo(image)),
                "model_input": png_data_url(tensor_to_cloud_photo(input_img)),
                "selected_template": png_data_url(tensor_to_cloud_photo(chosen_template)),
                "generated": png_data_url(tensor_to_cloud_photo(chosen_output)),
                "original_mask": png_data_url(tensor_to_cloud_mask(image)),
                "template_mask": png_data_url(tensor_to_cloud_mask(chosen_template)),
                "generated_mask": png_data_url(tensor_to_cloud_mask(chosen_output)),
                "added_delta": png_data_url(tensor_added_delta(input_img, chosen_output)),
                "original_strip": png_data_url(tensor_to_strip(image)),
                "generated_strip": png_data_url(tensor_to_strip(chosen_output)),
                "channels": png_data_url(tensor_channel_grid(chosen_output, self.view_channels)),
            },
        }

    @torch.no_grad()
    def generate_live(
        self,
        sample: Dict[str, Any],
        mask_data_url: str,
        target_temperature_c: float,
        target_loss_wm2: Optional[float] = None,
        wm2_per_c: float = 80.0,
        input_mode: str = "full",
        run_oracle: bool = True,
        oracle_batch_size: int = 32,
        penalty_l1: float = 25.0,
        penalty_coverage: float = 80.0,
        max_coverage: Optional[float] = None,
        seed: int = 123,
    ) -> Dict[str, Any]:
        """Run the model-test-app selector on a live Sentinel-2 mask + same-date weather sample.

        The trained selector expects a lookback of 8-channel Sentinel tensors and v6/v8 radiation
        context. A live page only has one mask image, so we repeat that mask across the lookback and
        synthesize the missing S2 channels from the mask texture. Weather/radiation fields come from
        build_sample.mjs for the same selected date.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        lookback = int(self.reward_ckpt.get("lookback", 4))
        image_height = int(self.reward_ckpt.get("image_height", 160))
        image_width = int(self.reward_ckpt.get("image_width", 160))
        image = data_url_to_live_cloud_tensor(mask_data_url, image_height, image_width, lookback).unsqueeze(0).to(self.device).float()

        live_record = self._live_record_from_sample(sample, target_temperature_c)
        raw = self._live_raw_vector(live_record)
        raw_norm = self.x_norm.transform(raw)
        raw_to_idx = {name: i for i, name in enumerate(self.raw_names)}
        cloud_idx = [raw_to_idx[name] for name in self.cloud_names if name in raw_to_idx]
        cloud_vec = raw_norm[cloud_idx].astype(np.float32)
        cloud = torch.from_numpy(np.stack([cloud_vec for _ in range(lookback)], axis=0)).unsqueeze(0).to(self.device).float()
        context = torch.from_numpy(v8.record_context_features(live_record)).unsqueeze(0).to(self.device).float()
        rb = v8.base.radiation_bundle(live_record)
        clear = torch.tensor([[float(rb["clear_wm2"])]], dtype=torch.float32, device=self.device)
        actual = torch.tensor([[float(rb["loss_wm2"])]], dtype=torch.float32, device=self.device)

        current_temp = live_record.get("current_temperature_c")
        try:
            current_temp_f = float(current_temp)
        except Exception:
            current_temp_f = None
        target_temp_f = float(target_temperature_c)
        target_delta_c = 0.0 if current_temp_f is None else target_temp_f - current_temp_f

        requested_delta_wm2 = None
        if target_loss_wm2 is None:
            requested_delta_wm2 = c_delta_to_wm2_delta(target_delta_c, wm2_per_c)
            target = actual + float(requested_delta_wm2)
            target_mode = "absolute_temperature_proxy"
        else:
            target = torch.tensor([[float(target_loss_wm2)]], dtype=torch.float32, device=self.device)
            target_mode = "absolute_loss_wm2"
        target, target_was_clamped = clamp_target_loss(target, clear)

        input_img, _dropped, input_info = make_input(image, input_mode, self.train_args)
        base_pred, _ = reward_forward(self.reward_model, image, cloud, self.cloud_names, self.raw_mean, self.raw_std, context, clear)
        input_pred, cloud_input = reward_forward(self.reward_model, input_img, cloud, self.cloud_names, self.raw_mean, self.raw_std, context, clear)

        inp_cov = v13.batch_coverage(input_img, self.cov_channels)
        template_logits, mode_logits = self.selector(input_img, context, clear, actual, target, input_pred["loss_wm2"], inp_cov)
        tau = max(1e-6, float(self.train_args.get("eval_tau", 0.20)))
        template_probs = F.softmax(template_logits / tau, dim=-1)
        mode_probs = F.softmax(mode_logits / tau, dim=-1)
        template_idx = int(template_probs.argmax(dim=-1).item())
        mode_idx = int(mode_probs.argmax(dim=-1).item())

        selector_template = self.codebook[template_idx:template_idx + 1]
        selector_mode = torch.zeros(1, len(self.mode_names), device=self.device)
        selector_mode[:, mode_idx] = 1.0
        selector_output = v13.apply_modes(input_img, selector_template, selector_mode, self.edit_channels)
        selector_pred, _ = reward_forward(self.reward_model, selector_output, cloud_input, self.cloud_names, self.raw_mean, self.raw_std, context, clear)

        chosen_output = selector_output
        chosen_template = selector_template
        chosen_row = self._result_row(
            "selector",
            template_idx,
            mode_idx,
            selector_pred["loss_wm2"],
            target,
            input_img,
            selector_output,
            selector_template,
        )
        oracle_row: Optional[Dict[str, Any]] = None
        if run_oracle:
            oracle_row, oracle_output, oracle_template = self._oracle_search(
                input_img=input_img,
                cloud=cloud_input,
                context=context,
                clear=clear,
                target=target,
                oracle_batch_size=max(1, int(oracle_batch_size)),
                penalty_l1=float(penalty_l1),
                penalty_coverage=float(penalty_coverage),
                max_coverage=max_coverage,
            )
            chosen_output = oracle_output
            chosen_template = oracle_template
            chosen_row = oracle_row

        final_pred, _ = reward_forward(self.reward_model, chosen_output, cloud_input, self.cloud_names, self.raw_mean, self.raw_std, context, clear)
        final_loss = float(final_pred["loss_wm2"].item())
        actual_loss = float(actual.item())
        target_loss = float(target.item())
        clear_loss = float(clear.item())
        verified_delta_c = wm2_delta_to_c_delta(final_loss - actual_loss, wm2_per_c)
        target_delta_c_proxy = wm2_delta_to_c_delta(target_loss - actual_loss, wm2_per_c)
        base_delta_c_proxy = wm2_delta_to_c_delta(float(base_pred["loss_wm2"].item()) - actual_loss, wm2_per_c)
        input_delta_c_proxy = wm2_delta_to_c_delta(float(input_pred["loss_wm2"].item()) - actual_loss, wm2_per_c)

        top_templates = []
        probs = template_probs[0].detach().cpu().numpy()
        for ti in np.argsort(-probs)[:8]:
            meta = self.codebook_meta[int(ti)] if int(ti) < len(self.codebook_meta) else {}
            top_templates.append({
                "template_index": int(ti),
                "prob": float(probs[ti]),
                "sample_id": str(meta.get("sample_id", "")),
                "location": str(meta.get("location", "")),
                "coverage": self._finite_float(meta.get("coverage")),
            })

        mode_rows = []
        mode_np = mode_probs[0].detach().cpu().numpy()
        for mi, prob in enumerate(mode_np):
            mode_rows.append({"mode": self.mode_names[mi], "prob": float(prob)})
        mode_rows.sort(key=lambda row: -row["prob"])

        return {
            "sample": {
                "split": "live",
                "sample_index": None,
                "sample_id": str(sample.get("sample_id", "live_sample")),
                "location": str(sample.get("city", sample.get("place", ""))),
                "anchor": str(live_record.get("anchor", "")),
                "current_temperature_c": current_temp_f,
                "requested_target_temperature_c": target_temp_f,
                "requested_delta_c": None if current_temp_f is None else target_temp_f - current_temp_f,
            },
            "target": {
                "mode": target_mode,
                "requested_delta_c": None if target_loss_wm2 is not None else float(target_delta_c),
                "target_delta_c_proxy": target_delta_c_proxy,
                "wm2_per_c": float(wm2_per_c),
                "requested_delta_wm2": requested_delta_wm2,
                "target_loss_wm2": target_loss,
                "target_was_clamped": target_was_clamped,
            },
            "verification": {
                "actual_dataset_loss_wm2": actual_loss,
                "clear_sky_wm2": clear_loss,
                "bottom_original_loss_wm2": float(base_pred["loss_wm2"].item()),
                "bottom_input_loss_wm2": float(input_pred["loss_wm2"].item()),
                "bottom_generated_loss_wm2": final_loss,
                "target_abs_error_wm2": abs(final_loss - target_loss),
                "requested_temperature_change_c": None if target_loss_wm2 is not None else float(target_delta_c),
                "verified_temperature_change_c": verified_delta_c,
                "target_temperature_change_proxy_c": target_delta_c_proxy,
                "original_temperature_change_proxy_c": base_delta_c_proxy,
                "input_temperature_change_proxy_c": input_delta_c_proxy,
                "temperature_abs_error_c": abs(verified_delta_c - target_delta_c_proxy),
                "generated_attenuation": float(final_pred["attenuation"].item()),
                "coverage": batch_coverage(chosen_output, self.cov_channels),
            },
            "selector": {
                "template_index": template_idx,
                "mode": self.mode_names[mode_idx],
                "pred_loss_wm2": float(selector_pred["loss_wm2"].item()),
                "abs_error_wm2": abs(float(selector_pred["loss_wm2"].item()) - target_loss),
                "coverage": batch_coverage(selector_output, self.cov_channels),
                "top_templates": top_templates,
                "mode_probs": mode_rows,
            },
            "chosen": chosen_row,
            "oracle": oracle_row,
            "input": {**input_info, "live_sentinel_mask_repeated_as_lookback": True},
            "images": {
                "original": png_data_url(tensor_to_cloud_photo(image)),
                "model_input": png_data_url(tensor_to_cloud_photo(input_img)),
                "selected_template": png_data_url(tensor_to_cloud_photo(chosen_template)),
                "generated": png_data_url(tensor_to_cloud_photo(chosen_output)),
                "original_mask": tensor_channel_data_url(image, channel=0),
                "generated_mask": tensor_channel_data_url(chosen_output, channel=0),
                "original_strip": png_data_url(tensor_to_strip(image)),
                "generated_strip": png_data_url(tensor_to_strip(chosen_output)),
                "channels": png_data_url(tensor_channel_grid(chosen_output, self.view_channels)),
            },
        }

    def _live_record_from_sample(self, sample: Dict[str, Any], target_temperature_c: float) -> Dict[str, Any]:
        date = str(sample.get("date") or "")[:10]
        anchor = sample.get("anchor")
        if date:
            anchor = f"{date}T12:00:00Z"
        inputs = dict(sample.get("cloudforce_world_inputs") or {})
        inputs.update({k: v for k, v in (sample.get("model_inputs") or {}).items() if k not in inputs})
        current_temp = sample.get("current_temperature_c", sample.get("observed_temperature_c"))
        return {
            "sample_id": sample.get("sample_id", "live_sample"),
            "location": sample.get("city", sample.get("place", "live")),
            "city": sample.get("city", sample.get("place", "live")),
            "date": date,
            "anchor": anchor,
            "lat": sample.get("lat", 0.0),
            "lon": sample.get("lon", 0.0),
            "current_temperature_c": current_temp,
            "target_temperature_c": target_temperature_c,
            "inputs": inputs,
        }

    def _live_raw_vector(self, record: Dict[str, Any]) -> np.ndarray:
        values = np.asarray(self.x_norm.mean, dtype=np.float32).copy()
        inputs = record.get("inputs") or {}
        for i, name in enumerate(self.raw_names):
            raw = inputs.get(name)
            try:
                value = float(raw)
            except Exception:
                continue
            if math.isfinite(value):
                values[i] = value
        return values

    def _finite_float(self, value: Any) -> Optional[float]:
        try:
            x = float(value)
        except Exception:
            return None
        return x if math.isfinite(x) else None

    def _result_row(
        self,
        source: str,
        template_idx: int,
        mode_idx: int,
        pred_loss: torch.Tensor,
        target: torch.Tensor,
        input_img: torch.Tensor,
        output: torch.Tensor,
        template: torch.Tensor,
    ) -> Dict[str, Any]:
        meta = self.codebook_meta[template_idx] if template_idx < len(self.codebook_meta) else {}
        return {
            "source": source,
            "template_index": int(template_idx),
            "mode": self.mode_names[mode_idx],
            "template_sample_id": str(meta.get("sample_id", "")),
            "template_location": str(meta.get("location", "")),
            "template_anchor": str(meta.get("anchor", "")),
            "template_coverage": self._finite_float(meta.get("coverage")),
            "pred_loss_wm2": float(pred_loss.item()),
            "target_loss_wm2": float(target.item()),
            "abs_error_wm2": abs(float(pred_loss.item()) - float(target.item())),
            "image_l1_vs_input": float((output - input_img).abs().mean().item()),
            "template_l1_vs_input": float((template - input_img).abs().mean().item()),
            "coverage": batch_coverage(output, self.cov_channels),
        }

    def _oracle_search(
        self,
        input_img: torch.Tensor,
        cloud: torch.Tensor,
        context: torch.Tensor,
        clear: torch.Tensor,
        target: torch.Tensor,
        oracle_batch_size: int,
        penalty_l1: float,
        penalty_coverage: float,
        max_coverage: Optional[float],
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        best_score = float("inf")
        best_row: Optional[Dict[str, Any]] = None
        best_output: Optional[torch.Tensor] = None
        best_template: Optional[torch.Tensor] = None
        max_cov = float(max_coverage if max_coverage is not None else self.train_args.get("max_coverage", 0.70))

        for start in range(0, self.codebook.size(0), oracle_batch_size):
            end = min(self.codebook.size(0), start + oracle_batch_size)
            templates = self.codebook[start:end]
            k = templates.size(0)
            inp_b = input_img.expand(k, -1, -1, -1, -1).contiguous()
            cloud_b = cloud.expand(k, -1, -1).contiguous()
            ctx_b = context.expand(k, -1).contiguous()
            clear_b = clear.expand(k, -1).contiguous()
            target_b = target.expand(k, -1).contiguous()

            for mode_idx in range(len(self.mode_names)):
                mode_w = torch.zeros(k, len(self.mode_names), device=self.device)
                mode_w[:, mode_idx] = 1.0
                out = v13.apply_modes(inp_b, templates, mode_w, self.edit_channels)
                pred, _ = reward_forward(self.reward_model, out, cloud_b, self.cloud_names, self.raw_mean, self.raw_std, ctx_b, clear_b)
                loss = pred["loss_wm2"]
                err = (loss - target_b).abs().view(-1)
                l1 = (out - inp_b).abs().mean(dim=(1, 2, 3, 4))
                cov = v13.batch_coverage(out, self.cov_channels).view(-1)
                score = err + penalty_l1 * l1 + penalty_coverage * torch.relu(cov - max_cov)
                candidate_idx = int(score.argmin().item())
                candidate_score = float(score[candidate_idx].item())
                if candidate_score < best_score:
                    template_idx = start + candidate_idx
                    best_score = candidate_score
                    best_output = out[candidate_idx:candidate_idx + 1].detach().clone()
                    best_template = templates[candidate_idx:candidate_idx + 1].detach().clone()
                    best_row = self._result_row(
                        "oracle",
                        template_idx,
                        mode_idx,
                        loss[candidate_idx],
                        target,
                        input_img,
                        best_output,
                        best_template,
                    )
                    best_row["score"] = candidate_score

        if best_row is None or best_output is None or best_template is None:
            raise RuntimeError("oracle search produced no candidates")
        return best_row, best_output, best_template
