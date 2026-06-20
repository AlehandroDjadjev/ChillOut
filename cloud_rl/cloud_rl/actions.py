from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

# Operation IDs. The policy can emit fewer than max_actions by choosing NOOP.
NOOP = 0
REMOVE = 1
MODIFY = 2
CREATE = 3


@dataclass(frozen=True)
class PropertyRanges:
    cloud_probability: Tuple[float, float] = (0.0, 1.0)
    aot_proxy: Tuple[float, float] = (0.0, 2.5)
    cloud_layer: Tuple[float, float] = (0.0, 1.0)
    texture_proxy: Tuple[float, float] = (0.0, 1.0)
    radius_px: Tuple[float, float] = (6.0, 96.0)


def _scale01(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + x.clamp(0, 1) * (hi - lo)


@lru_cache(maxsize=32)
def _cached_grid(height: int, width: int, device_type: str, device_index: int | None) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    return yy.float()[None, None, :, :], xx.float()[None, None, :, :]


def decode_params(params_tanh: torch.Tensor, ranges: PropertyRanges = PropertyRanges()) -> Dict[str, torch.Tensor]:
    """Decode tanh-bounded params [-1,1] into physical-ish values.

    params dims:
      x, y, radius, cloud_probability, aot_proxy, cirrus_proxy, cloud_layer, texture_proxy

    x/y remain [0,1] normalized image coordinates. cloud_layer is a continuous
    low/medium/high proxy and cirrus_proxy is a continuous cloud subtype proxy.
    """
    u = (params_tanh + 1.0) * 0.5
    decoded = {
        "x01": u[..., 0],
        "y01": u[..., 1],
        "radius_px": _scale01(u[..., 2], *ranges.radius_px),
        "cloud_probability": _scale01(u[..., 3], *ranges.cloud_probability),
        "aot_proxy": _scale01(u[..., 4], *ranges.aot_proxy),
        "cirrus_proxy": u[..., 5],
        "cloud_layer": _scale01(u[..., 6], *ranges.cloud_layer),
        "texture_proxy": _scale01(u[..., 7], *ranges.texture_proxy),
    }
    # Backward-compatible aliases used by older diagnostics/output code.
    decoded["mass_proxy"] = decoded["cloud_probability"]
    decoded["humidity_pct"] = decoded["cirrus_proxy"] * 100.0
    decoded["optical_thickness"] = decoded["aot_proxy"] * 40.0
    decoded["cloud_top_height_m"] = decoded["cloud_layer"] * 12000.0
    decoded["cloud_type"] = decoded["cirrus_proxy"]
    return decoded


def gaussian_blobs(x01: torch.Tensor, y01: torch.Tensor, radius: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Return [B,K,H,W] soft blobs."""
    device = x01.device
    yy, xx = _cached_grid(height, width, device.type, device.index)
    x = x01[:, :, None, None] * (width - 1)
    y = y01[:, :, None, None] * (height - 1)
    r = radius[:, :, None, None].clamp_min(1e-3)
    return torch.exp(-((xx - x).pow(2) + (yy - y).pow(2)) / (2.0 * r.pow(2)))


def rasterize_actions(
    original_mask: torch.Tensor,
    op: torch.Tensor,
    params_tanh: torch.Tensor,
    ranges: PropertyRanges = PropertyRanges(),
) -> Dict[str, torch.Tensor]:
    """Convert up to K action tokens into generated mask and property maps.

    Args:
        original_mask: [B,1,H,W] binary cloud mask.
        op: [B,K] long in {NOOP, REMOVE, MODIFY, CREATE}.
        params_tanh: [B,K,8] continuous params in [-1,1].

    Returns dict with generated_mask, property_maps, action_maps, decoded params.
    property_maps channels: cloud_probability, aot_norm, layer_norm, texture_norm, cirrus_proxy.
    action_maps channels: noop, remove, modify, create, cloud_probability, aot_norm, layer_norm, texture_norm.
    """
    b, _, h, w = original_mask.shape
    decoded = decode_params(params_tanh, ranges)
    blobs = gaussian_blobs(decoded["x01"], decoded["y01"], decoded["radius_px"], h, w)

    noop_w = (op == NOOP).float()
    remove_w = (op == REMOVE).float()
    modify_w = (op == MODIFY).float()
    create_w = (op == CREATE).float()
    property_w = (modify_w + create_w).clamp(0, 1)

    remove_map = (blobs * remove_w[:, :, None, None]).amax(dim=1, keepdim=True)
    create_map = (blobs * create_w[:, :, None, None]).amax(dim=1, keepdim=True)
    modify_map = (blobs * modify_w[:, :, None, None]).amax(dim=1, keepdim=True)

    def weighted_prop(name: str, norm_divisor: float = 1.0) -> torch.Tensor:
        val = decoded[name] / norm_divisor
        weights = blobs * property_w[:, :, None, None]
        num = (weights * val[:, :, None, None]).sum(dim=1, keepdim=True)
        den = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.where(den > 1e-5, num / den, torch.zeros_like(num))

    probability_map = weighted_prop("cloud_probability", 1.0).clamp(0, 1)
    aot_norm = (weighted_prop("aot_proxy", 2.5) / 1.0).clamp(0, 1)
    layer_norm = weighted_prop("cloud_layer", 1.0).clamp(0, 1)
    texture_norm = weighted_prop("texture_proxy", 1.0).clamp(0, 1)
    cirrus_map = weighted_prop("cirrus_proxy", 1.0).clamp(0, 1)
    property_maps = torch.cat([probability_map, aot_norm, layer_norm, texture_norm, cirrus_map], dim=1)

    generated_mask = original_mask.clone()
    generated_mask = torch.clamp(generated_mask * (1.0 - remove_map), 0.0, 1.0)
    generated_mask = torch.clamp(generated_mask + create_map * probability_map, 0.0, 1.0)
    # Modify does not erase or create by itself; it marks property changes over existing/nearby clouds.
    generated_mask = torch.clamp(generated_mask + 0.25 * modify_map * original_mask, 0.0, 1.0)

    action_maps = torch.cat([
        (blobs * noop_w[:, :, None, None]).amax(dim=1, keepdim=True),
        remove_map,
        modify_map,
        create_map,
        probability_map,
        aot_norm,
        layer_norm,
        texture_norm,
    ], dim=1)

    return {
        "generated_mask": generated_mask,
        "property_maps": property_maps,
        "action_maps": action_maps,
        "decoded": decoded,
    }


def actions_to_jsonable(op: torch.Tensor, params_tanh: torch.Tensor) -> list:
    """Convert one sample's action tensors to JSON-serializable intervention list."""
    op_names = {NOOP: "noop", REMOVE: "remove", MODIFY: "modify", CREATE: "create"}
    dec = decode_params(params_tanh[None, ...])
    out = []
    for k in range(op.shape[0]):
        oi = int(op[k].item())
        if oi == NOOP:
            continue
        out.append({
            "op": op_names[oi],
            "x01": float(dec["x01"][0, k].item()),
            "y01": float(dec["y01"][0, k].item()),
            "radius_px": float(dec["radius_px"][0, k].item()),
            "cloud_probability": float(dec["cloud_probability"][0, k].item()),
            "aot_proxy": float(dec["aot_proxy"][0, k].item()),
            "cirrus_proxy": float(dec["cirrus_proxy"][0, k].item()),
            "cloud_layer": float(dec["cloud_layer"][0, k].item()),
            "texture_proxy": float(dec["texture_proxy"][0, k].item()),
        })
    return out
