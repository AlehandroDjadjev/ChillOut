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
    humidity_pct: Tuple[float, float] = (0.0, 100.0)
    optical_thickness: Tuple[float, float] = (0.0, 100.0)
    cloud_top_height_m: Tuple[float, float] = (0.0, 12000.0)
    mass_proxy: Tuple[float, float] = (0.0, 1.0)
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

    params dims: x, y, radius, mass, humidity, optical_thickness, height, cloud_type
    x,y remain [0,1] normalized image coordinates. cloud_type is [0,1] continuous proxy.
    """
    u = (params_tanh + 1.0) * 0.5
    return {
        "x01": u[..., 0],
        "y01": u[..., 1],
        "radius_px": _scale01(u[..., 2], *ranges.radius_px),
        "mass_proxy": _scale01(u[..., 3], *ranges.mass_proxy),
        "humidity_pct": _scale01(u[..., 4], *ranges.humidity_pct),
        "optical_thickness": _scale01(u[..., 5], *ranges.optical_thickness),
        "cloud_top_height_m": _scale01(u[..., 6], *ranges.cloud_top_height_m),
        "cloud_type": u[..., 7],
    }


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
    property_maps channels: humidity_norm, optical_thickness_norm, height_norm, mass_proxy, cloud_type.
    action_maps channels: noop, remove, modify, create, mass, humidity, thickness, height.
    """
    b, _, h, w = original_mask.shape
    decoded = decode_params(params_tanh, ranges)
    blobs = gaussian_blobs(decoded["x01"], decoded["y01"], decoded["radius_px"], h, w)

    noop_w = (op == NOOP).float()
    remove_w = (op == REMOVE).float()
    modify_w = (op == MODIFY).float()
    create_w = (op == CREATE).float()
    active_w = 1.0 - noop_w

    remove_map = (blobs * remove_w[:, :, None, None]).amax(dim=1, keepdim=True)
    create_map = (blobs * create_w[:, :, None, None]).amax(dim=1, keepdim=True)
    modify_map = (blobs * modify_w[:, :, None, None]).amax(dim=1, keepdim=True)

    generated_mask = original_mask.clone()
    generated_mask = torch.clamp(generated_mask * (1.0 - remove_map), 0.0, 1.0)
    generated_mask = torch.clamp(generated_mask + create_map, 0.0, 1.0)
    # Modify does not erase or create by itself; it marks property changes over existing/nearby clouds.
    generated_mask = torch.clamp(generated_mask + 0.25 * modify_map * original_mask, 0.0, 1.0)

    def weighted_prop(name: str, norm_divisor: float = 1.0) -> torch.Tensor:
        val = decoded[name] / norm_divisor
        weights = blobs * active_w[:, :, None, None]
        num = (weights * val[:, :, None, None]).sum(dim=1, keepdim=True)
        den = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.where(den > 1e-5, num / den, torch.zeros_like(num))

    humidity_norm = weighted_prop("humidity_pct", 100.0).clamp(0, 1)
    thickness_norm = weighted_prop("optical_thickness", 100.0).clamp(0, 1)
    height_norm = weighted_prop("cloud_top_height_m", 12000.0).clamp(0, 1)
    mass_map = weighted_prop("mass_proxy", 1.0).clamp(0, 1)
    type_map = weighted_prop("cloud_type", 1.0).clamp(0, 1)
    property_maps = torch.cat([humidity_norm, thickness_norm, height_norm, mass_map, type_map], dim=1)

    action_maps = torch.cat([
        (blobs * noop_w[:, :, None, None]).amax(dim=1, keepdim=True),
        remove_map,
        modify_map,
        create_map,
        mass_map,
        humidity_norm,
        thickness_norm,
        height_norm,
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
            "mass_proxy": float(dec["mass_proxy"][0, k].item()),
            "humidity_pct": float(dec["humidity_pct"][0, k].item()),
            "optical_thickness": float(dec["optical_thickness"][0, k].item()),
            "cloud_top_height_m": float(dec["cloud_top_height_m"][0, k].item()),
            "cloud_type": float(dec["cloud_type"][0, k].item()),
        })
    return out
