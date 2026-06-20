from __future__ import annotations

import numpy as np


def rgb_cloud_mask(
    rgb: np.ndarray,
    brightness_threshold: float = 0.65,
    saturation_threshold: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple white/gray cloud filter for Sentinel-2-like RGB crops.

    Args:
        rgb: [H, W, 3], float array scaled 0..1.
        brightness_threshold: pixels brighter than this can be cloud.
        saturation_threshold: pixels less saturated than this can be cloud.

    Returns:
        cloud_mask: [H, W], float 0/1
        cloud_brightness: [H, W], brightness retained only where cloud_mask=1

    This is intentionally simple. It will confuse snow/ice/bright ground in some cases.
    Use official cloud probability/mask products when possible.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected rgb [H,W,3], got {rgb.shape}")

    rgb = np.clip(rgb.astype("float32"), 0.0, 1.0)
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    brightness = maxc
    saturation = (maxc - minc) / np.maximum(maxc, 1e-6)

    mask = (brightness >= brightness_threshold) & (saturation <= saturation_threshold)
    cloud_mask = mask.astype("float32")
    cloud_brightness = brightness * cloud_mask
    return cloud_mask, cloud_brightness


def make_anomaly(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Simple anomaly helper: actual - normal/baseline."""
    return values - baseline


def make_delta(values: np.ndarray, lag: int) -> np.ndarray:
    """Compute time-lag delta for [T,...] arrays.

    First lag frames are filled with 0.
    """
    out = np.zeros_like(values)
    out[lag:] = values[lag:] - values[:-lag]
    return out


def normalize_train_val_test(train, val, test, eps: float = 1e-6):
    """Channel-wise normalization for [T,C,H,W] arrays.

    Returns normalized arrays and mean/std.
    """
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    std = train.std(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(std, eps)
    return (train - mean) / std, (val - mean) / std, (test - mean) / std, mean, std
