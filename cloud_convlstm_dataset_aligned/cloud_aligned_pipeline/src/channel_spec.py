from __future__ import annotations

INPUT_CHANNELS = [
    # Cloud / cloud-physics channels
    "cloud_mask_or_fraction",
    "low_cloud_cover",
    "mid_cloud_cover",
    "high_cloud_cover",
    "cloud_liquid_water",
    "cloud_ice_water",
    "cloud_optical_or_brightness_proxy",
    "cloud_top_height_or_pressure_proxy",

    # Radiation bridge channels
    "surface_shortwave_down",
    "surface_shortwave_clear_sky",
    "surface_longwave_down",
    "surface_longwave_clear_sky",
    "surface_net_shortwave",
    "surface_net_longwave",

    # Dynamic atmosphere / rain support
    "temperature_anomaly",
    "dewpoint_or_relative_humidity",
    "total_column_water_vapour",
    "surface_pressure",
    "precipitation",
    "wind_u_or_speed",
]

TARGET_CHANNELS = [
    "shortwave_anomaly_next_24h",
    "longwave_anomaly_next_24h",
    "net_radiation_anomaly_next_24h",
    "temperature_anomaly_next_24h",
]

CLOUD_CHANNEL_INDICES = list(range(8))


def check_channel_counts(input_channels: int, target_channels: int) -> None:
    if input_channels != len(INPUT_CHANNELS):
        raise ValueError(
            f"Expected {len(INPUT_CHANNELS)} input channels, got {input_channels}. "
            f"Edit channel_spec.py/config.yaml if intentionally changing the model contract."
        )
    if target_channels != len(TARGET_CHANNELS):
        raise ValueError(
            f"Expected {len(TARGET_CHANNELS)} target channels, got {target_channels}."
        )
