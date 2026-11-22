"""Single-image dehazing via the dark channel prior (numpy only)."""

from dark_channel_dehazing.dehazing import (
    dark_channel,
    dehaze,
    estimate_atmospheric_light,
    estimate_transmission,
    make_clean_scene,
    make_hazy_image,
    recover,
)

__all__ = [
    "dark_channel",
    "dehaze",
    "estimate_atmospheric_light",
    "estimate_transmission",
    "make_clean_scene",
    "make_hazy_image",
    "recover",
]
