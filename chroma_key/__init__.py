"""Green-screen (chroma key) compositing, from scratch (numpy only)."""

from chroma_key.chroma import (
    color_distance,
    composite,
    green_mask,
    replaced_pixel_count,
)

__all__ = [
    "color_distance",
    "composite",
    "green_mask",
    "replaced_pixel_count",
]
