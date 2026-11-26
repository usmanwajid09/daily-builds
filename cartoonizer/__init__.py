"""Public API for the cartoonizer package."""

from .model import (
    cartoonize,
    edge_mask,
    edge_preserving_smooth,
    quantize_colors,
)

__all__ = [
    "cartoonize",
    "edge_preserving_smooth",
    "quantize_colors",
    "edge_mask",
]
