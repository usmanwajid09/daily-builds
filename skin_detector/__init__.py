"""Skin segmentation in YCbCr colour space (numpy only)."""

from skin_detector.skin import (
    CB_RANGE,
    CR_RANGE,
    detect_skin,
    rgb_to_ycbcr,
    skin_stats,
)

__all__ = [
    "CB_RANGE",
    "CR_RANGE",
    "detect_skin",
    "rgb_to_ycbcr",
    "skin_stats",
]
