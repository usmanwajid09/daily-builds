"""Simple watershed segmentation from scratch (numpy only)."""

from watershed_segmentation.watershed import (
    WATERSHED,
    distance_transform,
    find_markers,
    segment_count,
    watershed,
    watershed_lines,
)

__all__ = ["distance_transform", "find_markers", "watershed",
           "segment_count", "watershed_lines", "WATERSHED"]
