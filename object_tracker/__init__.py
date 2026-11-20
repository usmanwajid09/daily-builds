"""Template-based object tracking across synthetic frames (numpy only)."""

from object_tracker.model import (
    ObjectTracker,
    bbox_centre,
    make_frame,
    make_sequence,
    mean_drift,
    ncc,
)

__all__ = [
    "ObjectTracker",
    "bbox_centre",
    "make_frame",
    "make_sequence",
    "mean_drift",
    "ncc",
]
