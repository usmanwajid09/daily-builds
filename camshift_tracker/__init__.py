"""CamShift colour tracking, implemented from scratch (numpy only)."""

from camshift_tracker.camshift import (
    backproject,
    build_histogram,
    camshift_track,
    mean_shift,
    quantise_colour,
    track_sequence,
)

__all__ = [
    "backproject",
    "build_histogram",
    "camshift_track",
    "mean_shift",
    "quantise_colour",
    "track_sequence",
]
