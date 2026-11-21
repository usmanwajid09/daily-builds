"""Mean-shift colour-histogram object tracking from scratch."""

from mean_shift_tracker.mean_shift_tracker import (
    MeanShiftTracker,
    color_histogram,
    moving_square_frame,
)

__all__ = ["MeanShiftTracker", "color_histogram", "moving_square_frame"]
