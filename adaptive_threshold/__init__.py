"""Adaptive (local) thresholding, with a global-threshold comparison."""

from adaptive_threshold.adaptive_threshold import (
    adaptive_threshold,
    compare_thresholds,
    global_threshold,
)

__all__ = ["adaptive_threshold", "compare_thresholds", "global_threshold"]
