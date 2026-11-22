"""Frame-differencing motion detector from scratch (numpy only)."""

from motion_detector.motion_detector import (
    MotionDetector,
    connected_components,
    detect_motion,
)

__all__ = [
    "MotionDetector",
    "connected_components",
    "detect_motion",
]
