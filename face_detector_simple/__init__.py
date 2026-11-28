"""Simple face detection via skin-colour segmentation (numpy only)."""

from face_detector_simple.detection import (
    detect_faces,
    rgb_to_ycbcr,
    skin_mask,
)

__all__ = ["detect_faces", "rgb_to_ycbcr", "skin_mask"]
