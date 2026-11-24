"""Licence-plate detection in synthetic grayscale images (numpy only)."""

from plate_detector.plate import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    box_iou,
    component_boxes,
    detect_plates,
    extract_plate_crop,
    label_components,
    make_plate_scene,
    match_detections,
    threshold_image,
)

__all__ = [
    "make_plate_scene",
    "threshold_image",
    "binary_dilation",
    "binary_erosion",
    "binary_closing",
    "binary_opening",
    "label_components",
    "component_boxes",
    "detect_plates",
    "extract_plate_crop",
    "box_iou",
    "match_detections",
]
