"""Sliding-window object detector from scratch (numpy only)."""

from sliding_window_detector.detector import (
    LogisticClassifier,
    SlidingWindowDetector,
    feature_length,
    hog_features,
    iou,
    make_circle_window,
    make_negative_window,
    make_scene,
    non_max_suppression,
)

__all__ = [
    "LogisticClassifier",
    "SlidingWindowDetector",
    "feature_length",
    "hog_features",
    "iou",
    "make_circle_window",
    "make_negative_window",
    "make_scene",
    "non_max_suppression",
]
