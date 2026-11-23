"""Synthetic traffic-sign classifier (numpy only, kNN on colour+shape)."""

from traffic_sign_classifier.signs import (
    CLASSES,
    TrafficSignClassifier,
    accuracy,
    extract_features,
    make_dataset,
    make_sign,
)

__all__ = [
    "CLASSES",
    "TrafficSignClassifier",
    "accuracy",
    "extract_features",
    "make_dataset",
    "make_sign",
]
