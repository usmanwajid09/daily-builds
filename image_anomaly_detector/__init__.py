"""Image anomaly detection via PCA reconstruction error (numpy only)."""

from image_anomaly_detector.anomaly import (
    PCAAnomalyDetector,
    detect_anomalies,
    extract_patches,
    reconstruction_error,
    train_detector,
)

__all__ = [
    "PCAAnomalyDetector",
    "detect_anomalies",
    "extract_patches",
    "reconstruction_error",
    "train_detector",
]
