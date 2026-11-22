"""Simplified SIFT-like feature detection and description (NumPy only)."""

from sift_features.sift_features import (
    SIFTFeatures,
    gaussian_blur,
    gaussian_kernel_1d,
    match_descriptors,
)

__all__ = [
    "SIFTFeatures",
    "gaussian_blur",
    "gaussian_kernel_1d",
    "match_descriptors",
]
