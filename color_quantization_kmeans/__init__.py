"""Colour quantisation via k-means clustering (numpy only)."""

from color_quantization_kmeans.quantization import (
    kmeans_quantize,
    quantization_mse,
)

__all__ = ["kmeans_quantize", "quantization_mse"]
