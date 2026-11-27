"""Public API for the floyd_steinberg_dithering package."""

from .model import dither_color, dither_grayscale

__all__ = ["dither_grayscale", "dither_color"]
