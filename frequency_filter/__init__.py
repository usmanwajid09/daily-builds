"""Frequency-domain filtering: low-pass and high-pass with FFT masks."""

from .filter import band_pass, high_pass, low_pass, make_mask

__all__ = ["make_mask", "low_pass", "high_pass", "band_pass"]
