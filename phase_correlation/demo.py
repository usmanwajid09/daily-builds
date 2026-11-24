"""Demo: phase correlation recovers a known sub-pixel shift.

Builds a synthetic 64x64 image (seeded random blobs, blurred so the
spectrum is well behaved), shifts it by (dy=+3.5, dx=-2.25) pixels with
the Fourier shift theorem, and prints the shift phase correlation
estimates plus the error.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_correlation.phase_correlation import (
    estimate_translation,
    shift_image_fourier,
)


def make_test_image(height=64, width=64, seed=0):
    """Smooth random-blobs image with a non-trivial spectrum."""
    rng = np.random.default_rng(seed)
    image = rng.random((height, width))
    kernel = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2)
    kernel /= kernel.sum()
    for axis in (0, 1):
        image = np.apply_along_axis(
            lambda row: np.convolve(np.pad(row, 3, mode="wrap"), kernel, mode="valid"),
            axis, image)
    return image * 255.0


def main():
    image = make_test_image()
    true_dy, true_dx = 3.5, -2.25
    shifted = shift_image_fourier(image, true_dy, true_dx)
    result = estimate_translation(image, shifted)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale")
    print(f"true shift: dy={true_dy}, dx={true_dx}")
    print(f"estimated:  dy={result['dy']:.3f}, dx={result['dx']:.3f}")
    print(f"error: dy={result['dy'] - true_dy:+.3f}px, "
          f"dx={result['dx'] - true_dx:+.3f}px")
    print(f"correlation peak: {result['peak']:.4f}")
    return result


if __name__ == "__main__":
    main()
