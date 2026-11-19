"""Demo: median filtering on a noisy synthetic image.

Builds a 40x60 grayscale image — a bright disc and a bright rectangle
on a dark background — corrupts 8% of its pixels with salt-and-pepper
noise, median-filters it with a 3x3 window, and prints the noisy-pixel
count and the mean absolute error against the clean image before and
after.
"""

import os
import sys

import numpy as np

# Allow running as `python median_filter/demo.py` from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from median_filter.filter import median_filter


def make_test_image(height=40, width=60, seed=0, noise_fraction=0.08):
    """Return (clean, noisy) images: disc + rectangle, salt-and-pepper added."""
    rng = np.random.default_rng(seed)
    clean = np.zeros((height, width), dtype=np.float64)
    rows, cols = np.mgrid[0:height, 0:width]
    clean[(rows - 20) ** 2 + (cols - 18) ** 2 <= 10 ** 2] = 200.0
    clean[8:20, 38:54] = 150.0
    noisy = clean.copy()
    flip = rng.random(size=clean.shape) < noise_fraction
    salt = rng.random(size=clean.shape) < 0.5
    noisy[flip & salt] = 255.0
    noisy[flip & ~salt] = 0.0
    return clean, noisy


def main():
    clean, noisy = make_test_image()
    denoised = median_filter(noisy, kernel_size=3)
    # Pixels at 0/255 include the clean image's own black background, so
    # the honest noise measure is pixels that differ from the clean image.
    wrong_before = int(np.count_nonzero(noisy != clean))
    wrong_after = int(np.count_nonzero(denoised != clean))
    mae_before = float(np.mean(np.abs(noisy - clean)))
    mae_after = float(np.mean(np.abs(denoised - clean)))
    print(f"image: {clean.shape[0]}x{clean.shape[1]} grayscale "
          f"(disc + rectangle, 8% salt-and-pepper noise)")
    print(f"pixels differing from the clean image: {wrong_before} noisy "
          f"-> {wrong_after} after 3x3 median filter")
    print(f"mean absolute error vs clean image: {mae_before:.2f} -> {mae_after:.2f}")
    return denoised


if __name__ == "__main__":
    main()
