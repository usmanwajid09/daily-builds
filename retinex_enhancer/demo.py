"""Demo: Retinex enhancement of a dark, unevenly lit image.

Builds a 64x64 RGB image that is dark overall with a smooth brightness
gradient (a vignette) plus a brighter disc, runs single-scale Retinex,
multi-scale Retinex and MSRCR on it, and prints mean/std before and
after each method.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retinex_enhancer.retinex import (
    enhance,
    multi_scale_retinex,
    multi_scale_retinex_cr,
    single_scale_retinex,
)


def make_test_image(height=64, width=64):
    """Dark image (values ~10-60) with a left-to-right lighting gradient."""
    rows, cols = np.mgrid[0:height, 0:width]
    gradient = 10.0 + 40.0 * (cols / width)  # dim on the left, brighter right
    image = np.repeat(gradient[:, :, None], 3, axis=2)
    disc = (rows - 32) ** 2 + (cols - 32) ** 2 <= 12 ** 2
    image[disc] = [90.0, 70.0, 50.0]
    return image


def _stats(name, image):
    print(f"{name:>12}: mean={image.mean():7.2f}  std={image.std():6.2f}  "
          f"range=[{image.min():.0f}, {image.max():.0f}]")


def main():
    image = make_test_image()
    _stats("original", image)
    _stats("SSR", single_scale_retinex(image, sigma=15.0))
    _stats("MSR", multi_scale_retinex(image))
    _stats("MSRCR", multi_scale_retinex_cr(image))
    enhanced = enhance(image, method="msr")
    print(f"enhance('msr') dtype={enhanced.dtype}, shape={enhanced.shape}")
    return enhanced


if __name__ == "__main__":
    main()
