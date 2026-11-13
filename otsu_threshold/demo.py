"""Demo: Otsu's thresholding on a synthetic bimodal image.

Builds a 40x60 grayscale image with two clear populations — a dark
region around 50 and a bright region around 200, each with a little
noise — runs :func:`otsu_threshold.otsu.otsu_threshold` on it, and
prints the threshold found plus the foreground pixel share.
"""

import numpy as np

from otsu_threshold.otsu import binarize, otsu_threshold


def make_test_image(height=40, width=60, seed=0):
    """Return a synthetic bimodal image: dark left, bright right."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.float64)
    image[:, :width // 2] = rng.normal(50.0, 10.0, size=(height, width // 2))
    image[:, width // 2:] = rng.normal(200.0, 10.0, size=(height, width - width // 2))
    return image


def main():
    image = make_test_image()
    threshold = otsu_threshold(image)
    mask, _ = binarize(image)
    foreground = int(mask.sum())
    total = image.size
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, two modes ~50 and ~200")
    print(f"Otsu threshold: {threshold}")
    print(f"foreground pixels: {foreground} of {total} ({100.0 * foreground / total:.1f}%)")
    return threshold


if __name__ == "__main__":
    main()
