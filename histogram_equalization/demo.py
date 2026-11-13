"""Demo: histogram equalization on a synthetic low-contrast image.

Builds a 40x60 grayscale image whose values only span roughly 100-140
(a smooth gradient plus a little noise), equalizes it, and prints the
min/max/mean before and after so you can see the range stretch to the
full 0-255.
"""

import numpy as np

from histogram_equalization.equalize import contrast_stretch, equalize_histogram


def make_test_image(height=40, width=60, seed=0):
    """Return a synthetic low-contrast image (values ~100-140)."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:height, 0:width]
    image = 110.0 + 0.4 * rows + 0.3 * cols
    image += rng.normal(0.0, 3.0, size=image.shape)
    return image


def _stats(image):
    return (
        f"min={image.min():3.0f} max={image.max():3.0f} "
        f"mean={image.mean():6.1f} std={image.std():5.1f}"
    )


def main():
    image = make_test_image()
    equalized = equalize_histogram(image)
    stretched = contrast_stretch(image)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, low contrast")
    print(f"before:         {_stats(image)}")
    print(f"equalized:      {_stats(equalized)}")
    print(f"contrast stretch: {_stats(stretched)}")
    return equalized


if __name__ == "__main__":
    main()
