"""Demo: affine warping of a synthetic image.

Builds a 60x80 image with a bright square in the middle, rotates it
30 degrees about its centre with
:func:`affine_warp.warp.warp_rotate`, and prints the total brightness
before/after (rotation should preserve it, roughly) plus the centre
and corner pixel values of the warped image.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from affine_warp.warp import warp_rotate, warp_translate


def make_test_image(height=60, width=80):
    """Return a black image with a bright 20x20 square in the middle."""
    image = np.zeros((height, width), dtype=np.float64)
    image[20:40, 30:50] = 255.0
    return image


def main():
    image = make_test_image()
    rotated = warp_rotate(image, 30.0)
    shifted = warp_translate(image, dx=5, dy=-3)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale")
    print(f"total brightness before: {image.sum():.0f}")
    print(f"total brightness after 30 deg rotation: {rotated.sum():.0f}")
    print(f"total brightness after translate (5, -3): {shifted.sum():.0f}")
    print(f"rotated centre pixel: {rotated[30, 40]:.1f}")
    print(f"rotated corner pixel: {rotated[0, 0]:.1f}")
    return rotated


if __name__ == "__main__":
    main()
