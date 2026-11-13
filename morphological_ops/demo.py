"""Demo: morphological opening on a synthetic noisy binary image.

Builds a 30x40 binary image — a solid 12x16 rectangle plus scattered
single-pixel noise — and prints the foreground pixel count before and
after opening (erosion then dilation). Opening should remove the
isolated noise pixels while leaving the rectangle (almost) untouched,
and closing should fill the small hole punched in the rectangle.
"""

import numpy as np

from morphological_ops.morphology import closing, dilate, erode, opening


def make_test_image(height=30, width=40, seed=0):
    """Return a noisy binary image: rectangle + specks + one hole."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=bool)
    image[9:21, 12:28] = True  # 12x16 solid rectangle
    image[15, 20] = False      # one-pixel hole inside the rectangle
    noise = rng.random((height, width)) < 0.03
    image |= noise
    return image


def main():
    image = make_test_image()
    opened = opening(image)
    closed = closing(image)
    eroded = erode(image)
    dilated = dilate(image)
    print(f"image: {image.shape[0]}x{image.shape[1]} binary, rectangle + noise specks")
    print(f"foreground pixels before:  {int(image.sum())}")
    print(f"after erosion:             {int(eroded.sum())}")
    print(f"after dilation:            {int(dilated.sum())}")
    print(f"after opening (noise out): {int(opened.sum())}")
    print(f"after closing (hole in):   {int(closed.sum())}")
    return opened


if __name__ == "__main__":
    main()
