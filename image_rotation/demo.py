"""Demo: rotate a synthetic image by 45 degrees, both methods.

Builds a 21x21 grayscale image with a bright diagonal line, rotates it
45 degrees counter-clockwise with nearest-neighbour and bilinear
interpolation, and prints the shape, pixel sums and max-abs difference
from the input for a 0-degree rotation, plus a small ASCII rendering
before and after so you can see the diagonal turn horizontal.
"""

import os
import sys

# Allow `python image_rotation/demo.py` from the repo root: running a
# script by path puts its own folder on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from image_rotation.rotate import rotate_image


def make_test_image(size=21):
    """Return a square grayscale image with a bright diagonal line."""
    image = np.zeros((size, size), dtype=np.float64)
    idx = np.arange(size)
    image[idx, idx] = 255.0
    return image


def render_ascii(image, threshold=64.0):
    """Render a grayscale image as ASCII art ('#' = bright pixel)."""
    lines = []
    for row in image:
        lines.append("".join("#" if p >= threshold else " " for p in row))
    return "\n".join(lines)


def main():
    image = make_test_image()
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, "
          f"sum = {image.sum():.1f}")

    # Sanity check: 0 degrees must return the input (up to float rounding).
    identity = rotate_image(image, 0)
    print(f"0-degree rotation: max abs diff from input = "
          f"{np.abs(identity - image).max():.2e}")

    for method in ("nearest", "bilinear"):
        rotated = rotate_image(image, 45, method=method)
        print(f"\n45 degrees, {method}: shape {rotated.shape}, "
              f"sum = {rotated.sum():.1f}")
    print("\nbefore (diagonal line):")
    print(render_ascii(image))
    print("\nafter, 45 degrees bilinear (line turns horizontal):")
    print(render_ascii(rotate_image(image, 45, method="bilinear")))
    return image


if __name__ == "__main__":
    main()
