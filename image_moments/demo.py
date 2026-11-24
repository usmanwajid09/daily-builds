"""Demo: image moments on a synthetic shape.

Builds a 60x80 binary image (a disc plus a rectangle), prints its area,
centroid, orientation and Hu moments, then shows the Hu moments barely
change when the same shape is translated — the point of the invariants.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_moments.moments import hu_moments, moments_summary


def make_test_image(height=60, width=80):
    """Binary image: a disc and a rectangle on a black background."""
    image = np.zeros((height, width), dtype=np.float64)
    rows, cols = np.mgrid[0:height, 0:width]
    image[(rows - 30) ** 2 + (cols - 25) ** 2 <= 12 ** 2] = 1.0
    image[10:25, 50:71] = 1.0
    return image


def main():
    image = make_test_image()
    summary = moments_summary(image)
    print(f"image: {image.shape[0]}x{image.shape[1]} binary")
    print(f"area (m00): {summary['area']:.0f} pixels")
    cx, cy = summary["centroid"]
    print(f"centroid: x={cx:.2f}, y={cy:.2f}")
    print(f"orientation: {summary['orientation_rad']:.4f} rad "
          f"({np.degrees(summary['orientation_rad']):.1f} deg)")
    print("Hu moments:")
    for i, h in enumerate(summary["hu_moments"], start=1):
        print(f"  h{i} = {h:.6g}")
    shifted = np.roll(np.roll(image, 5, axis=0), -8, axis=1)
    hu_shifted = hu_moments(shifted)
    rel = np.abs(hu_shifted - summary["hu_moments"]) / np.maximum(
        np.abs(summary["hu_moments"]), 1e-12)
    print(f"max relative Hu change after translation: {rel.max():.2e}")
    return summary


if __name__ == "__main__":
    main()
