"""Demo: histogram backprojection localises a coloured disc.

Builds a 80x100 RGB image — a red disc on a green background with a
blue distractor disc — builds a colour histogram from a small ROI on
the red disc, backprojects it, and prints the peak and centroid of the
probability map next to the true disc centre.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from histogram_backprojection.backprojection import build_histogram, localise


def make_test_image(height=80, width=100):
    """Red disc at (60, 30), blue distractor at (25, 55), green background."""
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:, :, 1] = 120.0
    rows, cols = np.mgrid[0:height, 0:width]
    red = (rows - 30) ** 2 + (cols - 60) ** 2 <= 12 ** 2
    blue = (rows - 55) ** 2 + (cols - 25) ** 2 <= 10 ** 2
    image[red] = [220.0, 30.0, 30.0]
    image[blue] = [30.0, 30.0, 220.0]
    return image


def main():
    image = make_test_image()
    hist = build_histogram(image, roi=(54, 24, 12, 12), bins=16)
    result = localise(image, hist, threshold=0.5)
    px, py = result["peak"]
    cx, cy = result["centroid"]
    print(f"image: {image.shape[1]}x{image.shape[0]} RGB, true red disc centre: (60, 30)")
    print(f"peak: ({px}, {py}) value={result['peak_value']:.3f}")
    print(f"centroid of high-probability region: ({cx:.2f}, {cy:.2f})")
    print(f"centroid error vs true centre: {np.hypot(cx - 60, cy - 30):.2f}px")
    return result


if __name__ == "__main__":
    main()
