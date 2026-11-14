"""Demo: LoG blob detection on a synthetic image.

Builds a 100x120 grayscale image — three bright discs of radii 4, 8
and 14 on a black background — runs
:func:`blob_detection.blobs.detect_blobs` on it, and prints how many
blobs were found and their (row, col, radius).
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python blob_detection/demo.py`):
# put the repo root on sys.path so the package import below resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blob_detection.blobs import detect_blobs

# (row, col, radius) of the discs drawn in the synthetic image.
DISCS = [(25, 25, 4), (65, 30, 8), (30, 90, 14)]


def make_test_image(height=100, width=120):
    """Return a synthetic grayscale image: bright discs on black."""
    image = np.zeros((height, width), dtype=np.float64)
    rows, cols = np.mgrid[0:height, 0:width]
    for row, col, radius in DISCS:
        image[(rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2] = 1.0
    return image


def main():
    image = make_test_image()
    blobs = detect_blobs(image, min_sigma=1.0, max_sigma=10.0, num_sigma=5)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale")
    print(f"true discs: {len(DISCS)}")
    print(f"blobs found: {len(blobs)}")
    for row, col, radius, response in blobs:
        print(
            f"  blob at (row={row}, col={col}), "
            f"radius={radius:.2f}, response={response:.3f}"
        )
    return blobs


if __name__ == "__main__":
    main()
