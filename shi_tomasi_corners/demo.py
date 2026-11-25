"""Demo: Shi-Tomasi corner detection on a synthetic image.

Builds a 30x30 grayscale image — a bright square on a dark
background — computes the Shi-Tomasi corner response, and prints the
strongest response plus the detected corners (the square's four
corners) with their strengths.
"""

import os
import sys

import numpy as np

# Allow running as `python shi_tomasi_corners/demo.py` from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shi_tomasi_corners.shi_tomasi_corners import corner_response, detect_corners


def make_test_image(height=30, width=30):
    """Bright square on a dark background."""
    image = np.zeros((height, width), dtype=np.float64)
    image[10:20, 10:20] = 200.0
    return image


def main():
    image = make_test_image()
    response = corner_response(image)
    corners = detect_corners(image, max_corners=10, min_distance=5)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, bright square")
    print(
        f"response: min={response.min():.1f} max={response.max():.1f} "
        f"mean={response.mean():.2f}"
    )
    print(f"corners detected: {len(corners)}")
    for row, col, strength in corners:
        print(f"  corner at (row={row}, col={col}) strength={strength:.1f}")
    print("expected: the 4 corners of the square at rows/cols 10 and 19")
    return response, corners


if __name__ == "__main__":
    main()
