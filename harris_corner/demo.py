"""Demo: Harris corner detection on a synthetic image.

Builds a 60x80 grayscale image — a bright rectangle on a dark
background — runs :func:`harris_corner.corners.detect_corners` on it,
and prints the corner coordinates found. A rectangle has exactly four
corners, so the detector should find four, near the rectangle's
corners.
"""

import numpy as np

from harris_corner.corners import detect_corners, harris_response


def make_test_image(height=60, width=80):
    """Return a synthetic image: bright rectangle on black."""
    image = np.zeros((height, width), dtype=np.float64)
    image[15:45, 20:60] = 255.0
    return image


def main():
    image = make_test_image()
    response = harris_response(image)
    corners = detect_corners(image, max_corners=10)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, bright rectangle")
    print(f"rectangle corners at (15, 20), (15, 59), (44, 20), (44, 59)")
    print(f"Harris response peak: {response.max():.1f}")
    print(f"corners found: {len(corners)}")
    for row, col in corners:
        print(f"  corner at (row={row}, col={col}), response={response[row, col]:.1f}")
    return corners


if __name__ == "__main__":
    main()
