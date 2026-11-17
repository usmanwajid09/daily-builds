"""Demo: contour tracing on a synthetic binary image.

Builds a 50x70 binary image with a filled rectangle (rows 10-19,
cols 20-39) and a filled disc (centre (50, 38), radius 8), then runs
:func:`contour_detection.find_contours` and prints each contour's
point count, area, perimeter and bounding box.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from contour_detection.contour_detection import find_contours


def make_binary_image(height=50, width=70):
    """Return a binary image with a filled rectangle and a filled disc."""
    binary = np.zeros((height, width), dtype=bool)
    binary[10:20, 20:40] = True  # 10 rows x 20 cols rectangle
    yy, xx = np.mgrid[0:height, 0:width]
    binary |= (xx - 50) ** 2 + (yy - 38) ** 2 <= 8 ** 2  # disc, radius 8
    return binary


def main():
    binary = make_binary_image()
    print(f"binary image: {binary.shape}, {int(binary.sum())} foreground pixels")
    contours = find_contours(binary)
    print(f"found {len(contours)} contour(s):")
    for i, contour in enumerate(contours):
        x_min, y_min, x_max, y_max = contour.bbox
        print(
            f"  contour {i}: {len(contour.points)} points, "
            f"area={contour.area:.1f}, perimeter={contour.perimeter:.1f}, "
            f"bbox=({x_min}, {y_min})-({x_max}, {y_max})"
        )


if __name__ == "__main__":
    main()
