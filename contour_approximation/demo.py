"""Demo: approximate the contour of a synthetic binary image.

Builds a 50x70 binary image with a filled rectangle (rows 10-19,
cols 20-39) and a filled disc (centre (50, 38), radius 8) — the same
image as the contour_detection demo — then traces and simplifies
the first region's contour at two tolerances and prints the vertex
count, area and perimeter for each.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from contour_approximation.contour_approximation import approximate_contour, trace_contour


def make_binary_image(height=50, width=70):
    """Return a binary image with a filled rectangle and a filled disc."""
    binary = np.zeros((height, width), dtype=bool)
    binary[10:20, 20:40] = True  # 10 rows x 20 cols rectangle
    yy, xx = np.mgrid[0:height, 0:width]
    binary |= (xx - 50) ** 2 + (yy - 38) ** 2 <= 8 ** 2  # disc, radius 8
    return binary


def main():
    binary = make_binary_image()
    raw = trace_contour(binary)
    print(f"binary image: {binary.shape}, {int(binary.sum())} foreground pixels")
    print(f"traced contour: {len(raw)} raw boundary points")
    for epsilon in (0.5, 2.0):
        result = approximate_contour(binary, epsilon=epsilon)
        vertices = result["vertices"]
        print(
            f"  epsilon={epsilon}: {len(vertices) - 1} vertices, "
            f"area={result['area']:.1f}, perimeter={result['perimeter']:.1f}"
        )
        print(f"    vertices: {vertices[:-1].astype(int).tolist()}")


if __name__ == "__main__":
    main()
