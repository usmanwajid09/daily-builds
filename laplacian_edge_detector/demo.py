"""Demo: Laplacian edge detection on a synthetic image.

Builds a 40x60 grayscale image — a bright disc and a bright rectangle
on a dark background — runs the Laplacian operator on it, and prints
the response statistics, the zero-crossing count, and the edge-pixel
count above a threshold, with a small ASCII rendering of the edge map.
"""

import os
import sys

import numpy as np

# Allow running as `python laplacian_edge_detector/demo.py` from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laplacian_edge_detector.laplacian_edge_detector import (
    edge_pixel_count,
    laplacian,
    zero_crossings,
)


def make_test_image(height=40, width=60):
    """Bright disc + bright rectangle on a dark background."""
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width), dtype=np.float64)
    image[(yy - 15) ** 2 + (xx - 15) ** 2 <= 8 ** 2] = 200.0
    image[25:35, 40:55] = 180.0
    return image


def main():
    image = make_test_image()
    response = laplacian(image)
    crossings = zero_crossings(response, threshold=50.0)
    threshold = 150.0
    edges = response
    count = edge_pixel_count(image, threshold)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, disc + rectangle")
    print(
        f"laplacian: min={response.min():.1f} max={response.max():.1f} "
        f"mean={response.mean():.2f}"
    )
    print(f"response at disc centre (row 15, col 15): {response[15, 15]:.1f}")
    print(f"response at disc edge (row 15, col 7): {response[15, 7]:.1f}")
    print(f"zero crossings (threshold 50): {int(crossings.sum())} pixels")
    print(f"edge pixels (|laplacian| >= {threshold:.0f}): {count} of {edges.size}")
    # ASCII rendering of the zero-crossing map, downsampled 2x.
    print("zero-crossing map (2x downsampled, # = crossing):")
    for row in crossings[::2]:
        print("".join("#" if v else "." for v in row[::2]))
    return response, crossings


if __name__ == "__main__":
    main()
