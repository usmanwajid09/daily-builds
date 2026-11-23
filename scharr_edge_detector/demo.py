"""Demo: Scharr edge detection on a synthetic image.

Builds a 40x60 grayscale image — a bright disc and a bright rectangle
on a dark background — runs Scharr edge detection on it, and prints the
gradient statistics plus how many edge pixels survive a threshold, with
a small ASCII rendering of the edge map.
"""

import os
import sys

import numpy as np

# Allow running as `python scharr_edge_detector/demo.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scharr_edge_detector.scharr import scharr_edges, threshold_edges


def make_test_image(height=40, width=60):
    """Bright disc + bright rectangle on a dark background."""
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width), dtype=np.float64)
    image[(yy - 15) ** 2 + (xx - 15) ** 2 <= 8 ** 2] = 200.0
    image[25:35, 40:55] = 180.0
    return image


def main():
    image = make_test_image()
    magnitude, direction = scharr_edges(image)
    threshold = 200.0
    edges = threshold_edges(magnitude, threshold)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, disc + rectangle")
    print(f"magnitude: min={magnitude.min():.1f} max={magnitude.max():.1f} "
          f"mean={magnitude.mean():.1f}")
    print(f"direction at disc edge (row 15, col 7): "
          f"{np.degrees(direction[15, 7]):.1f} degrees")
    print(f"edge pixels (magnitude >= {threshold:.0f}): {int(edges.sum())} "
          f"of {edges.size}")
    # ASCII rendering of the edge map, downsampled 2x.
    print("edge map (2x downsampled, # = edge):")
    for row in edges[::2]:
        print("".join("#" if v else "." for v in row[::2]))
    return magnitude, direction


if __name__ == "__main__":
    main()
