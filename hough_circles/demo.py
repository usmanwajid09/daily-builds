"""Demo: Hough circle detection on a synthetic edge map.

Draws two circles (centre (25, 20) radius 10 and centre (55, 40)
radius 14) on a 60x80 edge map, runs :func:`hough_circles.detect_circles`
and prints each detected circle's centre, radius and vote count.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hough_circles.hough_circles import circle_edge_map, detect_circles


def make_edge_image(height=60, width=80):
    """Return a synthetic edge map with two circles drawn on it."""
    return circle_edge_map((height, width), [(25, 20, 10), (55, 40, 14)])


def main():
    edges = make_edge_image()
    print(f"edge map: {edges.shape}, {int(edges.sum())} edge pixels")
    circles = detect_circles(edges, radii=[8, 10, 12, 14, 16])
    print(f"detected {len(circles)} circle(s):")
    for circle in circles:
        print(
            f"  centre=({circle.x}, {circle.y}) radius={circle.radius} "
            f"votes={circle.votes}"
        )


if __name__ == "__main__":
    main()
