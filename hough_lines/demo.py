"""Demo: Hough line detection on a synthetic edge map.

Builds an 80x100 edge map with two lines — a diagonal (y = x) and a
horizontal line (row 20) — runs :func:`hough_lines.hough.detect_lines`
and prints each detected line's rho, theta, vote count and endpoints.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hough_lines.hough import detect_lines, line_endpoints


def make_edge_image(height=80, width=100):
    """Return a synthetic edge map: diagonal line + horizontal line."""
    edges = np.zeros((height, width), dtype=bool)
    for i in range(10, 51):
        edges[i, i] = True  # diagonal, y = x
    edges[20, 10:90] = True  # horizontal line, row 20
    return edges


def main():
    edges = make_edge_image()
    lines = detect_lines(edges, threshold=15)
    print(f"edge pixels: {int(edges.sum())}")
    print(f"lines found: {len(lines)}")
    for rho, theta, votes in lines:
        (x1, y1), (x2, y2) = line_endpoints(rho, theta, edges.shape)
        print(f"  rho={rho:.1f} theta={theta:.0f} deg votes={votes} "
              f"endpoints=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
    return lines


if __name__ == "__main__":
    main()
