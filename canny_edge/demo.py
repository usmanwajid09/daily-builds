"""Demo: Canny edge detection on a synthetic image.

Builds a 60x80 grayscale image — a bright disc and a bright rectangle on
a dark background — runs :func:`canny_edge.canny.canny_edges` on it, and
prints how many edge pixels were found plus a small ASCII rendering of
the edge map so you can see the outlines.
"""

import os
import sys

import numpy as np

# Allow running as `python canny_edge/demo.py` from the repo root: the
# script's own directory lands on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canny_edge.canny import canny_edges, edge_pixel_count


def make_test_image(height=60, width=80):
    """Return a synthetic grayscale image: disc + rectangle on black."""
    image = np.zeros((height, width), dtype=np.float64)
    rows, cols = np.mgrid[0:height, 0:width]
    # Bright disc centred at (30, 25), radius 12.
    image[(rows - 30) ** 2 + (cols - 25) ** 2 <= 12 ** 2] = 255.0
    # Bright rectangle, rows 10-24, cols 50-70.
    image[10:25, 50:71] = 200.0
    return image


def render_ascii(edge_map, row_step=2, col_step=2):
    """Render a boolean edge map as ASCII art ('#' = edge).

    Each character covers a ``row_step`` x ``col_step`` block and shows
    '#' if any pixel in the block is an edge, so a one-pixel-wide edge
    never falls between sampled rows and disappears.
    """
    lines = []
    for r0 in range(0, edge_map.shape[0], row_step):
        line = []
        for c0 in range(0, edge_map.shape[1], col_step):
            block = edge_map[r0:r0 + row_step, c0:c0 + col_step]
            line.append("#" if block.any() else " ")
        lines.append("".join(line))
    return "\n".join(lines)


def main():
    image = make_test_image()
    edges = canny_edges(image, sigma=1.0, low_ratio=0.1, high_ratio=0.3)
    count = edge_pixel_count(edges)
    total = image.size
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale")
    print(f"edge pixels: {count} of {total} ({100.0 * count / total:.1f}%)")
    print(render_ascii(edges))
    return edges


if __name__ == "__main__":
    main()
