"""Demo: Sobel edge detection on a synthetic image.

Builds a 60x80 grayscale image — a bright disc and a bright rectangle on
a dark background — runs :func:`edge_detector.edges.detect_edges` on it,
and prints how many edge pixels were found plus a small ASCII rendering
of the edge map so you can see the outlines.
"""

import numpy as np

from edge_detector.edges import detect_edges, edge_pixel_count


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
    """Render a boolean edge map as ASCII art ('#' = edge)."""
    lines = []
    for row in edge_map[::row_step]:
        lines.append("".join("#" if p else " " for p in row[::col_step]))
    return "\n".join(lines)


def main():
    image = make_test_image()
    edges = detect_edges(image, threshold=0.15)
    count = edge_pixel_count(edges)
    total = image.size
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale")
    print(f"edge pixels: {count} of {total} ({100.0 * count / total:.1f}%)")
    print(render_ascii(edges))
    return edges


if __name__ == "__main__":
    main()
