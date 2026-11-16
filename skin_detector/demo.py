"""Demo: skin segmentation in YCbCr on a synthetic image.

Builds a 120x160 scene: a dark-blue background, a green circle, and a
skin-coloured rectangle (RGB 210, 170, 140) at rows 40-89, cols 50-109.
Runs :func:`skin_detector.skin_stats` and prints the skin fraction, the
skin pixel count, and the bounding box of the largest skin region, which
should match the rectangle exactly.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python skin_detector/demo.py`):
# make the repo root importable so `skin_detector` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skin_detector.skin import skin_stats

HEIGHT, WIDTH = 120, 160
SKIN_COLOR = (210, 170, 140)
RECT = (40, 50, 89, 109)  # row_min, col_min, row_max, col_max (inclusive)


def make_scene():
    """Return the synthetic scene described above."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    image[:, :] = (20, 40, 120)  # dark-blue background
    rows, cols = np.indices((HEIGHT, WIDTH))
    circle = (rows - 25) ** 2 + (cols - 130) ** 2 <= 15 ** 2
    image[circle] = (30, 160, 60)  # green circle (not skin)
    r_min, c_min, r_max, c_max = RECT
    image[r_min:r_max + 1, c_min:c_max + 1] = SKIN_COLOR
    return image


def main():
    image = make_scene()
    stats = skin_stats(image)
    expected_fraction = (RECT[2] - RECT[0] + 1) * (RECT[3] - RECT[1] + 1) \
        / (HEIGHT * WIDTH)
    print(f"scene: {HEIGHT}x{WIDTH}, skin rectangle rows {RECT[0]}-{RECT[2]}, "
          f"cols {RECT[1]}-{RECT[3]}")
    print(f"skin fraction: {stats['skin_fraction']:.4f} "
          f"(expected {expected_fraction:.4f})")
    print(f"skin pixels:   {stats['skin_pixels']}")
    print(f"largest region: {stats['largest_pixels']} pixels, "
          f"bbox={stats['largest_bbox']} (expected {RECT})")
    print("bbox correct:", stats["largest_bbox"] == RECT)
    return stats


if __name__ == "__main__":
    main()
