"""Demo: privacy-blur the skin region of a synthetic image.

Builds a 120x160 scene: a dark-blue background, a green circle, and a
skin-coloured rectangle (RGB 210, 170, 140) at rows 40-89, cols 50-109.
Runs :func:`privacy_blur.privacy_blur` and prints the skin fraction,
the skin pixel count, and a check that blurred pixels differ from the
original inside the rectangle while pixels outside it are unchanged.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python privacy_blur/demo.py`):
# make the repo root importable so `privacy_blur` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy_blur.privacy import blur_stats, privacy_blur

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
    blurred, mask = privacy_blur(image)
    stats = blur_stats(image)
    r_min, c_min, r_max, c_max = RECT
    rect = (slice(r_min, r_max + 1), slice(c_min, c_max + 1))
    inside_changed = bool(np.any(blurred[rect] != image[rect]))
    outside = np.ones_like(mask, dtype=bool)
    outside[rect] = False
    outside_unchanged = bool(np.array_equal(blurred[outside], image[outside]))
    expected_fraction = (RECT[2] - RECT[0] + 1) * (RECT[3] - RECT[1] + 1) \
        / (HEIGHT * WIDTH)
    print(f"scene: {HEIGHT}x{WIDTH}, skin rectangle rows {RECT[0]}-{RECT[2]}, "
          f"cols {RECT[1]}-{RECT[3]}")
    print(f"skin fraction: {stats['skin_fraction']:.4f} "
          f"(expected {expected_fraction:.4f})")
    print(f"skin pixels:   {stats['skin_pixels']}")
    print(f"blurred pixels (value changed): {stats['blurred_pixels']}")
    print(f"inside rectangle changed: {inside_changed}")
    print(f"outside rectangle unchanged: {outside_unchanged}")
    return stats


if __name__ == "__main__":
    main()
