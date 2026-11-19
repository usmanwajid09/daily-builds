"""Demo: green-screen compositing on synthetic images.

Builds a 60x80 foreground — a red disc on a pure green screen — and a
blue gradient background, composites them, and prints how many pixels
were replaced plus the colours before and after at a few sample points.
"""

import os
import sys

import numpy as np

# Allow running as `python chroma_key/demo.py` from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chroma_key.chroma import composite, replaced_pixel_count


def make_foreground(height=60, width=80):
    """Return a synthetic foreground: red disc on a green screen."""
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:, :, 1] = 255.0  # green screen
    rows, cols = np.mgrid[0:height, 0:width]
    disc = (rows - 30) ** 2 + (cols - 40) ** 2 <= 15 ** 2
    image[disc] = (255.0, 0.0, 0.0)  # red disc in front of the screen
    return image


def make_background(height=60, width=80):
    """Return a synthetic background: a left-to-right blue gradient."""
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:, :, 0] = 30.0
    image[:, :, 1] = 60.0
    image[:, :, 2] = np.linspace(80.0, 255.0, width)[None, :]
    return image


def main():
    foreground = make_foreground()
    background = make_background()
    replaced = replaced_pixel_count(foreground, tolerance=60.0)
    result = composite(foreground, background, tolerance=60.0)
    total = foreground.shape[0] * foreground.shape[1]
    print(f"foreground: 60x80 RGB, red disc on a green screen")
    print(f"background: 60x80 RGB, blue gradient")
    print(f"screen pixels replaced: {replaced} of {total} "
          f"({100.0 * replaced / total:.1f}%)")
    print(f"centre of disc (kept foreground): {result[30, 40]}")
    print(f"corner (background shows through): {result[0, 0]}")
    print(f"result range: min={result.min():.0f} max={result.max():.0f}")
    return result


if __name__ == "__main__":
    main()
