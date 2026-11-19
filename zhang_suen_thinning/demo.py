"""Demo: Zhang-Suen thinning on a synthetic thick plus sign.

Builds a 20x30 binary image — a thick vertical bar crossed by a thick
horizontal bar — thins it to a 1-pixel-wide skeleton and prints the
pixel counts before and after plus an ASCII rendering of both.
"""

import os
import sys

import numpy as np

# Allow running as `python zhang_suen_thinning/demo.py` from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zhang_suen_thinning.thinning import skeleton_pixel_count, zhang_suen_thin


def make_test_image(height=20, width=30):
    """Return a binary image: a thick plus sign."""
    image = np.zeros((height, width), dtype=bool)
    image[3:17, 13:17] = True   # vertical bar, 4 px wide
    image[9:13, 5:25] = True    # horizontal bar, 4 px wide
    return image


def render_ascii(binary, step=1):
    """Render a bool image as ASCII art ('#' = foreground)."""
    return "\n".join(
        "".join("#" if pixel else " " for pixel in row[::step])
        for row in binary[::step]
    )


def main():
    image = make_test_image()
    skeleton = zhang_suen_thin(image)
    before = skeleton_pixel_count(image)
    after = skeleton_pixel_count(skeleton)
    print(f"image: {image.shape[0]}x{image.shape[1]} binary, thick plus sign")
    print(f"foreground pixels: {before} -> {after} after thinning "
          f"({100.0 * after / before:.1f}% kept)")
    print("before:")
    print(render_ascii(image))
    print("after (skeleton):")
    print(render_ascii(skeleton))
    return skeleton


if __name__ == "__main__":
    main()
