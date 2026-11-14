"""Demo: colour histograms on synthetic images.

Builds a red-dominant image and a blue-dominant image, compares their
red-channel histograms (and each image with itself) using all three
comparison methods, and backprojects the red image's own histogram
onto both images to show which pixels the model "recognises".
"""

import os
import sys

# Allow running as a script (`python color_histogram/demo.py`): the
# repo root is the script's parent directory, not the script's own.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from color_histogram.hist import backproject, channel_histograms, compare_histograms


def make_red_image(height=40, width=60):
    """Return a red-dominant RGB image (R high, G and B low)."""
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:, :, 0] = 200.0  # red
    image[:, :, 1] = 30.0   # green
    image[:, :, 2] = 30.0   # blue
    return image


def make_blue_image(height=40, width=60):
    """Return a blue-dominant RGB image (B high, R and G low)."""
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:, :, 0] = 30.0
    image[:, :, 1] = 30.0
    image[:, :, 2] = 200.0
    return image


def main():
    red = make_red_image()
    blue = make_blue_image()
    red_hist = channel_histograms(red)[0]    # red channel of red image
    blue_hist = channel_histograms(blue)[0]  # red channel of blue image

    print("red vs itself:")
    for method in ("correlation", "chi_square", "intersection"):
        print(f"  {method:12s} {compare_histograms(red_hist, red_hist, method):.4f}")
    print("red vs blue (red channel):")
    for method in ("correlation", "chi_square", "intersection"):
        print(f"  {method:12s} {compare_histograms(red_hist, blue_hist, method):.4f}")

    back_red = backproject(red, red_hist)
    back_blue = backproject(blue, red_hist)
    print("backprojection of the red histogram:")
    print(f"  on red image:  mean {back_red.mean():.4f}, "
          f"min {back_red.min():.4f}, max {back_red.max():.4f}")
    print(f"  on blue image: mean {back_blue.mean():.4f}, "
          f"min {back_blue.min():.4f}, max {back_blue.max():.4f}")
    return red_hist, blue_hist, back_red, back_blue


if __name__ == "__main__":
    main()
