"""Demo: colour-space conversions on a synthetic RGB image.

Builds a small synthetic RGB image (red/green/blue/white patches plus a
smooth gradient), converts it to HSV, YCbCr and grayscale, converts
back, and prints the round-trip error for each conversion — the proof
that the conversions are correct.
"""

import os
import sys

import numpy as np

# Allow running as `python color_space_converter/demo.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color_space_converter.color_space import (
    grayscale_to_rgb,
    hsv_to_rgb,
    rgb_to_grayscale,
    rgb_to_hsv,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


def make_test_image(height=8, width=12):
    """Colour patches (red, green, blue, white) plus a smooth gradient."""
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[0:4, 0:3] = [255, 0, 0]      # red
    image[0:4, 3:6] = [0, 255, 0]      # green
    image[0:4, 6:9] = [0, 0, 255]      # blue
    image[0:4, 9:12] = [255, 255, 255]  # white
    ramp = np.linspace(0, 255, width)
    image[4:8, :, 0] = ramp             # red ramp
    image[4:8, :, 1] = ramp[::-1]       # green ramp (reversed)
    image[4:8, :, 2] = 128.0            # constant blue
    return image


def main():
    image = make_test_image()
    hsv = rgb_to_hsv(image)
    rgb_back = hsv_to_rgb(hsv)
    hsv_error = np.abs(rgb_back - image).max()
    ycbcr = rgb_to_ycbcr(image)
    rgb_back2 = ycbcr_to_rgb(ycbcr)
    ycbcr_error = np.abs(rgb_back2 - image).max()
    gray = rgb_to_grayscale(image)
    rgb_gray = grayscale_to_rgb(gray)
    print(f"image: {image.shape[0]}x{image.shape[1]} RGB, patches + gradient")
    print(f"pure red   -> HSV: H={hsv[0, 0, 0]:.1f} S={hsv[0, 0, 1]:.2f} "
          f"V={hsv[0, 0, 2]:.2f} (expect H=0, S=1, V=1)")
    print(f"pure green -> HSV: H={hsv[0, 3, 0]:.1f} S={hsv[0, 3, 1]:.2f} "
          f"V={hsv[0, 3, 2]:.2f} (expect H=120, S=1, V=1)")
    print(f"pure blue  -> HSV: H={hsv[0, 6, 0]:.1f} S={hsv[0, 6, 1]:.2f} "
          f"V={hsv[0, 6, 2]:.2f} (expect H=240, S=1, V=1)")
    print(f"white      -> YCbCr: Y={ycbcr[0, 9, 0]:.1f} Cb={ycbcr[0, 9, 1]:.1f} "
          f"Cr={ycbcr[0, 9, 2]:.1f} (expect Y=255, Cb=128, Cr=128)")
    print(f"red luma: {gray[0, 0]:.1f} (expect 76.2), green luma: "
          f"{gray[0, 3]:.1f} (expect 149.7), blue luma: {gray[0, 6]:.1f} "
          f"(expect 29.1)")
    print(f"RGB -> HSV -> RGB max round-trip error: {hsv_error:.2e}")
    print(f"RGB -> YCbCr -> RGB max round-trip error: {ycbcr_error:.2e}")
    print(f"grayscale -> RGB channels equal: "
          f"{bool((rgb_gray[..., 0] == gray).all() and (rgb_gray[..., 1] == gray).all())}")
    return hsv, ycbcr, gray


if __name__ == "__main__":
    main()
