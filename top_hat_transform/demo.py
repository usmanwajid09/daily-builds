"""Demo: top-hat transforms on a synthetic image.

Builds a 30x40 grayscale image — a slowly varying background gradient
with a small bright spot and a small dark spot — and applies the white
and black top-hat transforms. The top-hats pull the two small spots
out of the background while the background itself (which opening and
closing leave alone) cancels to near zero.
"""

import os
import sys

import numpy as np

# Allow running as `python top_hat_transform/demo.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from top_hat_transform.tophat import black_tophat, top_hat, white_tophat


def make_test_image(height=30, width=40):
    """Background gradient + one small bright spot + one small dark spot."""
    yy, xx = np.mgrid[0:height, 0:width]
    image = 100.0 + 0.5 * xx + 0.3 * yy
    image[12:16, 10:14] = 220.0   # bright spot (4x4)
    image[20:24, 28:32] = 20.0    # dark spot (4x4)
    return image


def main():
    image = make_test_image()
    white, black = top_hat(image, kernel_size=7, shape="square")
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, gradient + "
          f"bright spot + dark spot")
    print(f"white top-hat: max={white.max():.1f} at bright spot "
          f"(centre value {white[14, 12]:.1f}), mean={white.mean():.2f}")
    print(f"black top-hat: max={black.max():.1f} at dark spot "
          f"(centre value {black[22, 30]:.1f}), mean={black.mean():.2f}")
    print(f"white top-hat away from spots (row 5): max={white[5].max():.2f} "
          f"(background cancels out)")
    print(f"black top-hat away from spots (row 5): max={black[5].max():.2f}")
    # Disk element, for comparison.
    white_disk = white_tophat(image, kernel_size=7, shape="disk")
    black_disk = black_tophat(image, kernel_size=7, shape="disk")
    print(f"disk element: white max={white_disk.max():.1f}, "
          f"black max={black_disk.max():.1f}")
    return white, black


if __name__ == "__main__":
    main()
