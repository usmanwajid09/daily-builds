"""Demo: perspective warp of a synthetic gradient image.

Builds a 40x40 image (a smooth x-gradient plus a bright square),
computes the homography from the image corners to a trapezoid, warps
the image, and warps it back again to check the round trip.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from perspective_warp.perspective_warp import (
    apply_homography,
    compute_homography,
    warp_perspective,
)


def make_image(height=40, width=40):
    """Return a gradient image with a bright square in the middle."""
    yy, xx = np.mgrid[0:height, 0:width]
    image = 5.0 * xx + 2.0 * yy
    image[15:25, 15:25] = 200.0
    return image


def main():
    image = make_image()
    src = np.array([[0, 0], [39, 0], [39, 39], [0, 39]], dtype=float)
    dst = np.array([[5, 0], [34, 0], [39, 39], [0, 39]], dtype=float)
    homography = compute_homography(src, dst)
    mapped = apply_homography(homography, src)
    print("homography:")
    print(np.array2string(homography, precision=4, suppress_small=True))
    print(f"corner mapping max error: {np.abs(mapped - dst).max():.2e}")
    warped = warp_perspective(image, homography, (40, 40))
    print(f"warped shape: {warped.shape}, min={warped.min():.1f}, max={warped.max():.1f}")
    print(f"warped centre pixel (20, 20): {warped[20, 20]:.1f}")
    print(f"warped top-left corner (0, 0): {warped[0, 0]:.1f} (outside source -> fill 0)")
    back = warp_perspective(warped, np.linalg.inv(homography), (40, 40))
    interior = np.abs(back[5:35, 5:35] - image[5:35, 5:35])
    print(f"round-trip mean abs error (interior): {interior.mean():.3f}")


if __name__ == "__main__":
    main()
