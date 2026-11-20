"""Demo: chamfer distance transform on a synthetic binary image.

Builds a 40x50 image with a filled disc as foreground, runs the
3-4 and 5-7-11 chamfer transforms, and prints the maximum distance
(the disc centre, ~ the radius) plus the nearest background pixel of
the disc centre.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from distance_transform.distance_transform import (
    distance_transform,
    nearest_background,
)


def make_test_image(height=40, width=50, radius=12):
    """Filled disc (foreground=1) on a zero background."""
    y, x = np.mgrid[0:height, 0:width]
    disc = (y - height // 2) ** 2 + (x - width // 2) ** 2 < radius ** 2
    return disc.astype(np.uint8)


def main():
    binary = make_test_image()
    dist34 = distance_transform(binary, mask="chamfer34")
    dist5711 = distance_transform(binary, mask="chamfer5711")
    centre = (20, 25)
    near_r, near_c = nearest_background(binary, mask="chamfer5711")
    print(f"image: {binary.shape[0]}x{binary.shape[1]} binary, "
          f"foreground pixels = {int(binary.sum())}")
    print(f"chamfer 3-4:   max distance = {dist34.max():.2f} "
          f"(disc centre), centre value = {dist34[centre]:.2f}")
    print(f"chamfer 5-7-11: max distance = {dist5711.max():.2f}, "
          f"centre value = {dist5711[centre]:.2f} (true radius = 12)")
    print(f"nearest background of disc centre {centre}: "
          f"({near_r[centre]}, {near_c[centre]})")
    print(f"background pixels have distance 0: "
          f"{bool(np.all(dist5711[binary == 0] == 0.0))}")
    return dist5711


if __name__ == "__main__":
    main()
