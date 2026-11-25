"""Demo: region growing segmentation on a synthetic image.

Builds a 40x60 grayscale image — two bright discs of different
intensity on a dark background — grows a region from a seed inside
the first disc, and prints the region size and a small ASCII rendering
of the mask.
"""

import os
import sys

import numpy as np

# Allow running as `python region_growing/demo.py` from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from region_growing.region_growing import grow_from_seed, region_grow


def make_test_image(height=40, width=60):
    """Two bright discs (200 and 120) on a dark background."""
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width), dtype=np.float64)
    image[(yy - 12) ** 2 + (xx - 15) ** 2 <= 8 ** 2] = 200.0
    image[(yy - 28) ** 2 + (xx - 45) ** 2 <= 6 ** 2] = 120.0
    return image


def main():
    image = make_test_image()
    seed = (12, 15)  # centre of the 200-valued disc
    mask = grow_from_seed(image, seed, tolerance=10.0)
    both = region_grow(image, [(12, 15), (28, 45)], tolerance=10.0)
    disc_area = int(np.pi * 8 ** 2)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, two discs")
    print(f"seed {seed} (value {image[seed]:.0f}), tolerance 10")
    print(f"region size: {int(mask.sum())} pixels (disc area ~{disc_area})")
    print(f"both discs grown together: {int(both.sum())} pixels")
    print(f"seed pixel in mask: {bool(mask[seed])}")
    print(f"other disc excluded: {not mask[28, 45]}")
    # ASCII rendering of the mask, downsampled 2x.
    print("region mask (2x downsampled, # = region):")
    for row in mask[::2]:
        print("".join("#" if v else "." for v in row[::2]))
    return mask


if __name__ == "__main__":
    main()
