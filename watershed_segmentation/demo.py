"""Demo: watershed segmentation of two overlapping discs.

Builds a 60x80 binary mask with two overlapping discs, runs
:func:`watershed_segmentation.watershed.watershed` with automatic
markers and prints how many segments were found, how many watershed
pixels separate them, and how many pixels each segment holds.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from watershed_segmentation.watershed import (
    segment_count,
    watershed,
    watershed_lines,
)


def make_mask(height=60, width=80):
    """Return a mask of two overlapping discs (radius 14)."""
    rows, cols = np.mgrid[0:height, 0:width]
    disc1 = (rows - 30) ** 2 + (cols - 25) ** 2 <= 14 ** 2
    disc2 = (rows - 30) ** 2 + (cols - 45) ** 2 <= 14 ** 2
    return disc1 | disc2


def main():
    mask = make_mask()
    labels = watershed(mask)
    count = segment_count(labels)
    line_pixels = int(watershed_lines(labels).sum())
    print(f"mask: {mask.shape[0]}x{mask.shape[1]}, "
          f"foreground pixels: {int(mask.sum())}")
    print(f"segments found: {count}")
    print(f"watershed line pixels: {line_pixels}")
    for label in sorted(np.unique(labels[labels > 0])):
        print(f"  segment {label}: {int((labels == label).sum())} pixels")
    return labels


if __name__ == "__main__":
    main()
