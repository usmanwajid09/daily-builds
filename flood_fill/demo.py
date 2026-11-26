"""Demo: flood-fill segmentation of a synthetic image.

Builds a 60x80 image (background 0, a disc of value 100 and a
separate rectangle of value 180), flood-fills from a seed inside the
disc with a small tolerance, and prints the filled pixel count next
to the disc's true pixel area. Also shows :func:`fill_holes` on a
small ring mask.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from flood_fill.model import fill_holes, flood_fill


def make_image(height=60, width=80):
    """Return (image, disc mask) for the synthetic scene."""
    rows, cols = np.mgrid[0:height, 0:width]
    disc = (rows - 30) ** 2 + (cols - 25) ** 2 <= 14 ** 2
    image = np.zeros((height, width), dtype=np.uint8)
    image[disc] = 100
    image[10:25, 55:75] = 180
    return image, disc


def make_ring(size=9):
    """Return a ring mask: True border, False 3x3 hole in the middle."""
    ring = np.zeros((size, size), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return ring


def main():
    image, disc = make_image()
    seed = (30, 25)
    mask = flood_fill(image, seed, tolerance=5.0, connectivity=4)
    filled = int(mask.sum())
    disc_area = int(disc.sum())
    print(f"image: {image.shape[0]}x{image.shape[1]}, "
          f"disc value 100, rectangle value 180")
    print(f"seed {seed} (value {image[seed]}), tolerance 5.0, "
          f"connectivity 4")
    print(f"filled pixels: {filled}")
    print(f"disc true area: {disc_area} pixels")
    print(f"match: {filled == disc_area} "
          f"(difference: {abs(filled - disc_area)})")

    ring = make_ring()
    filled_ring = fill_holes(ring)
    print(f"ring mask: {ring.shape[0]}x{ring.shape[1]}, "
          f"foreground before fill_holes: {int(ring.sum())}, "
          f"after: {int(filled_ring.sum())} "
          f"(hole of {int(filled_ring.sum()) - int(ring.sum())} pixels filled)")
    return mask


if __name__ == "__main__":
    main()
