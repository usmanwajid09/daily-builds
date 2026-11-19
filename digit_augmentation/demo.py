"""Demo: digit image augmentation on a synthetic 28x28 "1".

Builds a synthetic digit-like image — a bright vertical bar (cols
13-14, rows 4-23) with a small diagonal stroke at the top — then runs
:func:`digit_augmentation.augment_n` with n=5, seed=42, and prints
each copy's pixel sum and whether it differs from the original, plus
one example each of rotation, flips, translation and noise.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python digit_augmentation/demo.py`):
# make the repo root importable so `digit_augmentation` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digit_augmentation.augmentation import (
    add_gaussian_noise,
    augment_n,
    flip_horizontal,
    flip_vertical,
    rotate90,
    translate,
)

SIZE = 28


def make_digit():
    """Return a synthetic 28x28 digit "1" (values 0/1 floats)."""
    image = np.zeros((SIZE, SIZE), dtype=np.float64)
    image[4:24, 13:15] = 1.0  # vertical bar, cols 13-14, rows 4-23
    for i in range(6):         # small diagonal, rows 4-9, cols 8-13
        image[4 + i, 8 + i] = 1.0
    return image


def main():
    image = make_digit()
    print(f"original: shape={image.shape}, sum={image.sum():.1f}")

    copies = augment_n(image, n=5, seed=42)
    for i, copy in enumerate(copies):
        print(f"augmented {i}: shape={copy.shape}, "
              f"sum={copy.sum():.2f}, "
              f"differs from original: {not np.array_equal(copy, image)}")

    print(f"rotate90 k=1:       sum={rotate90(image, k=1).sum():.1f}")
    print(f"flip_horizontal:    sum={flip_horizontal(image).sum():.1f}")
    print(f"flip_vertical:      sum={flip_vertical(image).sum():.1f}")
    print(f"translate (2, 3):   sum={translate(image, dy=2, dx=3).sum():.1f}")
    print(f"noise std=0.1 s=7:  "
          f"sum={add_gaussian_noise(image, std=0.1, seed=7).sum():.2f}")
    return copies


if __name__ == "__main__":
    main()
