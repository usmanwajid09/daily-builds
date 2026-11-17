"""Demo: count objects in a synthetic binary image.

Builds a 40x50 test image with a fixed seed: three solid blobs
(a 6x4 rectangle, a 5x5 square and a 3x3 square), two diagonal
chains of pixels, and 12 isolated single-pixel specks of noise.
Runs count_objects() with min_size=4 and prints the result.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from object_counter.object_counter import count_objects, label_image

import numpy as np


def build_test_image(seed=42):
    rng = np.random.default_rng(seed)
    img = np.zeros((40, 50), dtype=np.uint8)
    img[2:8, 3:7] = 1        # 6x4 rectangle -> 24 px
    img[12:17, 20:25] = 1    # 5x5 square -> 25 px
    img[28:31, 40:43] = 1    # 3x3 square -> 9 px
    for i in range(6):        # diagonal chain, 8-connected only
        img[20 + i, 30 + i] = 1
    for i in range(5):        # second diagonal chain
        img[33 + i, 5 + i] = 1
    # 12 isolated noise pixels, placed away from the blobs
    placed = 0
    while placed < 12:
        r, c = int(rng.integers(0, 40)), int(rng.integers(0, 50))
        if img[max(0, r - 1):r + 2, max(0, c - 1):c + 2].sum() == 0:
            img[r, c] = 1
            placed += 1
    return img


def main():
    img = build_test_image()
    print(f"Image: {img.shape[0]}x{img.shape[1]}, foreground pixels: {int(img.sum())}")

    labelled = label_image(img, connectivity=8)
    print(f"Raw components (8-connectivity, no filtering): {int(labelled.max())}")

    objects = count_objects(img, min_size=4, connectivity=8)
    print(f"Objects with area >= 4: {len(objects)}")
    for obj in objects:
        r, c = obj["centroid"]
        print(
            f"  label {obj['label']}: area={obj['area']:2d} "
            f"centroid=({r:.1f}, {c:.1f}) bbox={obj['bbox']}"
        )

    # The two diagonal chains touch only diagonally, so 4-connectivity
    # splits each chain pixel into its own object.
    objects_4 = count_objects(img, min_size=4, connectivity=4)
    print(f"Objects with area >= 4 (4-connectivity): {len(objects_4)}")

    # Without size filtering the 12 noise specks show up as objects.
    all_objects = count_objects(img, min_size=1, connectivity=8)
    print(f"Objects with area >= 1: {len(all_objects)} "
          f"(includes {len(all_objects) - len(objects)} noise specks)")


if __name__ == "__main__":
    main()
