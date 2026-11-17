"""Demo: perceptual-hash image similarity on synthetic images.

Builds a small set of synthetic grayscale images — a gradient, a
near-duplicate of it with light noise, a horizontally flipped copy,
pure random noise and a solid image — then prints their average and
difference hashes, the Hamming distances to the gradient, and which
image in the set is most similar to the gradient.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_similarity.image_similarity import (
    average_hash,
    difference_hash,
    find_most_similar,
    hamming_distance,
)
import numpy as np


def main():
    rng = np.random.default_rng(42)

    # Base image: smooth left-to-right gradient, 32x32.
    base = np.tile(np.linspace(0, 255, 32), (32, 1))

    # Near-duplicate: same gradient plus tiny noise (clipped to 0-255).
    near_dup = np.clip(base + rng.normal(0, 3, base.shape), 0, 255)

    # Flipped: same gradient mirrored left-to-right.
    flipped = base[:, ::-1].copy()

    # Random noise image and a solid grey image.
    noise = rng.uniform(0, 255, base.shape)
    solid = np.full((32, 32), 128.0)

    images = [near_dup, flipped, noise, solid]
    names = ["near-duplicate", "flipped", "random noise", "solid grey"]

    print("Perceptual-hash image similarity (64-bit hashes)")
    print(f"query: gradient image, aHash = {average_hash(base):#018x}, "
          f"dHash = {difference_hash(base):#018x}")
    print()
    print(f"{'candidate':<16} {'aHash distance':>14} {'dHash distance':>14}")
    for name, img in zip(names, images):
        print(f"{name:<16} {hamming_distance(average_hash(base), average_hash(img)):>14} "
              f"{hamming_distance(difference_hash(base), difference_hash(img)):>14}")
    print()

    index, distance = find_most_similar(base, images, method="average")
    print(f"Most similar (aHash): {names[index]} at index {index}, "
          f"Hamming distance {distance} / 64")
    index, distance = find_most_similar(base, images, method="difference")
    print(f"Most similar (dHash): {names[index]} at index {index}, "
          f"Hamming distance {distance} / 64")
    print()
    print("Identical image check: distance of the gradient to itself is "
          f"{hamming_distance(average_hash(base), average_hash(base))}")


if __name__ == "__main__":
    main()
