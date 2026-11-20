"""Demo: histogram matching between two synthetic images.

Builds a dark, low-contrast source image and a bright, high-contrast
reference image, matches the source's histogram to the reference's,
and prints the mean / std of all three so the transfer is visible.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from histogram_matching.histogram_matching import match_histogram


def make_images(seed=0):
    """Dark low-contrast source, bright high-contrast reference."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:48, 0:64]
    source = 60.0 + 30.0 * np.sin(x / 6.0) * np.cos(y / 5.0)
    source += rng.normal(0.0, 8.0, size=source.shape)
    reference = 160.0 + 90.0 * np.sin(x / 4.0 + 1.0) * np.cos(y / 7.0)
    reference += rng.normal(0.0, 15.0, size=reference.shape)
    return (np.clip(source, 0, 255).astype(np.uint8),
            np.clip(reference, 0, 255).astype(np.uint8))


def main():
    source, reference = make_images()
    matched = match_histogram(source, reference)
    print(f"source:    mean={source.mean():6.2f} std={source.std():6.2f} "
          f"range=[{source.min()}, {source.max()}]")
    print(f"reference: mean={reference.mean():6.2f} std={reference.std():6.2f} "
          f"range=[{reference.min()}, {reference.max()}]")
    print(f"matched:   mean={matched.mean():6.2f} std={matched.std():6.2f} "
          f"range=[{matched.min()}, {matched.max()}]")
    print(f"mean |matched - reference| stats: "
          f"mean diff = {abs(matched.mean() - reference.mean()):.2f}, "
          f"std diff = {abs(matched.std() - reference.std()):.2f}")
    return matched


if __name__ == "__main__":
    main()
