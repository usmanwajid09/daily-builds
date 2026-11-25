"""Demo: guided filter edge-preserving smoothing on a synthetic image.

Builds a 40x60 grayscale image — a bright step edge with Gaussian
noise (fixed seed) — smooths it with the guided filter, and prints
the noise level in a flat region before/after plus the step height
before/after, showing noise reduced while the edge survives.
"""

import os
import sys

import numpy as np

# Allow running as `python guided_filter/demo.py` from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_filter.guided_filter import edge_preserving_smooth, guided_filter


def make_test_image(height=40, width=60, seed=0):
    """Step edge (0 -> 100) plus Gaussian noise, std 10."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.float64)
    image[:, width // 2:] = 100.0
    image += rng.normal(0.0, 10.0, size=image.shape)
    return image


def main():
    image = make_test_image()
    smoothed = edge_preserving_smooth(image, radius=4, eps=1000.0)
    # Flat region: top-left corner, away from the edge.
    flat_before = image[:15, :15].std()
    flat_after = smoothed[:15, :15].std()
    # Step height: mean of the right half minus mean of the left half.
    step_before = image[:, 40:].mean() - image[:, :20].mean()
    step_after = smoothed[:, 40:].mean() - smoothed[:, :20].mean()
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, noisy step edge")
    print(f"flat-region noise std: {flat_before:.2f} -> {flat_after:.2f}")
    print(f"noise reduction: {100.0 * (1 - flat_after / flat_before):.1f}%")
    print(f"step height: {step_before:.2f} -> {step_after:.2f} "
          f"({100.0 * step_after / step_before:.1f}% preserved)")
    print(f"output range: [{smoothed.min():.2f}, {smoothed.max():.2f}]")
    return smoothed


if __name__ == "__main__":
    main()
