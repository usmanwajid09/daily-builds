"""Demo: adaptive vs global thresholding under uneven illumination.

Builds a 60x80 document-like image: dark bars ("ink") on a smooth
left-to-right illumination ramp (100 on the left, 220 on the right),
then prints, for each method, how many bar pixels are correctly
found as ink and how many background pixels are correctly kept as
paper.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from adaptive_threshold.adaptive_threshold import (
    adaptive_threshold,
    global_threshold,
)


def bars_mask(height=60, width=80):
    """Boolean mask of the three dark bars."""
    bars = np.zeros((height, width), dtype=bool)
    for cx in (10, 35, 60):
        bars[15:45, cx - 3:cx + 3] = True
    return bars


def make_test_image(height=60, width=80, seed=0):
    """Illumination ramp (100 -> 220) with dark bars at value 40."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    image = 100.0 + 120.0 * (x / width)
    image[bars_mask(height, width)] = 40.0
    return np.clip(image + rng.normal(0.0, 3.0, size=image.shape), 0, 255)


def _score(mask, bars):
    """Fraction of bars found as ink (False) and background kept (True)."""
    ink_found = 1.0 - mask[bars].mean()
    paper_kept = mask[~bars].mean()
    return ink_found, paper_kept


def main():
    image = make_test_image()
    bars = bars_mask()
    adapt = adaptive_threshold(image, block_size=15, c=10.0, method="mean")
    glob = global_threshold(image)
    ink_a, paper_a = _score(adapt, bars)
    ink_g, paper_g = _score(glob, bars)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, "
          f"mean={image.mean():.1f}, bars = {int(bars.sum())} px")
    print(f"adaptive: ink found = {ink_a:.2%}, paper kept = {paper_a:.2%}")
    print(f"global:   ink found = {ink_g:.2%}, paper kept = {paper_g:.2%}")
    print("global threshold calls the whole dark left side 'ink'; "
          "adaptive threshold finds the bars on both sides.")
    return {"adaptive": adapt, "global": glob}


if __name__ == "__main__":
    main()
