"""Demo: CLAHE on a synthetic low-contrast grayscale image.

Builds a 64x96 image whose values only span roughly 90-150: a smooth
gradient, a dark patch, a bright patch, and a little noise. CLAHE is run
on it and the min/max/mean/std before and after are printed, along with
the per-region contrast (std) inside the dark and bright patches, so you
can see local contrast improve without the whole range just being
stretched.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clahe.clahe import apply_clahe


def make_test_image(height=64, width=96, seed=0):
    """Return a synthetic low-contrast image (values ~90-150)."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:height, 0:width]
    image = 110.0 + 0.3 * rows + 0.2 * cols
    image[8:24, 10:30] = 95.0 + 0.4 * rows[8:24, 10:30]  # dark patch
    image[36:56, 60:88] = 140.0 + 0.3 * cols[36:56, 60:88]  # bright patch
    image += rng.normal(0.0, 2.0, size=image.shape)
    return np.clip(image, 0, 255)


def _stats(image):
    return (
        f"min={image.min():3.0f} max={image.max():3.0f} "
        f"mean={image.mean():6.1f} std={image.std():5.1f}"
    )


def main():
    image = make_test_image()
    enhanced = apply_clahe(image)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, low contrast")
    print(f"before: {_stats(image)}")
    print(f"after:  {_stats(enhanced)}")
    print("per-region contrast (std inside each patch):")
    for name, sl in (
        ("dark patch  [8:24, 10:30]", (slice(8, 24), slice(10, 30))),
        ("bright patch [36:56, 60:88]", (slice(36, 56), slice(60, 88))),
        ("whole image", (slice(None), slice(None))),
    ):
        before = image[sl].std()
        after = enhanced[sl].astype(float).std()
        print(f"  {name}: {before:5.1f} -> {after:5.1f} (x{after / before:.2f})")
    return enhanced


if __name__ == "__main__":
    main()
