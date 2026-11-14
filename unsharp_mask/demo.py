"""Demo: unsharp masking on a synthetic image.

Builds a 64x64 image — a sharp grey square on a black background,
plus slight seeded noise — then blurs and sharpens it, printing the
statistics that show sharpening working: blur lowers the std, the
unsharp mask raises it back (and past the original near edges), edge
pixels get a boost, and a flat region stays unchanged with a
threshold > 0.

Run from the repo root:  python unsharp_mask/demo.py
"""

import os
import sys

import numpy as np

# Running as `python unsharp_mask/demo.py` puts the package dir first on
# sys.path, so add the repo root (this file's parent) to find the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsharp_mask.sharpen import unsharp_mask, gaussian_blur


def main():
    rng = np.random.default_rng(42)

    # Grey square (200) on a black background, plus noise — grey, not
    # white, so the sharpened edge pixel has headroom to rise above it.
    image = np.zeros((64, 64), dtype=np.float64)
    image[16:48, 16:48] = 200.0
    image = image + rng.normal(0.0, 2.0, size=image.shape)
    image = np.clip(image, 0, 255)

    blurred = gaussian_blur(image, sigma=1.0)
    sharpened = unsharp_mask(image, amount=1.0, radius=1.0, threshold=0.0)
    sharpened_thresh = unsharp_mask(image, amount=1.0, radius=1.0, threshold=10.0)

    print(f"Input:      mean={image.mean():.2f}, std={image.std():.2f}")
    print(f"Blurred:    mean={blurred.mean():.2f}, std={blurred.std():.2f}")
    print(f"Sharpened:  mean={sharpened.mean():.2f}, std={sharpened.std():.2f}")

    # Edge pixel: just inside the square's top edge (row 16, col 32).
    # Blurring pulls it down toward black; sharpening pushes it back up.
    edge_before = image[16, 32]
    edge_blurred = blurred[16, 32]
    edge_after = sharpened[16, 32]
    print(
        f"Edge pixel (row 16, col 32): "
        f"input={edge_before:.2f}, blurred={edge_blurred:.2f}, "
        f"sharpened={edge_after:.2f} (boost={edge_after - edge_before:+.2f})"
    )

    # Flat region: the black corner far from the square. Its detail is
    # only noise (~a few intensity units), below threshold=10, so the
    # thresholded sharpening must leave it exactly unchanged.
    flat = image[0:8, 0:8]
    flat_sharp = sharpened_thresh[0:8, 0:8]
    max_change = np.abs(flat_sharp - flat).max()
    print(
        f"Flat region (rows 0-7, cols 0-7) with threshold=10: "
        f"max change={max_change:.6f} "
        f"({'unchanged' if max_change == 0.0 else 'CHANGED!'})"
    )


if __name__ == "__main__":
    main()
