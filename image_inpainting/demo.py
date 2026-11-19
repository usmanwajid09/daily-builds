"""Demo: inpaint a hole in a synthetic gradient image.

Builds a 24x24 grayscale gradient (value rises left to right and top
to bottom), knocks out a 6x6 hole in the middle, inpaints it, and
prints the reconstruction error against the known original. A second
run does the same on an RGB version of the image.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from image_inpainting.image_inpainting import inpaint, inpaint_with_info


def make_gradient(size=24, hole=6):
    """Return (original, mask) for a smooth gradient with a square hole."""
    yy, xx = np.mgrid[0:size, 0:size]
    original = 40.0 + 4.0 * xx + 2.0 * yy
    mask = np.zeros((size, size), dtype=bool)
    start = (size - hole) // 2
    mask[start:start + hole, start:start + hole] = True
    return original, mask


def main():
    original, mask = make_gradient()
    filled, iterations, converged = inpaint_with_info(original, mask)
    error = np.abs(filled - original)
    print(f"grayscale {original.shape}, hole {int(mask.sum())} pixels")
    print(f"  inpainted in {iterations} iterations (converged={converged})")
    print(f"  hole MAE={error[mask].mean():.3f}, max error={error[mask].max():.3f}")
    print(f"  known pixels unchanged: {np.array_equal(filled[~mask], original[~mask])}")

    rgb = np.stack([original, original * 0.5 + 30, 200 - original], axis=2)
    rgb_filled = inpaint(rgb, mask)
    rgb_error = np.abs(rgb_filled - rgb)
    print(f"RGB {rgb.shape}: hole MAE={rgb_error[mask].mean():.3f}, "
          f"max error={rgb_error[mask].max():.3f}")


if __name__ == "__main__":
    main()
