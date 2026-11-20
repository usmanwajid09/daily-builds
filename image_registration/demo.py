"""Demo: translation registration via phase correlation.

Builds a synthetic 64x64 white-noise pattern, shifts it by a known
integer amount (dy=+5, dx=-7) and a fractional amount (dy=+2.5,
dx=-3.5), then prints the true shift, the estimated shift and the
error for both the integer and the subpixel estimator.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from image_registration.image_registration import (
    apply_shift,
    phase_correlation,
    register_translation,
)


def make_pattern(height=64, width=64, seed=0):
    """White-noise pattern: a full spectrum, ideal for phase correlation."""
    rng = np.random.default_rng(seed)
    pattern = rng.normal(0.0, 1.0, size=(height, width))
    return pattern - pattern.mean()


def main():
    reference = make_pattern()
    true_shift = (5, -7)
    moving = apply_shift(reference, *true_shift)
    shift, peak = phase_correlation(reference, moving)
    result = register_translation(reference, moving)
    error = np.hypot(result["shift"][0] - true_shift[0],
                     result["shift"][1] - true_shift[1])
    print(f"image: {reference.shape[0]}x{reference.shape[1]} grayscale")
    print(f"integer shift: true={true_shift}, "
          f"estimated={result['shift']}, error={error:.3f} px, "
          f"peak={result['peak']:.3f}")
    true_sub = (2.5, -3.5)
    moving_sub = apply_shift(reference, *true_sub)
    result_sub = register_translation(reference, moving_sub, subpixel=True)
    error_sub = np.hypot(result_sub["shift"][0] - true_sub[0],
                         result_sub["shift"][1] - true_sub[1])
    print(f"subpixel shift: true={true_sub}, "
          f"estimated=({result_sub['shift'][0]:.2f}, "
          f"{result_sub['shift'][1]:.2f}), error={error_sub:.3f} px, "
          f"peak={result_sub['peak']:.3f}")
    print(f"registered image matches reference: max abs diff = "
          f"{np.abs(result['registered'] - reference).max():.2e}")
    return result


if __name__ == "__main__":
    main()
