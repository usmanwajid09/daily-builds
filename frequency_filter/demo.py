"""Demo: FFT frequency filtering on a synthetic image.

Builds a 128x128 image that mixes a smooth gradient (low frequency)
with a fine checkerboard (high frequency) plus noise, then shows what
Gaussian low-pass and high-pass filtering do to it in numbers.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from frequency_filter.filter import high_pass, low_pass


def make_synthetic(size=128, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    # Smooth low-frequency part: a diagonal gradient.
    smooth = 0.5 * (x + y) / size
    # Fine high-frequency part: a checkerboard, one pixel per square.
    checker = 0.4 * ((x + y) % 2) * 2 - 0.4
    noise = rng.normal(0.0, 0.05, size=(size, size))
    return smooth + checker + noise


def main():
    image = make_synthetic()
    cutoff = 8.0

    lp = low_pass(image, cutoff, kind="gaussian")
    hp = high_pass(image, cutoff, kind="gaussian")

    # Energy kept by the low-pass, via Parseval (sum of squares in the
    # pixel domain equals the energy in the frequency domain).
    energy_orig = float(np.sum(image ** 2))
    energy_lp = float(np.sum(lp ** 2))
    energy_hp = float(np.sum(hp ** 2))

    print(f"image shape:            {image.shape}")
    print(f"original std:           {image.std():.4f}")
    print(f"original energy:        {energy_orig:.1f}")
    print(f"low-pass std:           {lp.std():.4f}  (lower: fine detail removed)")
    print(f"low-pass energy kept:   {energy_lp / energy_orig:.3f}")
    print(f"high-pass mean:         {hp.mean():+.5f}  (near 0: DC / smooth part removed)")
    print(f"high-pass std:          {hp.std():.4f}")
    print(f"high-pass energy kept:  {energy_hp / energy_orig:.3f}")


if __name__ == "__main__":
    main()
