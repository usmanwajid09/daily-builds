"""Demo: Reinhard colour transfer between two synthetic images.

Builds a warm-toned source image and a cool-toned target image, runs
:func:`color_transfer.transfer.color_transfer` in Lab space and prints
the per-channel mean/std of the source, the target, and the result —
the result's statistics should closely match the target's.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from color_transfer.transfer import color_transfer, rgb_to_lab


def make_gradient(height, width, base, slope):
    """Return an RGB image: per-channel linear gradients + noise."""
    rng = np.random.default_rng(0)
    rows, cols = np.mgrid[0:height, 0:width]
    image = np.empty((height, width, 3), dtype=np.float64)
    for channel in range(3):
        image[..., channel] = (base[channel] + slope[channel] * cols
                               + 0.5 * slope[channel] * rows)
    image += rng.normal(0.0, 5.0, size=image.shape)
    return np.clip(image, 0.0, 255.0)


def _stats(name, image):
    # Statistics are compared in Lab space: that is the space the
    # transfer matches, so these are the numbers that should agree.
    lab = rgb_to_lab(image)
    means = ", ".join(f"{lab[..., c].mean():.1f}" for c in range(3))
    stds = ", ".join(f"{lab[..., c].std():.1f}" for c in range(3))
    print(f"{name:8s} Lab mean: [{means}]  Lab std: [{stds}]")


def main():
    source = make_gradient(40, 60, base=(200.0, 120.0, 60.0),
                           slope=(0.5, 0.3, 0.2))
    target = make_gradient(40, 60, base=(60.0, 110.0, 190.0),
                           slope=(0.2, 0.4, 0.8))
    result = color_transfer(source, target, colorspace="lab")
    _stats("source", source)
    _stats("target", target)
    _stats("result", result)
    return result


if __name__ == "__main__":
    main()
