"""Demo: gamma correction and brightness/contrast adjustment.

Builds a synthetic 64x64 grayscale gradient (dark on the left,
bright on the right) plus a small RGB test image, then applies gamma
0.5 and 2.2, a log transform, a contrast stretch, and a
brightness/contrast adjustment, printing the mean / min / max after
each step so you can see gamma < 1 brighten, gamma > 1 darken, and
the stretch expand the range to [0, 255].
"""

import sys
from pathlib import Path

import numpy as np

# Allow running as `python gamma_correction/demo.py` from the repo root:
# a script's own directory (not the cwd) is put on sys.path, so add
# the repo root so the `gamma_correction` package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gamma_correction.correct import (
    adjust_brightness_contrast,
    contrast_stretch,
    gamma_correct,
    log_transform,
)


def make_gradient(height=64, width=64):
    """Return a 64x64 gradient: 0 on the left, 255 on the right."""
    row = np.linspace(0, 255, width, dtype=np.float64)
    return np.tile(row, (height, 1))


def make_rgb_test(height=32, width=32, seed=0):
    """Return a small RGB image with different ramps per channel."""
    rng = np.random.default_rng(seed)
    r = np.tile(np.linspace(0, 255, width, dtype=np.float64), (height, 1))
    g = np.tile(np.linspace(255, 0, width, dtype=np.float64), (height, 1))
    b = rng.uniform(0, 255, size=(height, width))
    return np.stack([r, g, b], axis=2)


def main():
    image = make_gradient()
    rgb = make_rgb_test()
    print(f"gradient: {image.shape[0]}x{image.shape[1]} grayscale, 0 (left) to 255 (right)")
    print(f"RGB test image: {rgb.shape[0]}x{rgb.shape[1]}x3")
    print(f"input mean: {image.mean():.2f}")
    print(f"after gamma 0.5 (brightens): {gamma_correct(image, 0.5).mean():.2f}")
    print(f"after gamma 2.2 (darkens): {gamma_correct(image, 2.2).mean():.2f}")
    print(f"after log transform: {log_transform(image).mean():.2f}")
    stretched = contrast_stretch(image)
    print(
        f"after contrast stretch: min={stretched.min():.1f} "
        f"max={stretched.max():.1f} mean={stretched.mean():.2f}"
    )
    adjusted = adjust_brightness_contrast(image, brightness=30.0)
    print(f"after brightness +30: {adjusted.mean():.2f}")
    adjusted2 = adjust_brightness_contrast(image, brightness=-20.0, contrast=1.5)
    print(
        f"after brightness -20, contrast 1.5: mean={adjusted2.mean():.2f} "
        f"min={adjusted2.min():.1f} max={adjusted2.max():.1f}"
    )
    rgb_gamma = gamma_correct(rgb, 0.5)
    print(
        f"RGB after gamma 0.5: shape={rgb_gamma.shape} "
        f"channel means={[round(float(m), 2) for m in rgb_gamma.mean(axis=(0, 1))]}"
    )
    return image


if __name__ == "__main__":
    main()
