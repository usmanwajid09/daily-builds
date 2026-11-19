"""Demo: high-boost sharpening on a synthetic soft edge.

Builds a 40x60 grayscale image — a vertical step from 50 to 200,
pre-blurred so the edge is soft — sharpens it with k=2 and prints
pixel statistics plus the edge profile before and after, so you can
see the step get steeper and the flat areas stay put.
"""

import os
import sys

import numpy as np

# Allow running as `python high_boost_filter/demo.py` from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from high_boost_filter.filter import gaussian_kernel_1d, high_boost_filter


def _blur_2d(channel, kernel):
    radius = len(kernel) // 2
    height, width = channel.shape
    padded = np.pad(channel, radius, mode="edge")
    tmp = np.zeros_like(channel)
    for i in range(len(kernel)):
        tmp += kernel[i] * padded[radius:radius + height, i:i + width]
    padded = np.pad(tmp, radius, mode="edge")
    out = np.zeros_like(channel)
    for i in range(len(kernel)):
        out += kernel[i] * padded[i:i + height, radius:radius + width]
    return out


def make_test_image(height=40, width=60):
    """Return a synthetic image: a soft vertical step, 50 -> 200."""
    image = np.zeros((height, width), dtype=np.float64)
    image[:, width // 2:] = 200.0
    image[:, :width // 2] = 50.0
    # Pre-blur so the edge is soft (a step a real camera would give).
    return _blur_2d(image, gaussian_kernel_1d(1.5))


def main():
    image = make_test_image()
    sharpened = high_boost_filter(image, k=2.0, sigma=1.0)
    mid = image.shape[1] // 2
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, soft step 50 -> 200")
    print(f"before: min={image.min():.1f} max={image.max():.1f} "
          f"mean={image.mean():.1f} std={image.std():.1f}")
    print(f"after high-boost (k=2.0): min={sharpened.min():.1f} "
          f"max={sharpened.max():.1f} mean={sharpened.mean():.1f} "
          f"std={sharpened.std():.1f}")
    print(f"edge profile before: {image[20, mid - 3:mid + 4].round(1)}")
    print(f"edge profile after:  {sharpened[20, mid - 3:mid + 4].round(1)}")
    step_before = image[20, mid + 2] - image[20, mid - 3]
    step_after = sharpened[20, mid + 2] - sharpened[20, mid - 3]
    print(f"edge step across 5 px: {step_before:.1f} -> {step_after:.1f}")
    return sharpened


if __name__ == "__main__":
    main()
