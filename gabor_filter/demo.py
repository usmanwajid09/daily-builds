"""Demo: Gabor filter bank on a synthetic striped texture image.

Builds a 64x64 image with vertical stripes on the left half and
horizontal stripes on the right half, filters it with a bank of 4
orientations x 2 scales, and prints the mean response energy per
filter plus the local texture energy of the strongest filter.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gabor_filter.gabor_filter import (
    apply_gabor_bank,
    gabor_bank,
    texture_energy,
)


def make_test_image(height=64, width=64):
    """Vertical stripes left, horizontal stripes right, plus noise."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width))
    image[:, :width // 2] = 128.0 + 100.0 * np.sin(2 * np.pi * x[:, :width // 2] / 8.0)
    image[:, width // 2:] = 128.0 + 100.0 * np.sin(2 * np.pi * y[:, width // 2:] / 8.0)
    return image + rng.normal(0.0, 5.0, size=image.shape)


def main():
    image = make_test_image()
    kernels = gabor_bank(n_orientations=4, wavelengths=(8.0, 16.0),
                         ksize=21, sigma=4.0)
    responses = apply_gabor_bank(image, kernels)
    energy = texture_energy(responses, window=9)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale")
    print(f"bank: {len(kernels)} filters "
          f"(4 orientations x 2 scales), response shape {responses.shape}")
    thetas = [i * 180 / 4 for i in range(4)]
    for scale in range(2):
        for i, theta in enumerate(thetas):
            k = scale * 4 + i
            print(f"  wavelength={8.0 * 2 ** scale:4.0f} theta={theta:5.0f}deg "
                  f"mean |response| = {np.abs(responses[:, :, k]).mean():7.2f}")
    # A 0-degree filter (vertical stripes) should fire on the left half.
    left = energy[:, :32, 0].mean()
    right = energy[:, 32:, 0].mean()
    print(f"theta=0deg energy: left (vertical stripes) = {left:.2f}, "
          f"right (horizontal stripes) = {right:.2f}")
    print(f"strongest filter index: {int(np.argmax(np.abs(responses).mean(axis=(0, 1))))}")
    return responses


if __name__ == "__main__":
    main()
