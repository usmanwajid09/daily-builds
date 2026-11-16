"""Demo: build a Laplacian pyramid, rebuild the image, and blend two images.

Builds a synthetic 64x64 image (a horizontal gradient, a bright disc and
a dark square), splits it into a 4-level Laplacian pyramid, reconstructs
it, and prints the level shapes plus the reconstruction error (which
should be ~0 — the pyramid is lossless by construction). Then it blends
two images with a half/half mask and prints the result's shape and mean.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from laplacian_pyramid.pyramid import (
    blend_images,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct,
)


def make_test_image(size=64):
    """Synthetic grayscale image: gradient + bright disc + dark square."""
    y, x = np.mgrid[0:size, 0:size]
    image = (x / size * 200.0).astype(np.float64)  # horizontal gradient
    disc = (x - size // 3) ** 2 + (y - size // 3) ** 2 < (size // 6) ** 2
    image[disc] = 255.0
    image[size // 2:size * 3 // 4, size // 2:size * 3 // 4] = 20.0
    return image


def main():
    image = make_test_image()
    levels = 4

    laplacian = build_laplacian_pyramid(image, levels)
    gaussian = build_gaussian_pyramid(image, levels)
    print("Gaussian pyramid level shapes:",
          [level.shape for level in gaussian])
    print("Laplacian pyramid level shapes:",
          [level.shape for level in laplacian])

    rebuilt = reconstruct(laplacian)
    error = np.abs(rebuilt - image)
    mse = float(np.mean((rebuilt - image) ** 2))
    print(f"Reconstructed shape: {rebuilt.shape}")
    print(f"Max abs reconstruction error: {error.max():.3e}")
    print(f"Mean squared reconstruction error: {mse:.3e}")

    # Blend demo: bright disc image vs dark gradient, split left/right.
    image_a = make_test_image()
    image_b = np.flipud(make_test_image()) * 0.4
    mask = np.zeros((64, 64))
    mask[:, :32] = 1.0  # left half from A, right half from B
    blended = blend_images(image_a, image_b, mask, levels=4)
    print(f"Blended image shape: {blended.shape}")
    print(f"Blended image mean: {blended.mean():.2f}")
    print(f"Blended image range: [{blended.min():.2f}, {blended.max():.2f}]")


if __name__ == "__main__":
    main()
