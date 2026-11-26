"""Demo: visible + invisible watermarking on a synthetic RGB image.

Builds a seeded 64x64 RGB gradient image with a dark patch and a bright
circle, stamps a white checkerboard watermark into the bottom-right
corner, and embeds an invisible LSB watermark keyed by "daily-builds".
Prints the mean absolute change inside vs outside the watermarked
corner, the PSNR of the invisibly watermarked image (well above 40 dB),
and verification results for the correct key (True) and a wrong key
(False).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from image_watermark.model import (
    add_visible_watermark,
    embed_invisible_watermark,
    psnr,
    verify_invisible_watermark,
)


def make_test_image(height=64, width=64, seed=0):
    """Return a synthetic RGB image: gradient + dark patch + bright circle."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:height, 0:width]
    base = (rows * 2.0 + cols * 1.5) % 256
    image = np.stack([base, (base + 40) % 256, (base + 90) % 256], axis=-1)
    image[8:24, 10:30] = 40.0  # dark patch
    cy, cx = 44, 44
    circle = (rows - cy) ** 2 + (cols - cx) ** 2 <= 10 ** 2
    image[circle] = 220.0  # bright circle
    image = image + rng.normal(0.0, 2.0, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def main():
    image = make_test_image()
    print(f"image: {image.shape[1]}x{image.shape[0]} RGB, "
          f"dtype={image.dtype}")

    # --- Visible watermark -------------------------------------------------
    watermarked = add_visible_watermark(image, alpha=0.3,
                                        position="bottom-right",
                                        block_size=8)
    # Corner region is 4 * block_size = 32 px square, bottom-right.
    corner = watermarked[32:, 32:].astype(float) - image[32:, 32:].astype(float)
    rest = np.concatenate([
        (watermarked[:32].astype(float) - image[:32].astype(float)).ravel(),
        (watermarked[32:, :32].astype(float)
         - image[32:, :32].astype(float)).ravel(),
    ])
    print(f"visible watermark (alpha=0.3, bottom-right, block_size=8):")
    print(f"  mean |change| in watermarked corner: {np.abs(corner).mean():.2f}")
    print(f"  mean |change| elsewhere:              {np.abs(rest).mean():.2f}")

    # --- Invisible watermark ----------------------------------------------
    key = "daily-builds"
    hidden = embed_invisible_watermark(image, key)
    value = psnr(image, hidden)
    print(f"invisible watermark (key={key!r}):")
    print(f"  PSNR vs original: {value:.2f} dB")
    print(f"  max |pixel change|: "
          f"{np.abs(hidden.astype(int) - image.astype(int)).max()}")
    print(f"  verify correct key {key!r}: "
          f"{verify_invisible_watermark(hidden, key)}")
    print(f"  verify wrong key 'wrong-key': "
          f"{verify_invisible_watermark(hidden, 'wrong-key')}")
    print(f"  verify unwatermarked image: "
          f"{verify_invisible_watermark(image, key)}")


if __name__ == "__main__":
    main()
