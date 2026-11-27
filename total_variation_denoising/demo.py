"""Demo: total-variation denoising of a noisy synthetic image.

Run from the repo root:  python total_variation_denoising/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from total_variation_denoising import psnr, tv_denoise


def build_image(height=48, width=64):
    """Piecewise-constant synthetic image: a bright block, a dark disc,
    and a step edge — the kind of image TV denoising preserves well."""
    img = np.full((height, width), 90.0)
    img[8:24, 8:28] = 200.0
    yy, xx = np.mgrid[0:height, 0:width]
    disc = (yy - 36) ** 2 + (xx - 46) ** 2 <= 9 ** 2
    img[disc] = 30.0
    return img


def main():
    clean = build_image()
    rng = np.random.default_rng(0)
    noisy = np.clip(clean + rng.normal(0, 15, size=clean.shape), 0, 255)
    denoised = tv_denoise(noisy, weight=15.0, max_iter=200)

    print(f"Image shape: {clean.shape}")
    print(f"PSNR noisy vs clean:    {psnr(clean, noisy):.2f} dB")
    print(f"PSNR TV-denoised:       {psnr(clean, denoised):.2f} dB")
    print(f"Denoised dtype: {denoised.dtype}, "
          f"range [{denoised.min():.1f}, {denoised.max():.1f}]")
    # Edge check: the step between the block and the background survives.
    print(f"Block mean: {denoised[12:20, 12:24].mean():.1f}, "
          f"background mean: {denoised[30:40, 4:12].mean():.1f}")


if __name__ == "__main__":
    main()
