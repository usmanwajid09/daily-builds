"""Demo: Haar wavelet denoising of a noisy synthetic image.

Run from the repo root:  python wavelet_denoising/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from wavelet_denoising import psnr, wavelet_denoise


def build_image(height=64, width=64):
    """Piecewise-constant synthetic image: blocks, a disc, a gradient."""
    img = np.zeros((height, width), dtype=np.float64)
    img[:, :] = 60.0
    img[8:28, 8:28] = 180.0
    img[36:56, 36:56] = 220.0
    yy, xx = np.mgrid[0:height, 0:width]
    disc = (yy - 48) ** 2 + (xx - 16) ** 2 <= 8 ** 2
    img[disc] = 140.0
    img += np.linspace(0, 40, width)[None, :]
    return np.clip(img, 0, 255)


def main():
    clean = build_image()
    rng = np.random.default_rng(0)
    noisy = np.clip(clean + rng.normal(0, 20, size=clean.shape), 0, 255)
    denoised_soft = wavelet_denoise(noisy, levels=2, mode="soft")
    denoised_hard = wavelet_denoise(noisy, levels=2, mode="hard")

    print(f"Image shape: {clean.shape}")
    print(f"PSNR noisy vs clean:          {psnr(clean, noisy):.2f} dB")
    print(f"PSNR soft-thresholded:        {psnr(clean, denoised_soft):.2f} dB")
    print(f"PSNR hard-thresholded:        {psnr(clean, denoised_hard):.2f} dB")
    print(f"Denoised (soft) dtype: {denoised_soft.dtype}, "
          f"range [{denoised_soft.min()}, {denoised_soft.max()}]")


if __name__ == "__main__":
    main()
