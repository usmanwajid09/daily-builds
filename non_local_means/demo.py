"""Demo: non-local means denoising on a synthetic step image.

Builds a 40x60 step image (50 on the left, 200 on the right), adds
Gaussian noise (std 15), and denoises it with non-local means. The
PSNR against the clean image is reported before and after, plus the
step height and flat-region noise on each side.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from non_local_means.filter import non_local_means, psnr


def make_step_image(rows=40, cols=60, seed=0):
    clean = np.zeros((rows, cols), dtype=np.float64)
    clean[:, : cols // 2] = 50.0
    clean[:, cols // 2:] = 200.0
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0, 15.0, size=clean.shape)
    return clean, noisy


def main():
    clean, noisy = make_step_image()
    denoised = non_local_means(noisy, search_radius=5, patch_radius=1, h=15.0)
    print(f"Image size: {clean.shape[0]}x{clean.shape[1]}, "
          f"noise std: 15.0, h: 15.0")
    print(f"PSNR before denoising: {psnr(clean, noisy):.2f} dB")
    print(f"PSNR after denoising:  {psnr(clean, denoised):.2f} dB")
    for name, img in (("noisy", noisy), ("denoised", denoised)):
        left_std = np.std(img[:, :20])
        right_std = np.std(img[:, 40:])
        step = np.mean(img[:, 40:]) - np.mean(img[:, :20])
        print(f"{name:>9}: flat-region std left={left_std:.2f} "
              f"right={right_std:.2f}, step height={step:.1f} (true 150.0)")


if __name__ == "__main__":
    main()
