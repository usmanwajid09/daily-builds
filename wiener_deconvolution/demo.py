"""Demo: Wiener deblurring of a synthetic image.

Builds a sharp 64x64 test image (gradient, bright square, dark
disc), blurs it with a Gaussian PSF (sigma 2), adds a little noise
(sigma 1.5), then restores it with Wiener deconvolution and compares
PSNR before and after. A second, noise-free run shows the near-exact
recovery Wiener filtering achieves when there is no noise to
amplify.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from wiener_deconvolution import (
    apply_blur,
    estimate_nsr,
    make_gaussian_psf,
    psnr,
    wiener_deconvolve,
)


def make_sharp_image(size=64):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    image = 40.0 + 160.0 * xx / size
    image[20:36, 20:36] = 240.0
    disc = (yy - 46) ** 2 + (xx - 46) ** 2 <= 10 ** 2
    image[disc] = 10.0
    return image


def main():
    sharp = make_sharp_image()
    psf = make_gaussian_psf(size=15, sigma=2.0)
    blurred = apply_blur(sharp, psf)
    rng = np.random.default_rng(0)
    noise_sigma = 1.5
    noisy = blurred + rng.normal(0.0, noise_sigma, size=sharp.shape)
    nsr = estimate_nsr(noise_sigma ** 2, float(np.var(sharp)))
    restored = wiener_deconvolve(noisy, psf, nsr=nsr)
    psnr_blurred = psnr(sharp, noisy)
    psnr_restored = psnr(sharp, restored)
    print(f"image: {sharp.shape[0]}x{sharp.shape[1]}, "
          f"gaussian PSF sigma=2.0, added noise sigma={noise_sigma}")
    print(f"estimated NSR: {nsr:.6f}")
    print(f"PSNR blurred+noisy: {psnr_blurred:.2f} dB")
    print(f"PSNR Wiener restored: {psnr_restored:.2f} dB "
          f"(gain {psnr_restored - psnr_blurred:+.2f} dB)")
    # Noise-free case: Wiener with nsr=0 is inverse filtering.
    restored_clean = wiener_deconvolve(blurred, psf, nsr=0.0)
    print(f"PSNR noise-free inverse-filtered: {psnr(sharp, restored_clean):.2f} dB")
    return {
        "psnr_blurred": psnr_blurred,
        "psnr_restored": psnr_restored,
        "psnr_clean": psnr(sharp, restored_clean),
    }


if __name__ == "__main__":
    main()
