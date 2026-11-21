"""Demo: noise estimation on a synthetic image with known noise.

Builds a smooth 64x64 test image (a gradient plus two soft discs),
adds Gaussian noise with sigma=12, then estimates sigma with both
methods, prints the noise level and the denoising recommendation,
and shows that the recommended Gaussian blur improves PSNR.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from noise_estimator import (
    estimate_sigma,
    gaussian_denoise,
    noise_level,
    psnr,
    recommend_denoising,
)


def make_clean_image(size=64, seed=0):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    image = 60.0 + 120.0 * xx / size + 40.0 * np.sin(yy / 9.0)
    # Smooth Gaussian blobs (no hard edges, so the Laplacian response
    # of the clean image stays near zero and the estimator measures
    # noise, not edges).
    for cy, cx, spread, value in ((20, 20, 7.0, 120.0), (44, 44, 9.0, -50.0)):
        image = image + value * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * spread ** 2))
    return np.clip(image, 0, 255)


def main():
    clean = make_clean_image()
    rng = np.random.default_rng(0)
    true_sigma = 12.0
    noisy = np.clip(clean + rng.normal(0.0, true_sigma, size=clean.shape), 0, 255)
    est_immerkaer = estimate_sigma(noisy, method="immerkaer")
    est_mad = estimate_sigma(noisy, method="mad")
    est_clean = estimate_sigma(clean, method="immerkaer")
    rec = recommend_denoising(est_immerkaer)
    denoised = gaussian_denoise(noisy, sigma=1.2)
    psnr_noisy = psnr(clean, noisy)
    psnr_denoised = psnr(clean, denoised)
    print(f"true noise sigma: {true_sigma:.1f}")
    print(f"estimated sigma (immerkaer): {est_immerkaer:.2f} "
          f"(error {abs(est_immerkaer - true_sigma):.2f})")
    print(f"estimated sigma (mad): {est_mad:.2f} "
          f"(error {abs(est_mad - true_sigma):.2f})")
    print(f"estimated sigma on the clean image: {est_clean:.2f}")
    print(f"noise level: {noise_level(est_immerkaer)}")
    print(f"recommendation: {rec['method']} {rec['parameters']} — {rec['note']}")
    print(f"PSNR noisy: {psnr_noisy:.2f} dB, PSNR after denoise: "
          f"{psnr_denoised:.2f} dB (gain {psnr_denoised - psnr_noisy:+.2f} dB)")
    return {
        "true_sigma": true_sigma,
        "immerkaer": est_immerkaer,
        "mad": est_mad,
        "psnr_gain": psnr_denoised - psnr_noisy,
    }


if __name__ == "__main__":
    main()
