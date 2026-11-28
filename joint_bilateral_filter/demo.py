"""Demo: denoise a noisy step image guided by the clean step image.

Builds a 40x60 image — dark on the left, bright on the right — adds
Gaussian noise to it, and filters it twice: once with the noisy image
as its own guide (ordinary bilateral) and once with the clean step
image as guidance. The guided version removes more noise while keeping
the step, because the clean guide knows exactly where the edge is.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from joint_bilateral_filter.filter import joint_bilateral_filter


def make_test_images(height=40, width=60, seed=0):
    """Return (clean step image, noisy version of it)."""
    rng = np.random.default_rng(seed)
    clean = np.full((height, width), 50.0)
    clean[:, width // 2:] = 200.0
    noisy = clean + rng.normal(0.0, 15.0, size=clean.shape)
    return clean, noisy


def main():
    clean, noisy = make_test_images()
    guided = joint_bilateral_filter(noisy, clean, diameter=5,
                                    sigma_color=25.0, sigma_space=2.0)
    plain = joint_bilateral_filter(noisy, noisy, diameter=5,
                                   sigma_color=25.0, sigma_space=2.0)
    flat = (slice(None), slice(0, 25))
    print(f"image: {noisy.shape[0]}x{noisy.shape[1]} grayscale")
    print(f"flat-region std before: {noisy[flat].std():.2f}")
    print(f"flat-region std after (self-guided): {plain[flat].std():.2f}")
    print(f"flat-region std after (clean-guided): {guided[flat].std():.2f}")
    step_before = noisy[:, 35].mean() - noisy[:, 24].mean()
    step_after = guided[:, 35].mean() - guided[:, 24].mean()
    print(f"step height before: {step_before:.1f}")
    print(f"step height after (clean-guided): {step_after:.1f}")
    return guided


if __name__ == "__main__":
    main()
