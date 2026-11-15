"""Demo: bilateral smoothing on a noisy synthetic step image.

Builds a 40x60 image — dark on the left, bright on the right, with
Gaussian noise — runs :func:`bilateral_filter.filter.bilateral_filter`
and prints how much noise was removed from the flat regions versus
how much of the sharp step survived (a plain blur would smear it).
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bilateral_filter.filter import bilateral_filter


def make_test_image(height=40, width=60, seed=0):
    """Return a noisy step image: 50 on the left, 200 on the right."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 50.0)
    image[:, width // 2:] = 200.0
    return image + rng.normal(0.0, 10.0, size=image.shape)


def main():
    image = make_test_image()
    filtered = bilateral_filter(image, diameter=5, sigma_color=25.0,
                                sigma_space=2.0)
    left = slice(None), slice(0, 25)
    right = slice(None), slice(35, 60)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale")
    print(f"flat-region std before: left={image[left].std():.2f} "
          f"right={image[right].std():.2f}")
    print(f"flat-region std after:  left={filtered[left].std():.2f} "
          f"right={filtered[right].std():.2f}")
    step_before = image[:, 35].mean() - image[:, 24].mean()
    step_after = filtered[:, 35].mean() - filtered[:, 24].mean()
    print(f"step height before: {step_before:.1f}")
    print(f"step height after:  {step_after:.1f}")
    return filtered


if __name__ == "__main__":
    main()
