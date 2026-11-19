"""Demo: DoG keypoints on a synthetic image.

Builds a 60x80 grayscale image — a bright disc and a bright rectangle
on a dark background, plus a little noise — computes the
difference-of-Gaussians and prints how many keypoints were found and
where the strongest few are.
"""

import os
import sys

import numpy as np

# Allow running as `python dog_keypoints/demo.py` from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dog_keypoints.keypoints import detect_keypoints, difference_of_gaussians


def make_test_image(height=60, width=80, seed=0):
    """Return a synthetic grayscale image: disc + rectangle on black."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.float64)
    rows, cols = np.mgrid[0:height, 0:width]
    image[(rows - 30) ** 2 + (cols - 25) ** 2 <= 12 ** 2] = 255.0
    image[10:25, 50:71] = 200.0
    image += rng.normal(0.0, 3.0, size=image.shape)
    return image


def main():
    image = make_test_image()
    dog = difference_of_gaussians(image, sigma1=1.0, sigma2=1.6)
    keypoints = detect_keypoints(image, sigma1=1.0, sigma2=1.6,
                                 contrast_threshold=5.0)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale (disc + rectangle)")
    print(f"DoG response: min={dog.min():.1f} max={dog.max():.1f} "
          f"mean|DoG|={np.abs(dog).mean():.2f}")
    print(f"keypoints found: {len(keypoints)} (contrast threshold 5.0)")
    for row, col, response in keypoints[:8]:
        print(f"  keypoint at (row={row}, col={col}) response={response:.1f}")
    return keypoints


if __name__ == "__main__":
    main()
