"""Demo: Gaussian blur on a synthetic noisy image with a sharp edge.

Builds a 40x60 grayscale image — left half dark, right half bright,
plus random noise — blurs it with sigma 1.5, and prints pixel
statistics before and after so you can see the noise drop while the
edge survives (softened, not erased).
"""

import numpy as np

from gaussian_blur.blur import gaussian_blur


def make_test_image(height=40, width=60, seed=0):
    """Return a synthetic image: sharp vertical edge + gaussian noise."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.float64)
    image[:, width // 2:] = 200.0
    image += rng.normal(0.0, 15.0, size=image.shape)
    return image


def _stats(image):
    return (
        f"min={image.min():6.1f} max={image.max():6.1f} "
        f"mean={image.mean():6.1f} std={image.std():6.1f}"
    )


def main():
    image = make_test_image()
    blurred = gaussian_blur(image, sigma=1.5)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, sharp edge + noise")
    print(f"before blur: {_stats(image)}")
    print(f"after blur (sigma=1.5): {_stats(blurred)}")
    # A column just left and just right of the edge shows the step
    # surviving as a smooth ramp instead of a cliff.
    mid = image.shape[1] // 2
    print(f"edge profile before: {image[20, mid - 3:mid + 4].round(1)}")
    print(f"edge profile after:  {blurred[20, mid - 3:mid + 4].round(1)}")
    return blurred


if __name__ == "__main__":
    main()
