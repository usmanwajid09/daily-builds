"""Image augmentation pipeline for digit images (numpy only).

Small geometric + noise transforms for grayscale digit images (e.g.
28x28 MNIST-style arrays): 90-degree rotations, horizontal/vertical
flips, integer-pixel translations (zero-filled), and Gaussian noise.
:func:`augment` applies a random combination of them, and
:func:`augment_n` returns ``n`` different augmented copies of one
image — the usual way to expand a small digit dataset before training.
"""

import operator

import numpy as np

__all__ = [
    "add_gaussian_noise",
    "augment",
    "augment_n",
    "flip_horizontal",
    "flip_vertical",
    "rotate90",
    "translate",
]


def _validate_image(image):
    """Return ``image`` as a float64 2D array, or raise ValueError."""
    try:
        image = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("image must be a numeric 2D numpy array") from None
    if image.ndim != 2:
        raise ValueError(
            f"image must be a 2D array, got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    return image


def _validate_int(value, name):
    """Return ``value`` as an int, or raise ValueError."""
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(
            f"{name} must be an integer, got {value!r}"
        ) from None


def rotate90(image, k=1):
    """Rotate ``image`` by ``k`` * 90 degrees counter-clockwise."""
    image = _validate_image(image)
    k = _validate_int(k, "k")
    return np.rot90(image, k=k)


def flip_horizontal(image):
    """Mirror ``image`` left-right."""
    return np.fliplr(_validate_image(image))


def flip_vertical(image):
    """Mirror ``image`` top-bottom."""
    return np.flipud(_validate_image(image))


def translate(image, dy=0, dx=0):
    """Shift ``image`` by ``dy`` rows and ``dx`` columns.

    The rolled-in area is filled with 0 (nothing wraps around), so a
    shift by (0, 0) is the identity and large shifts push the image
    out entirely.
    """
    image = _validate_image(image)
    dy = _validate_int(dy, "dy")
    dx = _validate_int(dx, "dx")
    height, width = image.shape
    out = np.zeros_like(image)
    r0, r1 = max(0, dy), min(height, height + dy)
    c0, c1 = max(0, dx), min(width, width + dx)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = image[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
    return out


def add_gaussian_noise(image, std=0.1, seed=None):
    """Add zero-mean Gaussian noise with standard deviation ``std``.

    The result is clipped to the image's own value range
    ``[min, max]``, or to ``[0, 1]`` when the image already lies in
    ``[0, 1]``. ``std`` must be >= 0; ``std=0`` returns the image
    unchanged.
    """
    image = _validate_image(image)
    try:
        std = float(std)
    except (TypeError, ValueError):
        raise ValueError(f"std must be a number, got {std!r}") from None
    if not np.isfinite(std) or std < 0:
        raise ValueError(f"std must be >= 0, got {std}")
    if std == 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(0.0, std, size=image.shape)
    if image.min() >= 0 and image.max() <= 1:
        low, high = 0.0, 1.0
    else:
        low, high = float(image.min()), float(image.max())
    return np.clip(noisy, low, high)


def augment(image, seed=None):
    """Return one randomly augmented copy of ``image``.

    Applies a random combination: rotation by a random ``k`` in 0..3
    (90-degree steps), a random horizontal and/or vertical flip,
    a random translation of -2..2 pixels in each direction, and
    Gaussian noise with a random std in 0..0.1. A fixed ``seed`` makes
    the result reproducible. Shape is preserved for square images.
    """
    image = _validate_image(image)
    rng = np.random.default_rng(seed)
    out = np.rot90(image, k=int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    out = translate(
        out,
        dy=int(rng.integers(-2, 3)),
        dx=int(rng.integers(-2, 3)),
    )
    std = float(rng.uniform(0.0, 0.1))
    return add_gaussian_noise(
        out, std=std, seed=int(rng.integers(0, 2 ** 31))
    )


def augment_n(image, n=5, seed=None):
    """Return a list of ``n`` augmented copies of ``image``.

    ``n`` must be a positive int. Each copy gets a different draw
    from one seeded generator, so the copies differ from each other
    in general while a fixed ``seed`` reproduces the whole list.
    """
    image = _validate_image(image)
    n = _validate_int(n, "n")
    if n <= 0:
        raise ValueError(f"n must be a positive int, got {n}")
    rng = np.random.default_rng(seed)
    return [
        augment(image, seed=int(rng.integers(0, 2 ** 31)))
        for _ in range(n)
    ]
