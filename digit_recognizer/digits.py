"""Template-matching digit recogniser for synthetic 8x8 digits.

Digits 0-9 are hand-defined 8x8 binary templates. Recognition is
nearest-template matching under Euclidean distance: fast, deterministic,
and accurate on clean or lightly-noised patterns. Noisy variants of a
template (random pixel flips) are still recognised reliably, which makes
the recogniser a good baseline for template-based OCR experiments.
"""

import numpy as np

_DIGIT_ART = {
    0: [
        ".######.",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        ".######.",
    ],
    1: [
        "...##...",
        "..###...",
        "...##...",
        "...##...",
        "...##...",
        "...##...",
        "...##...",
        "..####..",
    ],
    2: [
        ".######.",
        "#......#",
        ".......#",
        "......#.",
        "....##..",
        "...#....",
        "..#.....",
        "########",
    ],
    3: [
        ".######.",
        "#......#",
        ".......#",
        "..#####.",
        ".......#",
        ".......#",
        "#......#",
        ".######.",
    ],
    4: [
        "....##..",
        "...#.#..",
        "..#..#..",
        ".#...#..",
        "########",
        ".....#..",
        ".....#..",
        ".....#..",
    ],
    5: [
        "########",
        "#.......",
        "#.......",
        "######..",
        "......#.",
        "......#.",
        "#.....#.",
        ".#####..",
    ],
    6: [
        "..####..",
        ".#......",
        "#.......",
        "#.####..",
        "#.....#.",
        "#.....#.",
        ".#....#.",
        "..###...",
    ],
    7: [
        "########",
        "......#.",
        ".....#..",
        "....#...",
        "...#....",
        "...#....",
        "..#.....",
        "..#.....",
    ],
    8: [
        ".######.",
        "#......#",
        "#......#",
        ".######.",
        "#......#",
        "#......#",
        "#......#",
        ".######.",
    ],
    9: [
        ".######.",
        "#......#",
        "#......#",
        ".######.",
        ".......#",
        ".......#",
        "#......#",
        ".######.",
    ],
}

TEMPLATE_SHAPE = (8, 8)


def _art_to_array(art):
    return np.array(
        [[1.0 if ch == "#" else 0.0 for ch in row] for row in art],
        dtype=np.float64,
    )


TEMPLATES = {digit: _art_to_array(art) for digit, art in _DIGIT_ART.items()}


def template(digit):
    """Return a copy of the 8x8 binary template for ``digit`` (0-9).

    Raises:
        ValueError: if ``digit`` is not an integer in 0-9.
    """
    if not isinstance(digit, (int, np.integer)) or int(digit) not in TEMPLATES:
        raise ValueError(f"digit must be an integer in 0-9, got {digit!r}")
    return TEMPLATES[int(digit)].copy()


def _validate_image(image):
    image = np.asarray(image, dtype=np.float64)
    if image.shape != TEMPLATE_SHAPE:
        raise ValueError(
            f"image must have shape {TEMPLATE_SHAPE}, got {image.shape}"
        )
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("image values must lie in [0, 1]")
    return image


def add_noise(image, flip_prob=0.05, rng=None):
    """Return a copy of ``image`` with each pixel flipped with prob ``flip_prob``.

    Raises:
        ValueError: if ``flip_prob`` is not in [0, 1] or the image is invalid.
    """
    image = _validate_image(image)
    if not 0.0 <= float(flip_prob) <= 1.0:
        raise ValueError(f"flip_prob must be in [0, 1], got {flip_prob!r}")
    rng = np.random.default_rng(rng)
    flips = rng.random(TEMPLATE_SHAPE) < flip_prob
    noisy = image.copy()
    noisy[flips] = 1.0 - noisy[flips]
    return noisy


def distances(image, templates=None):
    """Return ``{digit: euclidean distance}`` from ``image`` to each template."""
    image = _validate_image(image)
    templates = TEMPLATES if templates is None else templates
    return {
        digit: float(np.linalg.norm(image - tmpl))
        for digit, tmpl in templates.items()
    }


def recognize(image, templates=None):
    """Recognise ``image`` as the nearest template.

    Returns:
        ``(digit, scores)`` where ``scores`` maps each digit 0-9 to its
        Euclidean distance from the image; the smallest distance wins.
        Ties are broken by the smallest digit, so results are deterministic.
    """
    scores = distances(image, templates)
    best = min(scores, key=lambda d: (scores[d], d))
    return best, scores


def accuracy(images, labels, templates=None):
    """Fraction of ``images`` correctly recognised against ``labels``."""
    images = list(images)
    labels = list(labels)
    if len(images) != len(labels):
        raise ValueError(
            f"got {len(images)} images but {len(labels)} labels"
        )
    if not images:
        raise ValueError("need at least one image to score")
    correct = sum(
        recognize(img, templates)[0] == label
        for img, label in zip(images, labels)
    )
    return correct / len(images)
