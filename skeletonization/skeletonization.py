"""Skeletonization by Zhang-Suen thinning, from scratch (numpy only).

Thinning peels boundary pixels off a binary shape, one layer at a
time, until only a 1-pixel-wide skeleton remains — the shape's
"centreline". The skeleton keeps the topology of the original
(a ring stays a ring, a bar stays a connected line) while making
shape features easy to count: endpoints (skeleton pixels with one
neighbour) mark stroke ends, branch points (three or more neighbours)
mark junctions.

The Zhang-Suen algorithm deletes a foreground pixel P in two
sub-iterations when all of these hold, where B(P) is the count of
foreground neighbours and A(P) the number of 0->1 transitions among
them in clockwise order:

- 2 <= B(P) <= 6 (not an endpoint, not an interior pixel),
- A(P) == 1 (deleting P does not break connectivity),
- sub-iteration 1: P2*P4*P6 == 0 and P4*P6*P8 == 0,
- sub-iteration 2: P2*P4*P8 == 0 and P2*P6*P8 == 0,

with neighbours labelled clockwise from north::

    P9 P2 P3
    P8 P1 P4
    P7 P6 P5

The two sub-iterations alternate so that opposite sides of a stroke
are not deleted in the same pass, which is what keeps 2-pixel-wide
strokes from vanishing entirely.
"""

import numpy as np

__all__ = [
    "count_branch_points",
    "count_endpoints",
    "neighbour_count",
    "zhang_suen_thin",
]


def _validate_binary(binary):
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError(f"binary image must be 2-D, got {binary.ndim}-D")
    if binary.size == 0:
        raise ValueError("binary image must not be empty")
    if binary.dtype != bool:
        raise ValueError(f"binary image must be boolean, got dtype {binary.dtype}")
    return binary


def _neighbours(image):
    """Return the eight neighbours P2..P9 of every pixel, clockwise from north."""
    padded = np.pad(image, 1, mode="constant", constant_values=False)
    p2 = padded[:-2, 1:-1]   # north
    p3 = padded[:-2, 2:]     # north-east
    p4 = padded[1:-1, 2:]    # east
    p5 = padded[2:, 2:]      # south-east
    p6 = padded[2:, 1:-1]    # south
    p7 = padded[2:, :-2]     # south-west
    p8 = padded[1:-1, :-2]   # west
    p9 = padded[:-2, :-2]    # north-west
    return p2, p3, p4, p5, p6, p7, p8, p9


def neighbour_count(binary):
    """Count foreground neighbours (0-8) of every pixel in ``binary``."""
    binary = _validate_binary(binary)
    neighbours = _neighbours(binary)
    count = np.zeros(binary.shape, dtype=int)
    for neighbour in neighbours:
        count += neighbour
    return count


def _deletable(image, step):
    """Boolean mask of foreground pixels Zhang-Suen may delete this pass."""
    p2, p3, p4, p5, p6, p7, p8, p9 = _neighbours(image)
    b = (p2.astype(int) + p3.astype(int) + p4.astype(int) + p5.astype(int)
         + p6.astype(int) + p7.astype(int) + p8.astype(int) + p9.astype(int))
    # A(P): 0->1 transitions in the clockwise sequence P2..P9,P2.
    sequence = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    a = np.zeros(image.shape, dtype=int)
    for first, second in zip(sequence, sequence[1:]):
        a += (~first & second)
    if step == 1:
        product = (p2 & p4 & p6) | (p4 & p6 & p8)
    else:
        product = (p2 & p4 & p8) | (p2 & p6 & p8)
    return image & (b >= 2) & (b <= 6) & (a == 1) & ~product


def zhang_suen_thin(binary, max_iterations=1000):
    """Thin ``binary`` to its 1-pixel skeleton with Zhang-Suen thinning.

    Args:
        binary: 2-D boolean array (foreground True).
        max_iterations: cap on full two-pass iterations.

    Returns:
        ``(skeleton, iterations)`` where ``skeleton`` is a 2-D
        boolean array and ``iterations`` the number of full passes
        run. An empty input returns an empty skeleton.

    Raises:
        ValueError: the image is not a non-empty 2-D boolean array,
            or ``max_iterations`` is below 1.
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
    binary = _validate_binary(binary)
    image = binary.copy()
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        changed = False
        for step in (1, 2):
            deletable = _deletable(image, step)
            if np.any(deletable):
                image[deletable] = False
                changed = True
        if not changed:
            break
    return image, iterations


def count_endpoints(skeleton):
    """Number of skeleton pixels with exactly one foreground neighbour."""
    skeleton = _validate_binary(skeleton)
    counts = neighbour_count(skeleton)
    return int(np.sum(skeleton & (counts == 1)))


def count_branch_points(skeleton):
    """Number of skeleton pixels with three or more foreground neighbours."""
    skeleton = _validate_binary(skeleton)
    counts = neighbour_count(skeleton)
    return int(np.sum(skeleton & (counts >= 3)))
