"""Count coins in a synthetic image (numpy only).

Pipeline: threshold the image to a binary mask, run a two-pass
chamfer distance transform on it, and take the local maxima of the
distance transform as coin centres.  The distance-transform value at
a centre is (approximately) the coin's radius, so each detected coin
is classified by matching its radius to the nearest known radius in
:const:`RADIUS_TO_VALUE`, which also gives its denomination.

Denomination map (radius in pixels -> value):

    radius 10 -> 1
    radius 14 -> 5
    radius 18 -> 10

Only numpy is used -- no OpenCV, no scipy.  All images are synthetic,
generated in code by :func:`make_coins_image`.
"""

import numpy as np

__all__ = [
    "RADIUS_TO_VALUE",
    "VALUE_TO_RADIUS",
    "CoinInfo",
    "make_coins_image",
    "distance_transform",
    "count_coins",
]

# Fixed denomination map: coin radius (pixels) -> value.
RADIUS_TO_VALUE = {10: 1, 14: 5, 18: 10}
# Inverse map: value -> radius, for generating scenes by denomination.
VALUE_TO_RADIUS = {value: radius for radius, value in RADIUS_TO_VALUE.items()}

# A detected radius maps to a known denomination only if it is this
# close (pixels) to one of the radii in RADIUS_TO_VALUE.
_RADIUS_TOLERANCE = 3.0

# Chamfer 3-4 mask: (d_row, d_col, weight) for the forward pass;
# the backward pass mirrors it.  Distances are divided by 3 so the
# result is in approximate pixel units.
_FORWARD_OFFSETS = [(-1, -1, 4), (-1, 0, 3), (-1, 1, 4), (0, -1, 3)]
_BACKWARD_OFFSETS = [(0, 1, 3), (1, -1, 4), (1, 0, 3), (1, 1, 4)]
_ORTHOGONAL_WEIGHT = 3


class CoinInfo(dict):
    """One detected coin.

    Behaves like a dict with keys ``x`` (column), ``y`` (row),
    ``radius`` (pixels, float) and ``denomination`` (int, or ``None``
    when the radius matches no known denomination) -- plus attribute
    access for convenience.
    """

    def __init__(self, x, y, radius, denomination):
        super().__init__(
            x=int(x),
            y=int(y),
            radius=float(radius),
            denomination=(
                None if denomination is None else int(denomination)
            ),
        )

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _validate_min_radius(min_radius):
    if isinstance(min_radius, bool) or not isinstance(
        min_radius, (int, np.integer)
    ):
        raise ValueError(
            f"min_radius must be an integer >= 1, got {min_radius!r}"
        )
    if min_radius < 1:
        raise ValueError(f"min_radius must be >= 1, got {min_radius}")


def _validate_size(size):
    if isinstance(size, bool) or isinstance(size, (int, np.integer)):
        if size < 20:
            raise ValueError(f"size must be >= 20, got {size}")
        return (int(size), int(size))
    if isinstance(size, (tuple, list)) and len(size) == 2:
        height, width = size
        if isinstance(height, bool) or isinstance(width, bool) or not (
            isinstance(height, (int, np.integer))
            and isinstance(width, (int, np.integer))
        ):
            raise ValueError(
                f"size must be an int or a pair of ints, got {size!r}"
            )
        if height < 20 or width < 20:
            raise ValueError(
                f"size dimensions must each be >= 20, got {size!r}"
            )
        return (int(height), int(width))
    raise ValueError(f"size must be an int or a pair of ints, got {size!r}")


def make_coins_image(coins, size=200, seed=None, by="radius",
                     noise_level=8.0, background=20.0, brightness=220.0):
    """Generate a synthetic grayscale image of bright coins.

    Parameters
    ----------
    coins : list of (x, y, size_or_value) tuples
        ``x`` is the column, ``y`` is the row of the coin centre.
        With ``by="radius"`` (default) the third element is the coin
        radius in pixels; with ``by="denomination"`` it is a value
        from :const:`VALUE_TO_RADIUS` (1, 5 or 10) and the radius is
        looked up.  Coins must lie fully inside the image.
    size : int or (height, width), default 200
        Image size; each dimension must be >= 20.
    seed : int or None
        Seed for the noise generator; same seed -> same image.
    by : {"radius", "denomination"}
        How to interpret the third element of each coin tuple.
    noise_level : float, default 8.0
        Standard deviation of the Gaussian noise added to every
        pixel (must be >= 0).
    background : float, default 20.0
        Grey level of the background (must be in [0, 255]).
    brightness : float, default 220.0
        Grey level of the coins (must be in [0, 255] and brighter
        than ``background``).

    Returns
    -------
    (H, W) uint8 ndarray
        Dark background with bright filled circles.
    """
    height, width = _validate_size(size)
    if by not in ("radius", "denomination"):
        raise ValueError(f"by must be 'radius' or 'denomination', got {by!r}")
    if noise_level < 0:
        raise ValueError(f"noise_level must be >= 0, got {noise_level}")
    for name, value in (("background", background),
                        ("brightness", brightness)):
        if not 0 <= value <= 255:
            raise ValueError(f"{name} must be in [0, 255], got {value}")
    if brightness <= background:
        raise ValueError(
            f"brightness ({brightness}) must be brighter than "
            f"background ({background})"
        )
    if not isinstance(coins, (list, tuple)):
        raise ValueError(f"coins must be a list of tuples, got {type(coins).__name__}")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError(f"seed must be an int or None, got {seed!r}")

    parsed = []
    for coin in coins:
        if not isinstance(coin, (tuple, list)) or len(coin) != 3:
            raise ValueError(
                f"each coin must be an (x, y, radius) or "
                f"(x, y, denomination) tuple, got {coin!r}"
            )
        x, y, third = coin
        if isinstance(x, bool) or isinstance(y, bool) or not (
            isinstance(x, (int, np.integer, float, np.floating))
            and isinstance(y, (int, np.integer, float, np.floating))
        ):
            raise ValueError(f"coin centre must be numeric, got {coin!r}")
        if by == "radius":
            radius = third
            if isinstance(radius, bool) or not isinstance(
                radius, (int, np.integer, float, np.floating)
            ):
                raise ValueError(f"coin radius must be numeric, got {coin!r}")
            if radius < 1:
                raise ValueError(f"coin radius must be >= 1, got {radius}")
        else:
            if third not in VALUE_TO_RADIUS:
                raise ValueError(
                    f"unknown denomination {third!r}; "
                    f"expected one of {sorted(VALUE_TO_RADIUS)}"
                )
            radius = VALUE_TO_RADIUS[third]
        x, y, radius = int(x), int(y), int(radius)
        if not (radius <= x < width - radius and radius <= y < height - radius):
            raise ValueError(
                f"coin at ({x}, {y}) with radius {radius} does not fit "
                f"fully inside a {height}x{width} image"
            )
        parsed.append((x, y, radius))

    rng = np.random.default_rng(seed)
    image = np.full((height, width), float(background), dtype=np.float64)
    rows, cols = np.indices((height, width))
    for x, y, radius in parsed:
        image[(rows - y) ** 2 + (cols - x) ** 2 <= radius ** 2] = brightness
    if noise_level > 0:
        image = image + rng.normal(0.0, noise_level, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _to_grayscale(image):
    """Return a 2D float array, or raise ValueError on bad input."""
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"image must be a numpy array, got {type(image).__name__}"
        )
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2D grayscale or 3D RGB, got {image.ndim}D "
            f"with shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim == 3:
        if image.shape[2] not in (3, 4):
            raise ValueError(
                "3D image must have 3 (RGB) or 4 (RGBA) channels, "
                f"got shape {image.shape}"
            )
        # Simple mean over channels (drop alpha if present).
        gray = image[:, :, :3].astype(np.float64).mean(axis=2)
    else:
        gray = image.astype(np.float64)
    return gray


def _validate_threshold(threshold, image_max):
    """Validate the threshold for the image's intensity scale.

    Normalised images (max <= 1) take a threshold in (0, 1);
    0-255 images take one in (0, 255).  A threshold in (0, 1] on a
    0-255 image is read as a fraction of 255.
    """
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"threshold must be a number, got {threshold!r}")
    threshold = float(threshold)
    if image_max <= 1.0:
        if not 0 < threshold < 1:
            raise ValueError(
                f"threshold must be in (0, 1) for a normalised image, "
                f"got {threshold}"
            )
        return threshold
    if not 0 < threshold < 255:
        raise ValueError(
            f"threshold must be in (0, 255) for a 0-255 image, got {threshold}"
        )
    if threshold <= 1:
        return threshold * 255.0
    return threshold


def _shifted(array, d_row, d_col, add):
    """``array`` shifted so entry (r, c) holds array[r + d_row, c + d_col].

    Out-of-bounds entries are +inf (no neighbour there).
    """
    h, w = array.shape
    out = np.full(array.shape, np.inf)
    r_src = slice(max(0, d_row), h - max(0, -d_row))
    c_src = slice(max(0, d_col), w - max(0, -d_col))
    r_dst = slice(max(0, -d_row), h - max(0, d_row))
    c_dst = slice(max(0, -d_col), w - max(0, d_col))
    out[r_dst, c_dst] = array[r_src, c_src] + add
    return out


def distance_transform(binary):
    """Chamfer (3-4) distance transform of a binary image.

    For every foreground pixel, the approximate distance (in pixels)
    to the nearest background pixel; background pixels are 0.  The
    two raster passes are repeated until nothing changes, which is
    the fixed point of the chamfer metric.

    Parameters
    ----------
    binary : (H, W) array of 0/1 ints or bools

    Returns
    -------
    (H, W) float64 ndarray
    """
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError(
            f"expected a 2D binary image, got shape {binary.shape}"
        )
    if binary.size == 0:
        raise ValueError("image must not be empty")
    if binary.dtype == bool:
        mask = binary
    elif np.issubdtype(binary.dtype, np.number):
        values = np.unique(binary)
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(
                "binary image may only contain 0 and 1, "
                f"got values {values[:5]}"
            )
        mask = binary.astype(bool)
    else:
        raise ValueError(f"image dtype {binary.dtype} is not numeric")
    if not np.any(~mask):
        raise ValueError(
            "image has no background pixels (all foreground); "
            "distances are undefined"
        )

    dist = np.where(mask, np.inf, 0.0)
    # Each sweep propagates distances a few pixels further; repeat
    # until the fixed point is reached (a disc of radius r needs
    # only a handful of sweeps, and every update is a valid path to
    # the background, so the fixed point is the chamfer distance).
    for _ in range(mask.shape[0] + mask.shape[1]):
        changed = False
        for offsets in (_FORWARD_OFFSETS, _BACKWARD_OFFSETS):
            for d_row, d_col, weight in offsets:
                cand = _shifted(dist, d_row, d_col, weight)
                better = cand < dist
                if np.any(better):
                    changed = True
                    dist = np.where(better, cand, dist)
        if not changed:
            break
    return dist / float(_ORTHOGONAL_WEIGHT)


def _local_maxima(dist, min_radius):
    """Pixels that are >= every neighbour in a 3x3 window and >= min_radius."""
    padded = np.pad(dist, 1, mode="constant", constant_values=-np.inf)
    neighbourhood_max = np.full(dist.shape, -np.inf)
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            neighbourhood_max = np.maximum(
                neighbourhood_max,
                padded[1 + d_row:1 + d_row + dist.shape[0],
                       1 + d_col:1 + d_col + dist.shape[1]],
            )
    rows, cols = np.nonzero(
        (dist >= neighbourhood_max) & (dist >= min_radius)
    )
    return list(zip(rows.tolist(), cols.tolist()))


def _denomination_for_radius(radius):
    """Nearest known denomination for ``radius``, or None if too far."""
    nearest = min(
        RADIUS_TO_VALUE, key=lambda known: abs(known - radius)
    )
    if abs(nearest - radius) <= _RADIUS_TOLERANCE:
        return RADIUS_TO_VALUE[nearest]
    return None


def count_coins(image, threshold=0.5, min_radius=5):
    """Count the coins in an image and total their value.

    Parameters
    ----------
    image : (H, W) grayscale or (H, W, 3/4) RGB(A) ndarray
        Bright coins on a dark background.
    threshold : float, default 0.5
        Binarisation threshold: in (0, 1) for a normalised image,
        in (0, 255) for a 0-255 image (a value in (0, 1] on a 0-255
        image is read as a fraction of 255).
    min_radius : int, default 5
        Coins with an estimated radius below this are ignored
        (drops speckle noise); must be >= 1.

    Returns
    -------
    (coins, summary) : (list[CoinInfo], dict)
        ``coins`` holds one entry per detected coin with ``x``,
        ``y``, ``radius`` and ``denomination`` (``None`` when the
        radius matches no known denomination).  ``summary`` has
        ``count`` (number of coins), ``total_value`` (sum of the
        denominations of the recognised coins) and
        ``by_denomination`` (``{denomination: count}``).
    """
    gray = _to_grayscale(image)
    _validate_min_radius(min_radius)
    level = _validate_threshold(threshold, float(gray.max()) if gray.size else 0.0)
    binary = gray > level
    if not np.any(binary):
        return [], {"count": 0, "total_value": 0, "by_denomination": {}}
    if not np.any(~binary):
        raise ValueError(
            "image has no background pixels (all foreground); "
            "cannot detect coins"
        )

    dist = distance_transform(binary)
    candidates = _local_maxima(dist, min_radius)
    # Non-maximum suppression, strongest peak first: a candidate is
    # dropped if it lies within half the radius of an already
    # accepted (stronger) coin -- that merges the small plateau of
    # equal maxima at one coin's centre without merging two
    # genuinely separate (even slightly overlapping) coins.
    candidates.sort(key=lambda rc: dist[rc[0], rc[1]], reverse=True)
    coins = []
    for row, col in candidates:
        radius = float(dist[row, col])
        if any(
            (row - kept["y"]) ** 2 + (col - kept["x"]) ** 2
            <= (0.5 * kept["radius"]) ** 2
            for kept in coins
        ):
            continue
        coins.append(
            CoinInfo(
                x=col,
                y=row,
                radius=radius,
                denomination=_denomination_for_radius(radius),
            )
        )

    coins.sort(key=lambda coin: (coin["y"], coin["x"]))
    by_denomination = {}
    total_value = 0
    for coin in coins:
        if coin["denomination"] is not None:
            by_denomination[coin["denomination"]] = (
                by_denomination.get(coin["denomination"], 0) + 1
            )
            total_value += coin["denomination"]
    summary = {
        "count": len(coins),
        "total_value": total_value,
        "by_denomination": by_denomination,
    }
    return coins, summary
