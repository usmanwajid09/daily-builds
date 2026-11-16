"""Content-based image retrieval (CBIR) from scratch (numpy only).

Images are indexed by their normalised RGB colour histogram (8 bins per
channel, 512 bins total). A query image is ranked against the index by
Pearson correlation between histograms, so images with similar colour
content come back first — the classic colour-based CBIR baseline.
"""

import numpy as np

BINS_PER_CHANNEL = 8
HIST_SIZE = BINS_PER_CHANNEL ** 3


def _validate_image(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must have shape (H, W, 3), got {image.shape}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(image.astype(np.float64)).all():
        raise ValueError("image must contain only finite values")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("image values must lie in [0, 255]")
    return image.astype(np.float64)


def color_histogram(image, bins_per_channel=BINS_PER_CHANNEL):
    """Return the normalised flattened RGB histogram of ``image``.

    The histogram has ``bins_per_channel ** 3`` entries and sums to 1.
    """
    image = _validate_image(image)
    if bins_per_channel < 1:
        raise ValueError(
            f"bins_per_channel must be >= 1, got {bins_per_channel}"
        )
    pixels = image.reshape(-1, 3)
    hist, _ = np.histogramdd(
        pixels,
        bins=bins_per_channel,
        range=[(0, 256), (0, 256), (0, 256)],
    )
    hist = hist.ravel().astype(np.float64)
    total = hist.sum()
    if total <= 0:
        raise ValueError("image has no pixels to histogram")
    return hist / total


def histogram_correlation(h1, h2):
    """Pearson correlation between two normalised histograms in [-1, 1].

    Returns 0.0 if either histogram has zero variance (e.g. a solid
    colour image correlated against itself is handled: identical solid
    histograms still score 1.0 via the identical-histogram shortcut).
    """
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    if h1.shape != h2.shape:
        raise ValueError(
            f"histograms must have the same shape, got {h1.shape} and {h2.shape}"
        )
    if np.array_equal(h1, h2):
        return 1.0
    h1 = h1 - h1.mean()
    h2 = h2 - h2.mean()
    denom = np.sqrt((h1 ** 2).sum() * (h2 ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float((h1 * h2).sum() / denom)


def make_solid(color, size=(64, 64)):
    """Return a solid-colour image for RGB ``color`` (0-255 per channel)."""
    color = np.asarray(color, dtype=np.float64)
    if color.shape != (3,) or (color < 0).any() or (color > 255).any():
        raise ValueError(f"color must be an RGB triple in [0, 255], got {color!r}")
    return np.tile(color, (*size, 1))


def make_split(color1, color2, size=(64, 64), vertical=True):
    """Return an image split into two solid halves of ``color1``/``color2``."""
    image = np.empty((*size, 3), dtype=np.float64)
    if vertical:
        mid = size[1] // 2
        image[:, :mid] = color1
        image[:, mid:] = color2
    else:
        mid = size[0] // 2
        image[:mid, :] = color1
        image[mid:, :] = color2
    return image


def make_gradient(color_start, color_end, size=(64, 64), horizontal=True):
    """Return a linear gradient from ``color_start`` to ``color_end``."""
    start = np.asarray(color_start, dtype=np.float64)
    end = np.asarray(color_end, dtype=np.float64)
    length = size[1] if horizontal else size[0]
    t = np.linspace(0.0, 1.0, length)
    ramp = start[None, :] * (1 - t)[:, None] + end[None, :] * t[:, None]
    ramp = np.clip(ramp, 0.0, 255.0)  # guard against float overshoot
    if horizontal:
        return np.tile(ramp[None, :, :], (size[0], 1, 1))
    return np.tile(ramp[:, None, :], (1, size[1], 1))


def make_blocks(colors, size=(64, 64), grid=4, rng=None):
    """Return an image of random solid blocks drawn from ``colors``."""
    colors = [np.asarray(c, dtype=np.float64) for c in colors]
    if not colors:
        raise ValueError("need at least one color for make_blocks")
    rng = np.random.default_rng(rng)
    image = np.empty((*size, 3), dtype=np.float64)
    bh, bw = size[0] // grid, size[1] // grid
    for i in range(grid):
        for j in range(grid):
            color = colors[int(rng.integers(len(colors)))]
            image[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = color
    return image


class ImageIndex:
    """A colour-histogram index supporting query-by-example retrieval."""

    def __init__(self, bins_per_channel=BINS_PER_CHANNEL):
        if bins_per_channel < 1:
            raise ValueError(
                f"bins_per_channel must be >= 1, got {bins_per_channel}"
            )
        self.bins_per_channel = bins_per_channel
        self._names = []
        self._histograms = []

    def add(self, name, image):
        """Index ``image`` under ``name``. Names need not be unique."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"name must be a non-empty string, got {name!r}")
        hist = color_histogram(image, self.bins_per_channel)
        self._names.append(name)
        self._histograms.append(hist)
        return self

    def __len__(self):
        return len(self._names)

    def query(self, image, top_k=5):
        """Rank the index against ``image`` by histogram correlation.

        Returns a list of ``(name, score)`` sorted by score descending,
        with at most ``top_k`` entries. An exact copy of an indexed
        image scores 1.0 and ranks first.
        """
        if len(self._names) == 0:
            raise ValueError("index is empty: add images before querying")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        query_hist = color_histogram(image, self.bins_per_channel)
        scored = [
            (name, histogram_correlation(query_hist, hist))
            for name, hist in zip(self._names, self._histograms)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]
