"""Template matching from scratch (numpy only).

Finds where a small template image occurs inside a larger image using
normalised cross-correlation (NCC): at every candidate position the
image patch is compared with the template after removing each one's
mean and dividing by each one's standard deviation, so the score is
invariant to a uniform brightness shift or contrast rescale of the
patch. Scores lie in [-1, 1]: 1 means an exact (up to an affine
brightness change) copy, -1 means an exact inverted copy, 0 means
uncorrelated.
"""

import numpy as np


def _validate_array(arr, name):
    """Return ``arr`` as a 2D float64 array or raise ``ValueError``."""
    arr = np.asarray(arr)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D grayscale array, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_pair(image, template):
    """Validate image/template and check the template fits the image."""
    image = _validate_array(image, "image")
    template = _validate_array(template, "template")
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise ValueError(
            f"template shape {template.shape} is larger than image shape "
            f"{image.shape}"
        )
    return image, template


def ncc_at(image, template, row, col):
    """Return the NCC score of ``template`` against ``image`` at ``(row, col)``.

    ``(row, col)`` is the top-left corner of the patch in the image.
    The score is in [-1, 1]. A constant template (or a constant image
    patch) has zero variance, so correlation is undefined there and the
    score is defined to be 0 rather than NaN.

    Raises:
        ValueError: on a bad image/template (see :func:`match_template_map`)
            or if ``(row, col)`` puts the template outside the image.
    """
    image, template = _validate_pair(image, template)
    row = int(row)
    col = int(col)
    th, tw = template.shape
    if row < 0 or col < 0 or row + th > image.shape[0] or col + tw > image.shape[1]:
        raise ValueError(
            f"template at ({row}, {col}) does not fit inside image "
            f"of shape {image.shape}"
        )
    patch = image[row:row + th, col:col + tw]
    patch = patch - patch.mean()
    templ = template - template.mean()
    denom = np.sqrt((patch ** 2).sum() * (templ ** 2).sum())
    if denom == 0.0:
        # Constant template and/or constant patch: no variance, so no
        # correlation to measure.
        return 0.0
    return float((patch * templ).sum() / denom)


def match_template_map(image, template):
    """Return the full 2D NCC score map of ``template`` over ``image``.

    The map has shape ``(H - th + 1, W - tw + 1)``; entry ``(r, c)``
    is the NCC score of the template against the image patch whose
    top-left corner is ``(r, c)``. Every entry lies in [-1, 1].

    A constant template (zero variance) scores 0 everywhere, since
    correlation is undefined for it — never NaN. The same applies to
    any constant image patch.

    Raises:
        ValueError: if either input is non-numeric, not 2D grayscale,
            empty, or contains non-finite values, or if the template
            is larger than the image in either dimension.
    """
    image, template = _validate_pair(image, template)
    th, tw = template.shape
    out_h = image.shape[0] - th + 1
    out_w = image.shape[1] - tw + 1

    templ = template - template.mean()
    templ_norm = np.sqrt((templ ** 2).sum())
    if templ_norm == 0.0:
        # Constant template: no variance, so nothing to correlate with.
        return np.zeros((out_h, out_w), dtype=np.float64)

    # Sliding-window view: shape (out_h, out_w, th, tw). For each
    # window, subtract its mean, then correlate with the template.
    windows = np.lib.stride_tricks.sliding_window_view(image, (th, tw))
    win_mean = windows.mean(axis=(2, 3), keepdims=True)
    win_centred = windows - win_mean
    win_norm = np.sqrt((win_centred ** 2).sum(axis=(2, 3)))
    numerator = (win_centred * templ).sum(axis=(2, 3))
    denom = win_norm * templ_norm
    # Divide only where the denominator is nonzero; constant patches
    # (zero variance) score 0 rather than NaN.
    score_map = np.zeros_like(numerator)
    np.divide(numerator, denom, out=score_map, where=denom != 0.0)
    # Guard against tiny floating-point overshoot at the extremes.
    return np.clip(score_map, -1.0, 1.0)


def match_template(image, template):
    """Return ``(row, col, score)`` of the best NCC match of ``template``.

    ``(row, col)`` is the top-left corner of the best-matching patch
    in the image and ``score`` is its NCC score in [-1, 1]. Ties are
    broken by the first (topmost, then leftmost) position. See
    :func:`match_template_map` for the scoring and the constant-
    template convention.

    Raises:
        ValueError: if either input is non-numeric, not 2D grayscale,
            empty, or contains non-finite values, or if the template
            is larger than the image in either dimension.
    """
    score_map = match_template_map(image, template)
    flat_best = int(np.argmax(score_map))
    row, col = divmod(flat_best, score_map.shape[1])
    return row, col, float(score_map[row, col])
