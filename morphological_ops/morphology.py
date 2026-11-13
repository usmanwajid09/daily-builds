"""Binary morphological operations, implemented from scratch (numpy only).

The four operations, all built from shifted copies of the image:

- **Erosion**: a pixel survives only if the whole structuring element
  fits inside the foreground — shrinks objects, removes specks.
- **Dilation**: a pixel is set if any part of the structuring element
  touches the foreground — grows objects, fills small gaps.
- **Opening** (erosion then dilation): removes small objects and thin
  protrusions while keeping larger shapes intact.
- **Closing** (dilation then erosion): fills small holes and narrow
  gaps while keeping larger shapes intact.

The border is treated as foreground for erosion and background for
dilation, so an object touching the image edge is not eaten away or
grown outward by the border itself.
"""

import numpy as np


def structuring_element(kind="square", size=3):
    """Return a structuring element as a boolean array.

    Args:
        kind: ``"square"`` (all True) or ``"cross"`` (only the centre
            row and column True).
        size: odd positive integer, the element's side length.

    Returns:
        A boolean array of shape ``(size, size)``.

    Raises:
        ValueError: on an unknown ``kind`` or a ``size`` that is not an
            odd positive integer.
    """
    if kind not in ("square", "cross"):
        raise ValueError(f"kind must be 'square' or 'cross', got {kind!r}")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise ValueError(f"size must be an odd positive integer, got {size!r}")
    size = int(size)
    if size < 1 or size % 2 == 0:
        raise ValueError(f"size must be an odd positive integer, got {size}")
    element = np.ones((size, size), dtype=bool)
    if kind == "cross":
        element = np.zeros((size, size), dtype=bool)
        element[size // 2, :] = True
        element[:, size // 2] = True
    return element


def _validate_binary(image):
    """Return ``image`` as a 2D boolean array or raise."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"image must be a 2D binary array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if arr.dtype == bool:
        return arr
    if arr.dtype.kind in "fiu":
        if not np.isfinite(arr.astype(np.float64)).all():
            raise ValueError("image must contain only finite values")
        values = np.unique(arr)
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError(
                f"binary image must contain only 0 and 1, got values {values}"
            )
        return arr.astype(bool)
    raise ValueError(f"image must be boolean or 0/1 numeric, got dtype {arr.dtype}")


def _validate_element(element):
    """Return ``element`` as a 2D boolean array or raise."""
    arr = np.asarray(element)
    if arr.dtype != bool:
        if arr.dtype.kind in "fiu" and np.all(np.isin(np.unique(arr), [0, 1])):
            arr = arr.astype(bool)
        else:
            raise ValueError("structuring element must be boolean or 0/1")
    if arr.ndim != 2:
        raise ValueError(f"structuring element must be 2D, got shape {arr.shape}")
    if arr.shape[0] % 2 == 0 or arr.shape[1] % 2 == 0:
        raise ValueError(
            f"structuring element must have odd dimensions, got {arr.shape}"
        )
    if not arr.any():
        raise ValueError("structuring element must contain at least one True value")
    return arr


def _shifted_or(image, element, fill, combine):
    """Apply ``combine`` over the image shifted by each True element cell.

    ``combine`` is ``np.logical_and`` (erosion) or ``np.logical_or``
    (dilation); ``fill`` is the value shifted in from outside the
    border (True for erosion, False for dilation).
    """
    h, w = image.shape
    eh, ew = element.shape
    rh, rw = eh // 2, ew // 2
    padded = np.pad(image, ((rh, rh), (rw, rw)), mode="constant",
                    constant_values=fill)
    out = None
    for dr in range(eh):
        for dc in range(ew):
            if not element[dr, dc]:
                continue
            shifted = padded[dr:dr + h, dc:dc + w]
            out = shifted.copy() if out is None else combine(out, shifted)
    return out


def erode(image, element=None):
    """Return ``image`` eroded by ``element`` (default 3x3 square)."""
    binary = _validate_binary(image)
    element = _validate_element(structuring_element() if element is None else element)
    return _shifted_or(binary, element, fill=True, combine=np.logical_and)


def dilate(image, element=None):
    """Return ``image`` dilated by ``element`` (default 3x3 square)."""
    binary = _validate_binary(image)
    element = _validate_element(structuring_element() if element is None else element)
    return _shifted_or(binary, element, fill=False, combine=np.logical_or)


def opening(image, element=None):
    """Return the opening of ``image``: erosion followed by dilation."""
    return dilate(erode(image, element), element)


def closing(image, element=None):
    """Return the closing of ``image``: dilation followed by erosion."""
    return erode(dilate(image, element), element)
