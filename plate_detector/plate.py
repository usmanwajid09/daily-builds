"""Licence-plate detection in synthetic grayscale images, from scratch.

Pipeline: threshold -> morphological closing -> connected-component
labelling -> bounding boxes -> aspect-ratio + area filtering.

Only numpy and the standard library are used (no OpenCV, no scipy).
All test data is generated in code; no external files are needed.
"""

from collections import deque

import numpy as np

__all__ = [
    "make_plate_scene",
    "threshold_image",
    "binary_dilation",
    "binary_erosion",
    "binary_closing",
    "binary_opening",
    "label_components",
    "component_boxes",
    "detect_plates",
    "extract_plate_crop",
    "box_iou",
    "match_detections",
]

# Default detection parameters.
DEFAULT_THRESHOLD = 150
DEFAULT_MIN_AREA = 300
DEFAULT_MIN_ASPECT = 2.0
DEFAULT_MAX_ASPECT = 6.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_image(image):
    """Return ``image`` as a 2-D float array, or raise ``ValueError``."""
    try:
        arr = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError("image must be a 2-D array-like of numbers") from exc
    if arr.ndim != 2:
        raise ValueError(
            f"image must be 2-D (grayscale), got {arr.ndim}-D with shape {arr.shape}"
        )
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("image must not be empty")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("image must contain numbers")
    return arr.astype(np.float64)


def _validate_threshold(threshold):
    if not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise ValueError(f"threshold must be a number, got {threshold!r}")
    if not 0 <= threshold <= 255:
        raise ValueError(f"threshold must be in [0, 255], got {threshold}")


def _validate_filter_params(min_area, min_aspect, max_aspect):
    if min_area < 1:
        raise ValueError(f"min_area must be >= 1, got {min_area}")
    if min_aspect <= 0:
        raise ValueError(f"min_aspect must be > 0, got {min_aspect}")
    if max_aspect < min_aspect:
        raise ValueError(
            f"max_aspect ({max_aspect}) must be >= min_aspect ({min_aspect})"
        )


# ---------------------------------------------------------------------------
# Synthetic scenes
# ---------------------------------------------------------------------------

def make_plate_scene(height=240, width=320, plate_specs=None, n_plates=3,
                     seed=0):
    """Generate a synthetic grayscale scene with licence plates.

    The background is dark (~40) with Gaussian noise and a few bright
    specks. Each plate is a bright rectangle (~220, width ~3-5x height)
    holding dark character blocks (small dark rectangles).

    Parameters
    ----------
    height, width : image size in pixels (both must be >= 40).
    plate_specs : optional list of ``(x, y, w, h)`` plate rectangles.
        When given, ``n_plates`` is ignored.
    n_plates : how many plates to place at random (non-overlapping)
        positions when ``plate_specs`` is not given.
    seed : random seed, so scenes are reproducible.

    Returns
    -------
    (image, truth) : ``image`` is a ``uint8`` array; ``truth`` is a list
        of ``{"x", "y", "width", "height"}`` dicts, one per plate.
    """
    if height < 40 or width < 40:
        raise ValueError(
            f"scene must be at least 40x40 pixels, got {height}x{width}"
        )
    if plate_specs is None and n_plates < 0:
        raise ValueError(f"n_plates must be >= 0, got {n_plates}")

    rng = np.random.default_rng(seed)

    # Dark background + noise.
    image = rng.normal(loc=40.0, scale=8.0, size=(height, width))

    if plate_specs is None:
        plate_specs = _random_plate_specs(rng, height, width, n_plates)
    else:
        plate_specs = [tuple(spec) for spec in plate_specs]

    truth = []
    for (x, y, w, h) in plate_specs:
        if w < 10 or h < 5:
            raise ValueError(
                f"plate {w}x{h} is too small (need at least 10x5 pixels)"
            )
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(
                f"plate at ({x}, {y}) size {w}x{h} does not fit in "
                f"a {width}x{height} scene"
            )
        _draw_plate(image, x, y, w, h, rng)
        truth.append({"x": int(x), "y": int(y), "width": int(w),
                      "height": int(h)})

    # A few isolated bright specks (sensor noise / reflections). They are
    # kept at least 4 px away from every plate: morphological closing
    # bridges gaps of <= 2 px, so a speck any closer would fuse with a
    # plate and stretch its bounding box.
    n_specks = int(0.0005 * height * width)
    forbidden = np.zeros((height, width), dtype=bool)
    for (x, y, w, h) in plate_specs:
        forbidden[max(0, y - 4):min(height, y + h + 4),
                  max(0, x - 4):min(width, x + w + 4)] = True
    placed = 0
    attempts = 0
    while placed < n_specks and attempts < 10 * n_specks + 100:
        attempts += 1
        r, c = int(rng.integers(0, height)), int(rng.integers(0, width))
        if forbidden[r, c]:
            continue
        image[r, c] = float(rng.uniform(180, 230))
        placed += 1

    return np.clip(image, 0, 255).astype(np.uint8), truth


def _random_plate_specs(rng, height, width, n_plates):
    """Place ``n_plates`` non-overlapping plates at random positions."""
    specs = []
    attempts = 0
    while len(specs) < n_plates and attempts < 500:
        attempts += 1
        w = int(rng.integers(50, 81))          # 50..80 px wide
        h = int(round(w / rng.uniform(3.0, 5.0)))
        h = max(12, min(h, 24))
        x = int(rng.integers(5, width - w - 5))
        y = int(rng.integers(5, height - h - 5))
        if all(x + w + 10 < sx or sx + sw + 10 < x or
               y + h + 10 < sy or sy + sh + 10 < y
               for (sx, sy, sw, sh) in specs):
            specs.append((x, y, w, h))
    if len(specs) < n_plates:
        raise ValueError(
            f"could not place {n_plates} non-overlapping plates in a "
            f"{width}x{height} scene"
        )
    return specs


def _draw_plate(image, x, y, w, h, rng):
    """Paint one bright plate with dark character blocks onto ``image``."""
    plate_value = float(rng.uniform(205, 235))
    image[y:y + h, x:x + w] = plate_value
    # Dark characters: 5-7 small rectangles, inset from the plate edge.
    n_chars = int(rng.integers(5, 8))
    margin_x, margin_y = 4, 3
    avail_w = w - 2 * margin_x
    char_w = max(3, avail_w // (2 * n_chars))
    gap = max(2, (avail_w - n_chars * char_w) // max(n_chars - 1, 1))
    char_h = h - 2 * margin_y
    cx = x + margin_x
    for _ in range(n_chars):
        if cx + char_w > x + w - margin_x:
            break
        image[y + margin_y:y + margin_y + char_h,
              cx:cx + char_w] = float(rng.uniform(20, 45))
        cx += char_w + gap


# ---------------------------------------------------------------------------
# Threshold + morphology (numpy shifts only)
# ---------------------------------------------------------------------------

def threshold_image(image, threshold=DEFAULT_THRESHOLD):
    """Binarise a grayscale image: ``True`` where pixel > ``threshold``."""
    arr = _validate_image(image)
    _validate_threshold(threshold)
    return arr > threshold


def _shift(mask, dy, dx):
    """``mask`` shifted by (dy, dx), zero-padded (same shape)."""
    h, w = mask.shape
    shifted = np.zeros_like(mask)
    y0_src, y1_src = max(0, -dy), min(h, h - dy)
    x0_src, x1_src = max(0, -dx), min(w, w - dx)
    y0_dst, y1_dst = max(0, dy), min(h, h + dy)
    x0_dst, x1_dst = max(0, dx), min(w, w + dx)
    shifted[y0_dst:y1_dst, x0_dst:x1_dst] = mask[y0_src:y1_src, x0_src:x1_src]
    return shifted


def binary_dilation(mask, kernel_size=3):
    """Binary dilation with a square structuring element (odd size >= 1)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a non-empty 2-D array")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )
    r = kernel_size // 2
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # Shift the ORIGINAL mask each time; shifting the
            # accumulator would compound and grow the kernel.
            out = out | _shift(mask, dy, dx)
    return out


def binary_erosion(mask, kernel_size=3):
    """Binary erosion with a square structuring element (odd size >= 1)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a non-empty 2-D array")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )
    r = kernel_size // 2
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # Shift the ORIGINAL mask each time; shifting the
            # accumulator would compound and shrink the kernel's effect.
            out = out & _shift(mask, dy, dx)
    return out


def binary_closing(mask, kernel_size=3):
    """Morphological closing (dilation then erosion).

    Bridges the small dark gaps between a plate's characters so the
    whole plate becomes one solid component.
    """
    return binary_erosion(binary_dilation(mask, kernel_size), kernel_size)


def binary_opening(mask, kernel_size=3):
    """Morphological opening (erosion then dilation).

    Removes isolated bright specks (sensor noise) that are smaller
    than the structuring element, while keeping larger shapes intact.
    """
    return binary_dilation(binary_erosion(mask, kernel_size), kernel_size)


# ---------------------------------------------------------------------------
# Connected components + bounding boxes
# ---------------------------------------------------------------------------

def label_components(mask):
    """Label 4-connected foreground components (flood fill).

    Returns ``(labels, n)`` where ``labels`` is an int array (0 =
    background) and ``n`` is the number of components found.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a non-empty 2-D array")
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    n = 0
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or labels[r, c]:
                continue
            n += 1
            queue = deque([(r, c)])
            labels[r, c] = n
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and mask[nr, nc] and not labels[nr, nc]):
                        labels[nr, nc] = n
                        queue.append((nr, nc))
    return labels, n


def component_boxes(labels, n_components):
    """Bounding box + area of each labelled component.

    Returns a list of ``{"x", "y", "width", "height", "area",
    "aspect_ratio"}`` dicts, one per component (label 1..n).
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError("labels must be a non-empty 2-D array")
    if n_components < 0:
        raise ValueError(f"n_components must be >= 0, got {n_components}")
    boxes = []
    for label in range(1, n_components + 1):
        rows, cols = np.nonzero(labels == label)
        if rows.size == 0:
            continue
        y0, y1 = int(rows.min()), int(rows.max())
        x0, x1 = int(cols.min()), int(cols.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        boxes.append({
            "x": x0, "y": y0, "width": w, "height": h,
            "area": int(rows.size),
            "aspect_ratio": round(w / h, 3),
        })
    return boxes


# ---------------------------------------------------------------------------
# Detection + crop extraction
# ---------------------------------------------------------------------------

def detect_plates(image, threshold=DEFAULT_THRESHOLD,
                  min_area=DEFAULT_MIN_AREA,
                  min_aspect=DEFAULT_MIN_ASPECT,
                  max_aspect=DEFAULT_MAX_ASPECT):
    """Detect licence plates in a grayscale image.

    Returns a list of ``{"x", "y", "width", "height", "area",
    "aspect_ratio"}`` dicts, sorted top-to-bottom, left-to-right.
    A region counts as a plate when its aspect ratio is within
    ``[min_aspect, max_aspect]`` and its bright-pixel area is at
    least ``min_area``.
    """
    arr = _validate_image(image)
    _validate_threshold(threshold)
    _validate_filter_params(min_area, min_aspect, max_aspect)

    # Closing bridges the dark gaps between a plate's characters so
    # each plate is one component. (Opening is NOT used here: erosion
    # would split a plate into its thin top/bottom strips, because the
    # characters nearly span the plate height.)
    mask = binary_closing(arr > threshold, kernel_size=3)
    labels, n = label_components(mask)
    boxes = component_boxes(labels, n)
    plates = [b for b in boxes
              if b["area"] >= min_area
              and min_aspect <= b["aspect_ratio"] <= max_aspect]
    plates.sort(key=lambda b: (b["y"], b["x"]))
    return plates


def extract_plate_crop(image, box):
    """Crop the sub-image for a detected plate box.

    ``box`` is a ``{"x", "y", "width", "height"}`` dict or an
    ``(x, y, width, height)`` tuple. Returns the cropped array
    (same dtype as ``image``).
    """
    arr = _validate_image(image)
    if isinstance(box, dict):
        try:
            x, y, w, h = (int(box["x"]), int(box["y"]),
                          int(box["width"]), int(box["height"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "box dict must have integer 'x', 'y', 'width', 'height' keys"
            ) from exc
    else:
        try:
            x, y, w, h = (int(v) for v in box)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "box must be a (x, y, width, height) tuple or a dict with "
                "'x', 'y', 'width', 'height' keys"
            ) from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"crop width and height must be > 0, got {w}x{h}")
    if x < 0 or y < 0:
        raise ValueError(f"crop origin must be >= 0, got ({x}, {y})")
    height, width = arr.shape
    if x + w > width or y + h > height:
        raise ValueError(
            f"crop ({x}, {y}, {w}, {h}) exceeds image size {width}x{height}"
        )
    return arr[y:y + h, x:x + w]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def box_iou(box_a, box_b):
    """Intersection-over-union of two boxes (dicts or (x, y, w, h))."""
    def _xywh(box):
        if isinstance(box, dict):
            return (box["x"], box["y"], box["width"], box["height"])
        return tuple(box)
    ax, ay, aw, ah = _xywh(box_a)
    bx, by, bw, bh = _xywh(box_b)
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def match_detections(detections, truth, iou_threshold=0.5):
    """Greedily match detections to ground-truth boxes by IoU.

    Returns ``(n_matched, matched_pairs)`` where ``matched_pairs`` is a
    list of ``(detection, truth_box, iou)`` tuples.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError(
            f"iou_threshold must be in (0, 1], got {iou_threshold}"
        )
    pairs = []
    used = set()
    for det in detections:
        best_iou, best_idx = 0.0, None
        for i, truth_box in enumerate(truth):
            if i in used:
                continue
            iou = box_iou(det, truth_box)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx is not None and best_iou >= iou_threshold:
            used.add(best_idx)
            pairs.append((det, truth[best_idx], round(best_iou, 3)))
    return len(pairs), pairs
