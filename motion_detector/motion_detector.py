"""Frame-differencing motion detector from scratch (numpy only).

A running background model is maintained as an exponential moving
average of the frames seen so far::

    background = (1 - lr) * background + lr * frame

A pixel counts as motion when its absolute difference from the
background exceeds ``threshold``.  Motion pixels are grouped into
regions with a from-scratch union-find connected-components labelling
(no scipy), and each region whose pixel area is at least ``min_area``
is reported as a bounding box.  Smaller regions (sensor noise, a few
isolated changed pixels) are filtered out.

Input convention
----------------
Frames are 2D grayscale arrays.  Integer arrays (e.g. ``uint8``) are
interpreted on the 0-255 scale and divided by 255; float arrays are
interpreted on the 0-1 scale and must lie in [0, 1].  Internally
everything is float64 on the 0-1 scale, so ``threshold`` is also on
the 0-1 scale (default 0.1, i.e. about 25 grey levels on 0-255).
"""

import numpy as np

__all__ = [
    "MotionDetector",
    "connected_components",
    "detect_motion",
]


def _validate_frame(frame, name="frame"):
    """Return the frame as a float64 2D array on the 0-1 scale.

    Integer arrays are scaled from 0-255; float arrays must already
    be in [0, 1].  Raises ValueError on bad input.
    """
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) / 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(
                f"{name} float values must be in [0, 1], "
                f"got range [{arr.min()}, {arr.max()}]"
            )
    else:
        raise ValueError(
            f"{name} must have a numeric dtype, got {arr.dtype}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_params(threshold, learning_rate, min_area):
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if not 0 <= learning_rate <= 1:
        raise ValueError(
            f"learning_rate must be in [0, 1], got {learning_rate}"
        )
    if min_area < 1:
        raise ValueError(f"min_area must be >= 1, got {min_area}")


def connected_components(mask):
    """Label 8-connected foreground regions of a boolean mask.

    From-scratch two-pass union-find labelling (no scipy).  Returns a
    list of dicts, one per component, each with ``label``,
    ``area`` (pixel count) and ``bbox`` as
    ``(y_min, x_min, y_max, x_max)`` with inclusive bounds.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got shape {mask.shape}")
    if mask.size == 0:
        raise ValueError("mask must not be empty")
    mask = mask.astype(bool)

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent = [0]  # index 0 = background, never used as a label

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    next_label = 0
    # First pass: provisional labels, unioning with already-seen
    # 8-connected neighbours (left, up, up-left, up-right).
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            neighbours = []
            if x > 0 and labels[y, x - 1]:
                neighbours.append(labels[y, x - 1])
            if y > 0:
                if labels[y - 1, x]:
                    neighbours.append(labels[y - 1, x])
                if x > 0 and labels[y - 1, x - 1]:
                    neighbours.append(labels[y - 1, x - 1])
                if x + 1 < width and labels[y - 1, x + 1]:
                    neighbours.append(labels[y - 1, x + 1])
            if not neighbours:
                next_label += 1
                parent.append(next_label)
                labels[y, x] = next_label
            else:
                labels[y, x] = neighbours[0]
                for other in neighbours[1:]:
                    union(neighbours[0], other)

    # Second pass: resolve roots and accumulate area + bounding box.
    components = {}
    for y in range(height):
        for x in range(width):
            if labels[y, x] == 0:
                continue
            root = find(int(labels[y, x]))
            comp = components.get(root)
            if comp is None:
                components[root] = {
                    "label": root,
                    "area": 1,
                    "bbox": (y, x, y, x),
                }
            else:
                y_min, x_min, y_max, x_max = comp["bbox"]
                comp["area"] += 1
                comp["bbox"] = (
                    min(y_min, y),
                    min(x_min, x),
                    max(y_max, y),
                    max(x_max, x),
                )
    return list(components.values())


def detect_motion(frame, background, threshold=0.1, min_area=20):
    """Detect motion in ``frame`` against a fixed ``background``.

    Pure function (does not update the background).  Returns a dict
    with ``mask`` (bool array), ``bounding_boxes`` (list of
    ``(y_min, x_min, y_max, x_max)`` tuples, inclusive bounds),
    ``motion_pixels`` (int) and ``motion_fraction`` (float).
    """
    frame = _validate_frame(frame, "frame")
    background = _validate_frame(background, "background")
    if frame.shape != background.shape:
        raise ValueError(
            f"frame and background must have the same shape, "
            f"got {frame.shape} and {background.shape}"
        )
    _validate_params(threshold, 0.05, min_area)

    mask = np.abs(frame - background) > threshold
    boxes = [
        comp["bbox"]
        for comp in connected_components(mask)
        if comp["area"] >= min_area
    ]
    motion_pixels = int(mask.sum())
    return {
        "mask": mask,
        "bounding_boxes": boxes,
        "motion_pixels": motion_pixels,
        "motion_fraction": motion_pixels / mask.size,
    }


class MotionDetector:
    """Running-background motion detector.

    Parameters
    ----------
    threshold : float, default 0.1
        Minimum absolute difference (0-1 scale) from the background
        for a pixel to count as motion.
    learning_rate : float, default 0.05
        Background update rate in [0, 1].  0 freezes the background
        after the first frame; 1 replaces it with each new frame.
    min_area : int, default 20
        Minimum component pixel area for a region to be reported.
    """

    def __init__(self, threshold=0.1, learning_rate=0.05, min_area=20):
        _validate_params(threshold, learning_rate, min_area)
        self.threshold = float(threshold)
        self.learning_rate = float(learning_rate)
        self.min_area = int(min_area)
        self.background = None
        self.frame_shape = None
        self.frame_count = 0

    def reset(self):
        """Forget the background; the next frame becomes the new one."""
        self.background = None
        self.frame_shape = None
        self.frame_count = 0

    def update_background(self, frame):
        """Fold ``frame`` into the background model and return it."""
        frame = _validate_frame(frame, "frame")
        if self.background is None:
            self.background = frame.copy()
            self.frame_shape = frame.shape
        else:
            if frame.shape != self.frame_shape:
                raise ValueError(
                    f"frame shape changed: expected {self.frame_shape}, "
                    f"got {frame.shape}"
                )
            lr = self.learning_rate
            self.background = (1.0 - lr) * self.background + lr * frame
        self.frame_count += 1
        return self.background

    def detect(self, frame):
        """Detect motion in ``frame`` against the running background.

        The first frame initialises the background and reports no
        motion.  Later frames are compared against the background
        *before* it is updated with the new frame.  Returns a dict
        with ``mask`` (bool array), ``bounding_boxes`` (list of
        ``(y_min, x_min, y_max, x_max)`` tuples, inclusive bounds),
        ``motion_pixels`` (int) and ``motion_fraction`` (float).
        """
        frame = _validate_frame(frame, "frame")
        if self.background is None:
            self.update_background(frame)
            return {
                "mask": np.zeros(frame.shape, dtype=bool),
                "bounding_boxes": [],
                "motion_pixels": 0,
                "motion_fraction": 0.0,
            }
        if frame.shape != self.frame_shape:
            raise ValueError(
                f"frame shape changed: expected {self.frame_shape}, "
                f"got {frame.shape}"
            )
        mask = np.abs(frame - self.background) > self.threshold
        boxes = [
            comp["bbox"]
            for comp in connected_components(mask)
            if comp["area"] >= self.min_area
        ]
        motion_pixels = int(mask.sum())
        self.update_background(frame)
        return {
            "mask": mask,
            "bounding_boxes": boxes,
            "motion_pixels": motion_pixels,
            "motion_fraction": motion_pixels / mask.size,
        }
