"""Running-average background subtraction from scratch (numpy only).

Maintains a background model that is the running average of the frames
seen so far: ``bg = (1 - alpha) * bg + alpha * frame``.  A pixel is
foreground when its absolute difference from the background exceeds a
threshold.  Isolated foreground pixels (sensor noise) are removed by
requiring each foreground pixel to have at least ``min_neighbors``
foreground neighbours in its 3x3 neighbourhood.
"""

import numpy as np

__all__ = [
    "BackgroundSubtractor",
    "foreground_mask",
    "running_average_update",
]


def _validate_frame(frame, name="frame"):
    """Return the frame as a float64 2D array, raising on bad input."""
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def running_average_update(background, frame, learning_rate):
    """Return the updated background: ``(1 - a) * bg + a * frame``."""
    bg = _validate_frame(background, "background")
    frame = _validate_frame(frame, "frame")
    if bg.shape != frame.shape:
        raise ValueError(
            f"background and frame must have the same shape, "
            f"got {bg.shape} and {frame.shape}"
        )
    if not 0 < learning_rate <= 1:
        raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
    return (1.0 - learning_rate) * bg + learning_rate * frame


def foreground_mask(frame, background, threshold, min_neighbors=2):
    """Return the foreground mask for one frame against a background.

    A pixel is foreground when ``|frame - background| > threshold`` and
    it has at least ``min_neighbors`` foreground neighbours in its 3x3
    neighbourhood (isolated pixels are treated as noise).
    """
    frame = _validate_frame(frame, "frame")
    background = _validate_frame(background, "background")
    if frame.shape != background.shape:
        raise ValueError(
            f"frame and background must have the same shape, "
            f"got {frame.shape} and {background.shape}"
        )
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if not 0 <= min_neighbors <= 8:
        raise ValueError(f"min_neighbors must be in [0, 8], got {min_neighbors}")
    mask = np.abs(frame - background) > threshold
    if min_neighbors == 0:
        return mask
    neighbours = np.zeros(frame.shape, dtype=int)
    padded = np.pad(mask.astype(int), 1)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neighbours += padded[dy:dy + frame.shape[0], dx:dx + frame.shape[1]]
    return mask & (neighbours >= min_neighbors)


class BackgroundSubtractor:
    """Running-average background model with foreground detection.

    Parameters
    ----------
    learning_rate : float in (0, 1]
        Weight of each new frame in the background model.  Larger
        values adapt faster to scene changes.
    threshold : float >= 0
        Absolute intensity difference above which a pixel is foreground.
    min_neighbors : int in [0, 8]
        Minimum number of foreground neighbours (3x3) a foreground
        pixel needs to survive the noise cleanup.
    """

    def __init__(self, learning_rate=0.05, threshold=25.0, min_neighbors=2):
        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if not isinstance(min_neighbors, (int, np.integer)) or isinstance(min_neighbors, bool):
            raise ValueError(f"min_neighbors must be an integer, got {min_neighbors!r}")
        if not 0 <= min_neighbors <= 8:
            raise ValueError(f"min_neighbors must be in [0, 8], got {min_neighbors}")
        self.learning_rate = float(learning_rate)
        self.threshold = float(threshold)
        self.min_neighbors = int(min_neighbors)
        self._background = None

    def apply(self, frame):
        """Update the background model with ``frame`` and return its mask.

        The first frame initialises the background model and returns an
        all-False mask (there is nothing to compare against yet).
        """
        frame = _validate_frame(frame, "frame")
        if self._background is None:
            self._background = frame.copy()
            return np.zeros(frame.shape, dtype=bool)
        if frame.shape != self._background.shape:
            raise ValueError(
                f"frame shape {frame.shape} does not match background "
                f"shape {self._background.shape}"
            )
        mask = foreground_mask(
            frame, self._background, self.threshold, self.min_neighbors
        )
        self._background = running_average_update(
            self._background, frame, self.learning_rate
        )
        return mask

    def get_background(self):
        """Return a copy of the current background model."""
        if self._background is None:
            raise ValueError("background model not initialised: call apply() first")
        return self._background.copy()

    def reset(self):
        """Forget the background model; the next frame re-initialises it."""
        self._background = None
