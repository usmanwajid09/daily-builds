"""Template-based object tracking across synthetic frames (numpy only).

A template patch is cut from the first frame (``init``) and then, for
each new frame, the patch most similar to the template — by normalised
cross-correlation (NCC) — is searched inside a window around the last
known position. The tracker reports the bounding box and the NCC score
each frame, and ``mean_drift`` measures how far the tracked centres
wander from a ground-truth path.
"""

import numpy as np


def make_frame(size=(120, 160), centre=(60.0, 80.0), radius=10,
               value=200.0, background=20.0, noise=0.0, rng=None):
    """Synthetic grayscale frame: a bright disc on a dark background.

    ``centre`` is ``(row, col)``. With ``noise > 0`` Gaussian noise of
    that standard deviation is added (clipped to [0, 255]).
    """
    height, width = size
    if height < 1 or width < 1:
        raise ValueError(f"size must be positive, got {size}")
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    rows, cols = np.indices((height, width))
    dist2 = (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2
    frame = np.where(dist2 <= radius * radius, value, background)
    frame = frame.astype(np.float64)
    if noise > 0:
        rng = np.random.default_rng(rng)
        frame = np.clip(frame + rng.normal(0.0, noise, frame.shape),
                        0.0, 255.0)
    return frame


def make_sequence(n_frames=20, size=(120, 160), start=(30.0, 30.0),
                  velocity=(3.0, 2.0), radius=10, noise=0.0, seed=0):
    """Generate frames of a disc moving at constant velocity.

    Returns ``(frames, true_centres)`` where ``true_centres`` is an
    ``(n_frames, 2)`` array of ``(row, col)`` positions.
    """
    if int(n_frames) < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    rng = np.random.default_rng(seed)
    frames, centres = [], []
    for t in range(int(n_frames)):
        centre = (start[0] + velocity[0] * t,
                  start[1] + velocity[1] * t)
        centres.append(centre)
        frames.append(make_frame(size=size, centre=centre, radius=radius,
                                 noise=noise, rng=rng))
    return frames, np.array(centres)


def ncc(template, patch):
    """Normalised cross-correlation between two same-shaped patches.

    Returns 1.0 for identical patches, -1.0 for perfectly anti-
    correlated ones. A constant patch scores 1.0 against an identical
    constant patch and -1.0 otherwise.
    """
    template = np.asarray(template, dtype=np.float64)
    patch = np.asarray(patch, dtype=np.float64)
    if template.shape != patch.shape:
        raise ValueError(
            f"template shape {template.shape} != patch shape {patch.shape}")
    t = template - template.mean()
    p = patch - patch.mean()
    denom = np.linalg.norm(t) * np.linalg.norm(p)
    if denom == 0:
        return 1.0 if np.array_equal(template, patch) else -1.0
    return float(np.dot(t.ravel(), p.ravel()) / denom)


class ObjectTracker:
    """NCC template tracker with a local search window.

    Parameters
    ----------
    search_radius : int
        Half-width of the square search window (in pixels) around the
        last known centre.
    min_score : float
        NCC score below which a frame is reported as lost (the best
        match is still returned).
    """

    def __init__(self, search_radius=15, min_score=0.5):
        if int(search_radius) < 0:
            raise ValueError(
                f"search_radius must be >= 0, got {search_radius}")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError(
                f"min_score must be in [-1, 1], got {min_score}")
        self.search_radius = int(search_radius)
        self.min_score = float(min_score)
        self._template = None
        self._bbox = None  # (row, col, height, width) of last position

    @property
    def bbox(self):
        """Last tracked bounding box ``(row, col, height, width)``."""
        if self._bbox is None:
            raise ValueError("call init() before reading bbox")
        return self._bbox

    @property
    def is_lost(self):
        """True if the last tracked frame scored below ``min_score``."""
        if self._bbox is None:
            raise ValueError("call init() before reading is_lost")
        return self._last_score < self.min_score

    def init(self, frame, bbox):
        """Initialise the tracker from ``frame`` and ``bbox``.

        ``bbox`` is ``(row, col, height, width)`` and must lie fully
        inside the frame. The template is the patch inside the box.
        """
        frame = np.asarray(frame, dtype=np.float64)
        if frame.ndim != 2:
            raise ValueError(f"frame must be 2D, got shape {frame.shape}")
        if len(bbox) != 4:
            raise ValueError(f"bbox must have 4 elements, got {bbox}")
        row, col, height, width = (int(v) for v in bbox)
        if height < 1 or width < 1:
            raise ValueError(
                f"bbox size must be positive, got {(height, width)}")
        if (row < 0 or col < 0
                or row + height > frame.shape[0]
                or col + width > frame.shape[1]):
            raise ValueError(
                f"bbox {bbox} lies outside frame of shape {frame.shape}")
        self._template = frame[row:row + height, col:col + width].copy()
        if self._template.std() == 0:
            raise ValueError(
                "template patch is constant: pick a bbox with texture")
        self._bbox = (row, col, height, width)
        self._last_score = 1.0
        return self

    def track(self, frame):
        """Track one frame -> ``(bbox, score)``.

        Searches the window around the last centre for the patch with
        the highest NCC against the template.
        """
        if self._template is None:
            raise ValueError("call init() before track()")
        frame = np.asarray(frame, dtype=np.float64)
        if frame.ndim != 2:
            raise ValueError(f"frame must be 2D, got shape {frame.shape}")
        height, width = self._template.shape
        row, col, _, _ = self._bbox
        centre_r, centre_c = row + height // 2, col + width // 2
        best_score, best_pos = -np.inf, (row, col)
        for dr in range(-self.search_radius, self.search_radius + 1):
            for dc in range(-self.search_radius, self.search_radius + 1):
                r, c = centre_r + dr - height // 2, centre_c + dc - width // 2
                if (r < 0 or c < 0
                        or r + height > frame.shape[0]
                        or c + width > frame.shape[1]):
                    continue
                score = ncc(self._template,
                            frame[r:r + height, c:c + width])
                if score > best_score:
                    best_score, best_pos = score, (r, c)
        self._bbox = (best_pos[0], best_pos[1], height, width)
        self._last_score = float(best_score)
        return self._bbox, self._last_score

    def track_sequence(self, frames):
        """Track a list of frames -> list of ``(bbox, score)``."""
        if self._template is None:
            raise ValueError("call init() before track_sequence()")
        frames = list(frames)
        if not frames:
            raise ValueError("need at least one frame to track")
        return [self.track(frame) for frame in frames]


def bbox_centre(bbox):
    """Centre ``(row, col)`` of a ``(row, col, height, width)`` bbox.

    The centre pixel of a ``height``-tall box starting at ``row`` is
    ``row + (height - 1) / 2`` (so a 21x21 box at row 20 is centred on
    pixel 30, not 30.5).
    """
    row, col, height, width = bbox
    return np.array([row + (height - 1) / 2.0, col + (width - 1) / 2.0])


def mean_drift(tracked_bboxes, true_centres):
    """Mean Euclidean distance (px) between tracked and true centres."""
    if len(tracked_bboxes) != len(true_centres):
        raise ValueError(
            f"got {len(tracked_bboxes)} tracked boxes but "
            f"{len(true_centres)} true centres")
    if len(tracked_bboxes) == 0:
        raise ValueError("need at least one frame to measure drift")
    tracked = np.stack([bbox_centre(b) for b in tracked_bboxes])
    true = np.asarray(true_centres, dtype=np.float64)
    if true.ndim != 2 or true.shape[1] != 2:
        raise ValueError(
            f"true_centres must have shape (n, 2), got {true.shape}")
    return float(np.mean(np.linalg.norm(tracked - true, axis=1)))
