"""Handwritten-style signature verification from synthetic pen strokes.

A signature is an ``(N, 2)`` array of ``(x, y)`` pen points (plus
optional timestamps).  The pipeline is:

1. :func:`make_signature` -- generate a synthetic signature trajectory
   for a person.  Each person has a distinctive smooth base stroke
   (seeded by their id); genuine samples add only small jitter and a
   mild speed warp, while forgeries use a different person's stroke
   shape plus extra wobble and a different speed profile.
2. :func:`extract_features` -- stroke features: aspect ratio, total
   path length, mean/std curvature, speed mean/std (from point
   spacing), number of direction changes, bounding-box fill.
3. :func:`dtw_distance` -- classic dynamic time warping between two
   point sequences on normalised coordinates (Euclidean point cost),
   returned as the mean cost per aligned step.
4. :func:`enroll` / :func:`verify` -- store reference trajectories per
   person; verification compares the mean DTW distance to the enrolled
   references against a threshold.

Only numpy and the standard library are used.  All data is synthetic
and generated in code; no external files are read.
"""

import hashlib

import numpy as np

__all__ = [
    "DEFAULT_THRESHOLD",
    "MIN_POINTS",
    "SignatureVerifier",
    "create_enrollment",
    "dtw_distance",
    "enroll",
    "extract_features",
    "make_signature",
    "normalise_trajectory",
    "verify",
]

MIN_POINTS = 10
DEFAULT_THRESHOLD = 0.12


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _as_trajectory(trajectory, name="trajectory"):
    """Validate and return a trajectory as a float ``(N, 2)`` array."""
    try:
        arr = np.asarray(trajectory, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array of (x, y) points") from exc
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"{name} must be a 2D array of shape (N, 2) with (x, y) points, "
            f"got shape {arr.shape}"
        )
    if arr.shape[0] < 2:
        raise ValueError(
            f"{name} has too few points: need at least 2, got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers")
    return arr


def _validate_threshold(threshold):
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold must be a positive number, got {threshold!r}") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"threshold must be a positive number, got {threshold!r}")
    return value


def _person_seed(person_id, salt=""):
    """Stable integer seed derived from a person id (hashlib, not hash())."""
    if not isinstance(person_id, str) or not person_id.strip():
        raise ValueError("person_id must be a non-empty string")
    digest = hashlib.sha256(f"{salt}:{person_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _base_params(rng):
    """One person's distinctive smooth-stroke parameters."""
    return {
        "fx1": float(rng.uniform(1.0, 3.0)),
        "fx2": float(rng.uniform(3.0, 6.0)),
        "fy1": float(rng.uniform(1.0, 3.0)),
        "fy2": float(rng.uniform(3.0, 6.0)),
        "ax1": float(rng.uniform(0.15, 0.35)),
        "ax2": float(rng.uniform(0.05, 0.20)),
        "ay1": float(rng.uniform(0.25, 0.55)),
        "ay2": float(rng.uniform(0.05, 0.20)),
        "px1": float(rng.uniform(0, 2 * np.pi)),
        "px2": float(rng.uniform(0, 2 * np.pi)),
        "py1": float(rng.uniform(0, 2 * np.pi)),
        "py2": float(rng.uniform(0, 2 * np.pi)),
        "drift": float(rng.uniform(0.3, 0.8)),
    }


def _base_curve(params, t):
    """Smooth parametric stroke for parameter set ``params`` at ``t`` in [0, 1]."""
    x = (
        2.0 * t
        + params["ax1"] * np.sin(2 * np.pi * params["fx1"] * t + params["px1"])
        + params["ax2"] * np.sin(2 * np.pi * params["fx2"] * t + params["px2"])
        + params["drift"] * t * t
    )
    y = (
        params["ay1"] * np.sin(2 * np.pi * params["fy1"] * t + params["py1"])
        + params["ay2"] * np.sin(2 * np.pi * params["fy2"] * t + params["py2"])
        + 0.35 * np.sin(np.pi * t)  # gentle overall arch
    )
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Synthetic signatures
# ---------------------------------------------------------------------------

def make_signature(person_id, genuine=True, seed=None, n_points=120):
    """Generate a synthetic signature trajectory for ``person_id``.

    Parameters
    ----------
    person_id : str
        Whose signature this is; determines the base stroke shape.
    genuine : bool
        True: the person's own signing (small jitter, mild speed warp).
        False: a forgery -- a different person's stroke shape with
        extra wobble and a different speed profile.
    seed : int or None
        Sample seed.  ``None`` derives a deterministic seed from
        ``person_id`` (same call twice gives the same trajectory);
        pass an explicit seed for distinct samples of one person.
    n_points : int
        Number of pen points; must be at least ``MIN_POINTS``.

    Returns
    -------
    numpy.ndarray of shape ``(n_points, 2)`` with ``(x, y)`` points.
    """
    if not isinstance(person_id, str) or not person_id.strip():
        raise ValueError("person_id must be a non-empty string")
    if not isinstance(n_points, (int, np.integer)) or int(n_points) < MIN_POINTS:
        raise ValueError(
            f"n_points must be an integer >= {MIN_POINTS}, got {n_points!r}"
        )
    n_points = int(n_points)
    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer or None, got {seed!r}")

    if genuine:
        rng = np.random.default_rng(_person_seed(person_id))
        params = _base_params(rng)
        sample_rng = np.random.default_rng(
            _person_seed(person_id, "sample") if seed is None else int(seed)
        )
        jitter = 0.010
        wobble_amp = 0.0
        warp_power = 1.0
    else:
        # A forger imitates the shape but cannot reproduce it: use a
        # different person's base stroke, add visible wobble, and sign
        # at a different (slower, uneven) speed.
        rng = np.random.default_rng(_person_seed(person_id, "forger"))
        params = _base_params(rng)
        sample_rng = np.random.default_rng(
            _person_seed(person_id, "forgery") if seed is None else int(seed)
        )
        jitter = 0.03
        wobble_amp = 0.09
        warp_power = 1.6

    t = np.linspace(0.0, 1.0, n_points)
    # Variable speed: genuine samples warp time only mildly, forgeries
    # unevenly.  The warp stays monotonic so point order is preserved.
    warp_phase = float(sample_rng.uniform(0, 2 * np.pi))
    warp = t + 0.02 * np.sin(2 * np.pi * t + warp_phase)
    warp = (warp - warp.min()) / (warp.max() - warp.min())
    warp = warp ** warp_power
    warp = (warp - warp.min()) / (warp.max() - warp.min())

    points = _base_curve(params, warp)
    if wobble_amp:
        points = points + wobble_amp * np.column_stack([
            np.sin(2 * np.pi * 9.0 * t + warp_phase),
            np.cos(2 * np.pi * 11.0 * t + warp_phase),
        ])
    points = points + sample_rng.normal(0.0, jitter, size=points.shape)
    return points


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def extract_features(trajectory, timestamps=None):
    """Extract stroke features from a signature trajectory.

    Parameters
    ----------
    trajectory : array-like of shape ``(N, 2)``
        ``(x, y)`` pen points; at least ``MIN_POINTS`` points.
    timestamps : array-like of shape ``(N,)`` or None
        Optional per-point times; when given, speeds are distance /
        time instead of distance per step.

    Returns
    -------
    dict with ``n_points``, ``aspect_ratio``, ``total_path_length``,
    ``mean_curvature``, ``std_curvature``, ``mean_speed``,
    ``std_speed``, ``n_direction_changes`` and ``bbox_fill``.
    """
    traj = _as_trajectory(trajectory)
    if traj.shape[0] < MIN_POINTS:
        raise ValueError(
            f"trajectory has too few points: need at least {MIN_POINTS}, "
            f"got {traj.shape[0]}"
        )
    if timestamps is not None:
        times = np.asarray(timestamps, dtype=float)
        if times.shape != (traj.shape[0],):
            raise ValueError(
                f"timestamps must have shape ({traj.shape[0]},), got {times.shape}"
            )
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("timestamps must be finite and strictly increasing")

    diffs = np.diff(traj, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    steps = steps[steps > 0]
    if steps.size == 0:
        raise ValueError("trajectory has no movement: all points are identical")

    if timestamps is not None:
        dt = np.diff(np.asarray(timestamps, dtype=float))
        speeds = steps / np.maximum(dt, 1e-12)
    else:
        speeds = steps  # per-step distance as a speed proxy

    # Curvature: turning angle between successive segments.
    seg = diffs / np.maximum(np.linalg.norm(diffs, axis=1, keepdims=True), 1e-12)
    dots = np.clip(np.sum(seg[:-1] * seg[1:], axis=1), -1.0, 1.0)
    curvature = np.arccos(dots)

    # Direction changes: sign flips of the x- or y-component of motion.
    dx_sign = np.sign(diffs[:, 0])
    dy_sign = np.sign(diffs[:, 1])
    direction_changes = int(
        np.sum(np.diff(dx_sign) != 0) + np.sum(np.diff(dy_sign) != 0)
    )

    mins = traj.min(axis=0)
    maxs = traj.max(axis=0)
    width, height = maxs - mins
    bbox_area = float(width * height) if width > 0 and height > 0 else 0.0
    # Fill: fraction of the bounding box the stroke's path length covers.
    bbox_fill = float(
        np.sum(steps) / bbox_area if bbox_area > 0 else 0.0
    )

    return {
        "n_points": int(traj.shape[0]),
        "aspect_ratio": float(width / height) if height > 0 else 0.0,
        "total_path_length": float(np.sum(steps)),
        "mean_curvature": float(np.mean(curvature)) if curvature.size else 0.0,
        "std_curvature": float(np.std(curvature)) if curvature.size else 0.0,
        "mean_speed": float(np.mean(speeds)),
        "std_speed": float(np.std(speeds)),
        "n_direction_changes": direction_changes,
        "bbox_fill": bbox_fill,
    }


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------

def normalise_trajectory(trajectory):
    """Centre a trajectory at the origin and scale it to unit spread."""
    traj = _as_trajectory(trajectory)
    centred = traj - traj.mean(axis=0)
    scale = centred.std()
    if scale == 0:
        raise ValueError("trajectory has no spread: all points are identical")
    return centred / scale


def dtw_distance(a, b, window=None):
    """Classic DTW distance between two point sequences.

    Coordinates are normalised first (centred, unit spread) so the
    distance measures shape, not position or size.  The returned
    value is the accumulated Euclidean point cost divided by the
    length of the warping path, i.e. a mean per-step cost.

    Parameters
    ----------
    a, b : array-like of shape ``(N, 2)`` / ``(M, 2)``
    window : int or None
        Sakoe-Chiba half-window; ``None`` means the full matrix.
    """
    seq_a = normalise_trajectory(a)
    seq_b = normalise_trajectory(b)
    n, m = len(seq_a), len(seq_b)
    if window is not None:
        if not isinstance(window, (int, np.integer)) or int(window) < 1:
            raise ValueError(f"window must be a positive integer or None, got {window!r}")
        window = int(window)
    else:
        window = max(n, m)

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - window)
        hi = min(m, i + window)
        for j in range(lo, hi + 1):
            point_cost = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            cost[i, j] = point_cost + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    path_length = n + m  # upper bound; normalises across lengths
    return float(cost[n, m] / path_length)


# ---------------------------------------------------------------------------
# Enrollment / verification
# ---------------------------------------------------------------------------

def create_enrollment():
    """Create an empty enrollment store: ``{person_id: [trajectories]}``."""
    return {}


def enroll(store, person_id, trajectories):
    """Enroll reference trajectories for ``person_id`` in ``store``.

    ``trajectories`` is a single ``(N, 2)`` trajectory or a list of
    them.  Returns the store, so calls can be chained.
    """
    if not isinstance(store, dict):
        raise ValueError("store must be a dict created by create_enrollment()")
    if not isinstance(person_id, str) or not person_id.strip():
        raise ValueError("person_id must be a non-empty string")
    if isinstance(trajectories, np.ndarray):
        trajectories = [trajectories]
    try:
        items = list(trajectories)
    except TypeError as exc:
        raise ValueError("trajectories must be a trajectory or a list of trajectories") from exc
    if not items:
        raise ValueError("trajectories must not be empty: enroll at least one reference")
    validated = [_as_trajectory(t, name=f"reference {k}") for k, t in enumerate(items)]
    for traj in validated:
        if traj.shape[0] < MIN_POINTS:
            raise ValueError(
                f"reference has too few points: need at least {MIN_POINTS}, "
                f"got {traj.shape[0]}"
            )
    store.setdefault(person_id, []).extend(validated)
    return store


def verify(store, person_id, trajectory, threshold=DEFAULT_THRESHOLD):
    """Verify a signature against enrolled references for ``person_id``.

    Returns a dict with ``person_id``, ``distance`` (mean DTW distance
    to the enrolled references), ``threshold``, ``decision``
    (``'genuine'`` or ``'forgery'``) and ``score`` (1 at distance 0,
    falling to 0 at the threshold and beyond).
    """
    threshold = _validate_threshold(threshold)
    if not isinstance(store, dict):
        raise ValueError("store must be a dict created by create_enrollment()")
    if not isinstance(person_id, str) or not person_id.strip():
        raise ValueError("person_id must be a non-empty string")
    if person_id not in store or not store[person_id]:
        raise ValueError(
            f"cannot verify {person_id!r}: no enrolled references "
            "(call enroll() first)"
        )
    traj = _as_trajectory(trajectory)
    if traj.shape[0] < MIN_POINTS:
        raise ValueError(
            f"trajectory has too few points: need at least {MIN_POINTS}, "
            f"got {traj.shape[0]}"
        )
    distances = [dtw_distance(traj, ref) for ref in store[person_id]]
    distance = float(np.mean(distances))
    decision = "genuine" if distance <= threshold else "forgery"
    score = float(max(0.0, 1.0 - distance / threshold))
    return {
        "person_id": person_id,
        "distance": distance,
        "threshold": threshold,
        "decision": decision,
        "score": score,
        "n_references": len(store[person_id]),
    }


class SignatureVerifier:
    """Stateful wrapper around :func:`enroll` / :func:`verify`."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.threshold = _validate_threshold(threshold)
        self.store = create_enrollment()

    def enroll(self, person_id, trajectories):
        """Enroll reference trajectories for ``person_id``; returns self."""
        enroll(self.store, person_id, trajectories)
        return self

    def verify(self, person_id, trajectory, threshold=None):
        """Verify a signature; see :func:`verify` for the result dict."""
        return verify(
            self.store,
            person_id,
            trajectory,
            self.threshold if threshold is None else threshold,
        )

    def is_enrolled(self, person_id):
        """True if ``person_id`` has at least one enrolled reference."""
        return person_id in self.store and bool(self.store[person_id])
