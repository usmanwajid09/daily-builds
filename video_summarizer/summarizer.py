"""Video summarisation by keyframe extraction, from scratch with numpy.

A synthetic video is a list of HxWx3 uint8 frames. Scenes are detected
from two complementary signals:

* ``frame_difference`` -- mean absolute pixel difference between
  consecutive frames (catches motion and abrupt cuts).
* ``histogram_change`` -- per-channel colour-histogram distance
  (intersection or chi-square; catches colour/content changes even when
  the average pixel value barely moves).

``extract_keyframes`` keeps frame 0 plus every frame whose combined
score exceeds a threshold, enforcing a minimum gap so one cut does not
produce a burst of near-duplicate keyframes. ``summarize_video`` turns
those keyframes into a shot list with timestamps and summary stats.
"""

import numpy as np

__all__ = [
    "make_synthetic_video",
    "frame_difference",
    "frame_histogram",
    "histogram_change",
    "combined_change",
    "extract_keyframes",
    "summarize_video",
]

# Distinct base colours (R, G, B) for the default synthetic scenes.
DEFAULT_SCENE_COLOURS = (
    (200, 60, 60),    # red-ish
    (60, 180, 80),    # green-ish
    (60, 90, 200),    # blue-ish
    (220, 180, 40),   # yellow-ish
    (150, 60, 170),   # purple-ish
)


def _validate_frame(frame, name="frame"):
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"{name} must be an HxWx3 array, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _validate_frames(frames):
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        raise ValueError("frames must be a non-empty list of HxWx3 arrays")
    first = _validate_frame(frames[0], "frames[0]")
    shape = first.shape
    for i, frame in enumerate(frames[1:], start=1):
        arr = _validate_frame(frame, f"frames[{i}]")
        if arr.shape != shape:
            raise ValueError(
                f"all frames must have the same shape: frames[0] is "
                f"{shape}, frames[{i}] is {arr.shape}"
            )
    return [np.asarray(f) for f in frames]


def _validate_fps(fps):
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        raise ValueError(f"fps must be a positive number, got {fps!r}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return float(fps)


def make_synthetic_video(num_frames=90, height=48, width=64, n_scenes=3,
                         fps=30, seed=42, scene_colours=None):
    """Build a synthetic video: a list of HxWx3 uint8 frames.

    The video is split into ``n_scenes`` consecutive scenes of (nearly)
    equal length. Each scene has its own base colour plus a fixed
    per-scene texture pattern, small per-frame Gaussian noise and a slow
    brightness drift -- so frames inside a scene change only slightly,
    while the boundary between scenes is an abrupt colour change.

    Parameters
    ----------
    num_frames : int, frames in the video (must be >= 1).
    height, width : int, frame size in pixels (must be >= 1).
    n_scenes : int, number of scenes (1 <= n_scenes <= num_frames).
    fps : frames per second (must be > 0).
    seed : int, random seed; the video is deterministic for a given seed.
    scene_colours : optional sequence of (R, G, B) tuples, one per scene.
        Defaults to distinct colours from ``DEFAULT_SCENE_COLOURS``.

    Returns
    -------
    list of np.ndarray, each of shape (height, width, 3), dtype uint8.
    """
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) \
            or num_frames < 1:
        raise ValueError(f"num_frames must be a positive int, got {num_frames!r}")
    if not isinstance(height, int) or isinstance(height, bool) or height < 1:
        raise ValueError(f"height must be a positive int, got {height!r}")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError(f"width must be a positive int, got {width!r}")
    if not isinstance(n_scenes, int) or isinstance(n_scenes, bool) \
            or n_scenes < 1 or n_scenes > num_frames:
        raise ValueError(
            f"n_scenes must be an int in [1, num_frames], got {n_scenes!r}"
        )
    _validate_fps(fps)
    if scene_colours is None:
        if n_scenes > len(DEFAULT_SCENE_COLOURS):
            raise ValueError(
                f"n_scenes={n_scenes} exceeds the {len(DEFAULT_SCENE_COLOURS)} "
                "default colours; pass scene_colours explicitly"
            )
        scene_colours = DEFAULT_SCENE_COLOURS[:n_scenes]
    if len(scene_colours) != n_scenes:
        raise ValueError(
            f"scene_colours must have one colour per scene "
            f"({n_scenes}), got {len(scene_colours)}"
        )

    rng = np.random.default_rng(seed)
    frames = []
    # Scene lengths: as equal as possible, e.g. 90 frames / 3 scenes = 30 each.
    base_len, extra = divmod(num_frames, n_scenes)
    scene_lengths = [base_len + (1 if s < extra else 0) for s in range(n_scenes)]

    frame_idx = 0
    for scene, length in enumerate(scene_lengths):
        colour = np.array(scene_colours[scene], dtype=np.float64)
        # Fixed per-scene texture: smooth low-frequency pattern, same for
        # every frame of the scene so it does not look like motion.
        texture = rng.normal(0.0, 12.0, size=(height, width, 1))
        for t in range(length):
            # Slow brightness drift within the scene (at most ~6 levels).
            drift = 6.0 * (t / max(length - 1, 1)) - 3.0
            noise = rng.normal(0.0, 6.0, size=(height, width, 3))
            frame = colour + texture + drift + noise
            frames.append(np.clip(frame, 0, 255).astype(np.uint8))
            frame_idx += 1
    return frames


def frame_difference(frame_a, frame_b):
    """Mean absolute pixel difference between two frames (0..255).

    0 means identical frames; a hard scene cut between very different
    colours scores roughly 100+.
    """
    a = _validate_frame(frame_a, "frame_a")
    b = _validate_frame(frame_b, "frame_b")
    if a.shape != b.shape:
        raise ValueError(
            f"frames must have the same shape: {a.shape} vs {b.shape}"
        )
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def frame_histogram(frame, bins=16):
    """Per-channel normalised histograms, shape (3, bins), rows sum to 1."""
    arr = _validate_frame(frame)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError(f"bins must be an int >= 2, got {bins!r}")
    hist = np.zeros((3, bins), dtype=np.float64)
    for c in range(3):
        counts, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256))
        hist[c] = counts / counts.sum()
    return hist


def histogram_change(frame_a, frame_b, bins=16, method="intersection"):
    """Colour-histogram distance between two frames, in [0, 1].

    ``method="intersection"``: 1 - mean per-channel histogram
    intersection (0 = identical colour distribution).
    ``method="chi_square"``: mean per-channel chi-square distance,
    normalised to [0, 1] (chi-square of normalised histograms is <= 2).
    """
    if method not in ("intersection", "chi_square"):
        raise ValueError(
            f"method must be 'intersection' or 'chi_square', got {method!r}"
        )
    ha = frame_histogram(frame_a, bins=bins)
    hb = frame_histogram(frame_b, bins=bins)
    if method == "intersection":
        intersection = np.minimum(ha, hb).sum(axis=1)  # in [0, 1] per channel
        return float(1.0 - intersection.mean())
    denom = ha + hb
    terms = np.where(denom > 0, (ha - hb) ** 2 / np.where(denom > 0, denom, 1.0), 0.0)
    return float(np.clip(terms.sum(axis=1).mean() / 2.0, 0.0, 1.0))


def combined_change(frame_a, frame_b, bins=16, method="intersection",
                    diff_weight=0.5):
    """Combined scene-change score in [0, 1].

    Weighted average of the normalised frame difference
    (``frame_difference / 255``) and the histogram distance. Either
    signal alone can flag a cut; together they are robust to noise
    (low pixel diff, similar histograms) and to global brightness
    shifts (high pixel diff, similar histograms).
    """
    if not 0.0 <= diff_weight <= 1.0:
        raise ValueError(f"diff_weight must be in [0, 1], got {diff_weight!r}")
    diff = frame_difference(frame_a, frame_b) / 255.0
    hist = histogram_change(frame_a, frame_b, bins=bins, method=method)
    return float(diff_weight * diff + (1.0 - diff_weight) * hist)


def extract_keyframes(frames, fps=30, threshold=0.25, min_gap=None,
                      bins=16, method="intersection", diff_weight=0.5):
    """Pick keyframe indices: frame 0 plus scene-change frames.

    A frame is a keyframe when its combined change score against the
    previous frame exceeds ``threshold`` and it is at least ``min_gap``
    frames after the previous keyframe (default ``max(1, fps // 2)``),
    which stops one cut producing a burst of near-duplicate keyframes.

    Returns a sorted list of frame indices (always includes 0).
    """
    frames = _validate_frames(frames)
    fps = _validate_fps(fps)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) \
            or not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    if min_gap is None:
        min_gap = max(1, int(fps) // 2)
    if not isinstance(min_gap, int) or isinstance(min_gap, bool) or min_gap < 1:
        raise ValueError(f"min_gap must be an int >= 1, got {min_gap!r}")

    keyframes = [0]
    for i in range(1, len(frames)):
        if i - keyframes[-1] < min_gap:
            continue
        score = combined_change(frames[i - 1], frames[i], bins=bins,
                                method=method, diff_weight=diff_weight)
        if score > threshold:
            keyframes.append(i)
    return keyframes


def summarize_video(frames, fps=30, threshold=0.25, min_gap=None,
                    bins=16, method="intersection", diff_weight=0.5):
    """Summarise a video into shots, one per keyframe.

    Returns a dict with ``shots`` -- a list of dicts, each with
    ``shot``, ``start_frame``, ``end_frame``, ``start_time_s``,
    ``end_time_s``, ``duration_s`` and ``keyframe_index`` -- plus
    ``total_frames``, ``num_keyframes``, ``duration_s`` and
    ``compression_ratio`` (total frames / keyframes: how much smaller
    the summary is than the full video).
    """
    frames = _validate_frames(frames)
    fps = _validate_fps(fps)
    keyframes = extract_keyframes(frames, fps=fps, threshold=threshold,
                                  min_gap=min_gap, bins=bins, method=method,
                                  diff_weight=diff_weight)
    total = len(frames)
    shots = []
    for shot, kf in enumerate(keyframes):
        start = kf
        end = keyframes[shot + 1] - 1 if shot + 1 < len(keyframes) else total - 1
        shots.append({
            "shot": shot,
            "start_frame": start,
            "end_frame": end,
            "start_time_s": round(start / fps, 3),
            "end_time_s": round((end + 1) / fps, 3),
            "duration_s": round((end - start + 1) / fps, 3),
            "keyframe_index": kf,
        })
    return {
        "shots": shots,
        "total_frames": total,
        "num_keyframes": len(keyframes),
        "duration_s": round(total / fps, 3),
        "compression_ratio": round(total / len(keyframes), 2),
    }
