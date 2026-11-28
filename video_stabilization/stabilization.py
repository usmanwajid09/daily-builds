"""Simple video stabilisation from scratch (numpy only).

Estimates the frame-to-frame translation between consecutive frames
with phase correlation (FFT), smooths the resulting camera trajectory
with a moving average, and warps each frame back by the difference
between the raw and smoothed trajectories. The intended jitter is
removed while the smooth, deliberate camera motion is kept.
"""

import numpy as np


def _as_2d(frame, name):
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def shift_image(image, dy, dx):
    """Translate a 2D image by (dy, dx) pixels, filling the gap with 0."""
    arr = _as_2d(image, "image")
    if not isinstance(dy, (int, np.integer)) or not isinstance(dx, (int, np.integer)):
        raise ValueError(f"dy and dx must be integers, got {dy!r} and {dx!r}")
    out = np.zeros_like(arr)
    h, w = arr.shape
    y_src = slice(max(0, -dy), h - max(0, dy))
    y_dst = slice(max(0, dy), h - max(0, -dy))
    x_src = slice(max(0, -dx), w - max(0, dx))
    x_dst = slice(max(0, dx), w - max(0, -dx))
    out[y_dst, x_dst] = arr[y_src, x_src]
    return out


def estimate_translation(prev, curr):
    """Estimate the integer (dy, dx) shift from ``prev`` to ``curr``.

    Phase correlation: the peak of the inverse FFT of the normalised
    cross-power spectrum gives the translation between the frames.
    """
    a = _as_2d(prev, "prev")
    b = _as_2d(curr, "curr")
    if a.shape != b.shape:
        raise ValueError(f"frames must have the same shape, got {a.shape} and {b.shape}")
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    cross = fa * np.conj(fb)
    magnitude = np.abs(cross)
    magnitude[magnitude == 0] = 1.0
    corr = np.fft.ifft2(cross / magnitude).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = int(peak[0]), int(peak[1])
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return -dy, -dx


def smooth_trajectory(trajectory, window=5):
    """Moving-average smoothing of an (N, 2) trajectory (edge-padded)."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError(f"trajectory must have shape (N, 2), got {traj.shape}")
    if window % 2 == 0 or window < 1:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    if len(traj) == 0:
        raise ValueError("trajectory must not be empty")
    radius = window // 2
    padded = np.pad(traj, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(traj)
    for i in range(window):
        out += padded[i:i + len(traj)]
    return out / float(window)


def stabilize_frames(frames, window=5):
    """Stabilise a list of grayscale frames; returns a list of frames.

    Frame-to-frame translations are accumulated into a camera
    trajectory, the trajectory is smoothed, and each frame is shifted
    back by (raw − smoothed) so jitter is removed and smooth motion
    is kept.
    """
    if len(frames) == 0:
        raise ValueError("frames must not be empty")
    frames = [_as_2d(f, "frame") for f in frames]
    shape = frames[0].shape
    for f in frames:
        if f.shape != shape:
            raise ValueError("all frames must have the same shape")
    trajectory = np.zeros((len(frames), 2), dtype=np.float64)
    for i in range(1, len(frames)):
        dy, dx = estimate_translation(frames[i - 1], frames[i])
        trajectory[i] = trajectory[i - 1] + np.array([dy, dx], dtype=np.float64)
    smoothed = smooth_trajectory(trajectory, window=window)
    correction = trajectory - smoothed
    return [shift_image(f, -int(round(c[0])), -int(round(c[1])))
            for f, c in zip(frames, correction)]
