"""Tests for video_stabilization."""

import numpy as np
import pytest

from video_stabilization import (
    estimate_translation,
    shift_image,
    smooth_trajectory,
    stabilize_frames,
)
from video_stabilization.demo import _jitter_std, make_test_frames


def _base_frame(height=24, width=32, seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.normal(100.0, 20.0, size=(height, width))
    frame[10:16, 14:20] = 230.0
    return frame


def test_estimate_known_shift():
    frame = _base_frame()
    shifted = shift_image(frame, 3, -2)
    assert estimate_translation(frame, shifted) == (3, -2)


def test_estimate_zero_shift():
    frame = _base_frame()
    assert estimate_translation(frame, frame) == (0, 0)


def test_shift_image_zero_fill():
    frame = np.ones((6, 6))
    out = shift_image(frame, 2, 0)
    assert out[0, 0] == 0.0 and out[2, 0] == 1.0
    assert out.shape == frame.shape


def test_shift_image_rejects_non_integer():
    with pytest.raises(ValueError, match="integers"):
        shift_image(np.zeros((4, 4)), 1.5, 0)


def test_stabilization_reduces_jitter():
    frames = make_test_frames()
    before, _ = _jitter_std(frames)
    stable = stabilize_frames(frames, window=5)
    after, _ = _jitter_std(stable)
    assert len(stable) == len(frames)
    assert all(s.shape == frames[0].shape for s in stable)
    assert after < before


def test_smooth_trajectory_constant_unchanged():
    traj = np.tile([2.0, -1.0], (9, 1))
    assert np.allclose(smooth_trajectory(traj, window=5), traj)


def test_smooth_trajectory_removes_spike():
    traj = np.zeros((9, 2))
    traj[4] = [9.0, 0.0]
    smoothed = smooth_trajectory(traj, window=5)
    assert smoothed[4, 0] < 9.0
    assert smoothed[4, 0] > 0.0


def test_invalid_frames_rejected():
    with pytest.raises(ValueError, match="empty"):
        stabilize_frames([])
    with pytest.raises(ValueError, match="same shape"):
        stabilize_frames([np.zeros((5, 5)), np.zeros((6, 5))])
    with pytest.raises(ValueError, match="same shape"):
        estimate_translation(np.zeros((5, 5)), np.zeros((5, 6)))
