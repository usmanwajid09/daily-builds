"""Tests for lucas_kanade_tracker."""

import numpy as np
import pytest

from lucas_kanade_tracker import (
    LucasKanadeTracker,
    textured_disc_frame,
    track_points,
)
from lucas_kanade_tracker.demo import main as demo_main


def _disc_pair(shift=(2, 3), center=(32, 32), radius=12, seed=0):
    prev = textured_disc_frame(center=center, radius=radius, seed=seed)
    curr = textured_disc_frame(
        center=(center[0] + shift[0], center[1] + shift[1]),
        radius=radius, seed=seed)
    return prev, curr


def _interior_points(center=(32, 32), n=8, seed=5):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    radii = rng.uniform(0, 4, size=n)
    return np.column_stack([
        center[0] + radii * np.sin(angles),
        center[1] + radii * np.cos(angles),
    ])


def test_zero_motion_returns_same_points():
    prev, curr = _disc_pair(shift=(0, 0))
    points = _interior_points()
    new_points, status, errors = track_points(prev, curr, points)
    assert np.all(status == 1)
    assert np.allclose(new_points, points, atol=0.05)
    assert np.all(errors < 0.01)


def test_known_shift_recovered():
    prev, curr = _disc_pair(shift=(2, 3))
    points = _interior_points()
    new_points, status, errors = track_points(prev, curr, points)
    assert np.all(status == 1)
    displacement = new_points - points
    assert np.allclose(displacement.mean(axis=0), [2.0, 3.0], atol=0.15)
    assert np.all(np.abs(displacement - [2.0, 3.0]) < 0.5)


def test_negative_shift_recovered():
    prev, curr = _disc_pair(shift=(-3, -2))
    points = _interior_points()
    new_points, status, _ = track_points(prev, curr, points)
    assert np.all(status == 1)
    assert np.allclose((new_points - points).mean(axis=0), [-3.0, -2.0], atol=0.15)


def test_one_pixel_vertical_shift():
    prev, curr = _disc_pair(shift=(1, 0))
    points = _interior_points()
    new_points, status, _ = track_points(prev, curr, points)
    assert np.all(status == 1)
    assert np.allclose((new_points - points).mean(axis=0), [1.0, 0.0], atol=0.15)


def test_flat_region_points_are_lost():
    frame = np.zeros((64, 64))
    points = np.array([[10.0, 10.0], [30.0, 40.0], [50.0, 50.0]])
    new_points, status, errors = track_points(frame, frame, points)
    assert np.all(status == 0)
    assert np.allclose(new_points, points)
    assert np.all(np.isnan(errors))


def test_point_near_border_is_lost():
    prev, curr = _disc_pair()
    points = np.array([[2.0, 2.0], [32.0, 32.0]])
    _, status, _ = track_points(prev, curr, points)
    assert status[0] == 0  # window does not fit
    assert status[1] == 1


def test_rejects_mismatched_frames():
    prev = np.zeros((64, 64))
    curr = np.zeros((32, 32))
    with pytest.raises(ValueError, match="same shape"):
        track_points(prev, curr, np.array([[20.0, 20.0]]))


def test_rejects_bad_inputs():
    prev = np.zeros((64, 64))
    with pytest.raises(ValueError, match="2D"):
        track_points(np.zeros((64, 64, 3)), prev, np.array([[20.0, 20.0]]))
    with pytest.raises(ValueError, match="empty"):
        track_points(np.zeros((0, 0)), prev, np.array([[20.0, 20.0]]))
    with pytest.raises(ValueError, match="finite"):
        track_points(np.full((8, 8), np.nan), prev, np.array([[4.0, 4.0]]))
    with pytest.raises(ValueError, match="shape \\(n_points, 2\\)"):
        track_points(prev, prev, np.array([20.0, 20.0]))
    with pytest.raises(ValueError, match="empty"):
        track_points(prev, prev, np.zeros((0, 2)))
    with pytest.raises(ValueError, match="window_size"):
        track_points(prev, prev, np.array([[20.0, 20.0]]), window_size=16)
    with pytest.raises(ValueError, match="window_size"):
        track_points(prev, prev, np.array([[20.0, 20.0]]), window_size=2)
    with pytest.raises(ValueError, match="max_iterations"):
        track_points(prev, prev, np.array([[20.0, 20.0]]), max_iterations=0)


def test_textured_disc_frame_validation():
    with pytest.raises(ValueError, match="shape"):
        textured_disc_frame(shape=(0, 64))
    with pytest.raises(ValueError, match="radius"):
        textured_disc_frame(radius=0)
    frame = textured_disc_frame()
    assert frame.shape == (64, 64)
    assert frame.min() >= 0.0 and frame.max() <= 1.0


def test_tracker_class_tracks_across_frames():
    tracker = LucasKanadeTracker()
    with pytest.raises(RuntimeError, match="not initialised"):
        tracker.update(np.zeros((64, 64)), np.zeros((64, 64)))
    prev, curr = _disc_pair(shift=(2, 3))
    points = _interior_points()
    tracker.init(prev, points)
    new_points, status, _ = tracker.update(prev, curr)
    assert np.all(status == 1)
    assert np.allclose((new_points - points).mean(axis=0), [2.0, 3.0], atol=0.15)
    # Second step continues from the tracked positions.
    curr2 = textured_disc_frame(center=(36, 38), radius=12, seed=0)
    new2, status2, _ = tracker.update(curr, curr2)
    assert np.all(status2 == 1)
    assert np.allclose((new2 - new_points).mean(axis=0), [2.0, 3.0], atol=0.15)


def test_tracker_class_init_validation():
    tracker = LucasKanadeTracker()
    with pytest.raises(ValueError, match="2D"):
        tracker.init(np.zeros((8, 8, 3)), np.array([[4.0, 4.0]]))


def test_small_window_still_tracks():
    prev, curr = _disc_pair(shift=(2, 3))
    points = _interior_points()
    new_points, status, _ = track_points(prev, curr, points, window_size=7)
    assert np.all(status == 1)
    assert np.allclose((new_points - points).mean(axis=0), [2.0, 3.0], atol=0.3)


def test_demo_runs():
    new_points, status, errors = demo_main()
    assert status[:8].sum() == 8
    assert (status[8:] == 0).sum() == 3
