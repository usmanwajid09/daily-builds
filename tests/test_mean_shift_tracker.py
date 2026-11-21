"""Tests for mean_shift_tracker."""

import numpy as np
import pytest

from mean_shift_tracker import (
    MeanShiftTracker,
    color_histogram,
    moving_square_frame,
)
from mean_shift_tracker.demo import main as demo_main
from mean_shift_tracker.mean_shift_tracker import mean_shift_step


def test_histogram_sums_to_one():
    frame = moving_square_frame()
    hist = color_histogram(frame)
    assert hist.shape == (8 ** 3,)
    assert hist.sum() == pytest.approx(1.0)
    assert np.all(hist >= 0)


def test_histogram_distinguishes_colours():
    red = np.zeros((10, 10, 3))
    red[..., 0] = 220
    blue = np.zeros((10, 10, 3))
    blue[..., 2] = 220
    h_red = color_histogram(red)
    h_blue = color_histogram(blue)
    assert np.argmax(h_red) != np.argmax(h_blue)
    assert h_red[np.argmax(h_red)] == pytest.approx(1.0)
    assert h_blue[np.argmax(h_blue)] == pytest.approx(1.0)


def test_histogram_bins_parameter():
    frame = moving_square_frame()
    assert color_histogram(frame, bins=4).shape == (4 ** 3,)
    assert color_histogram(frame, bins=16).shape == (16 ** 3,)
    with pytest.raises(ValueError, match="bins"):
        color_histogram(frame, bins=1)


def test_histogram_rejects_bad_frames():
    with pytest.raises(ValueError, match="RGB"):
        color_histogram(np.zeros((10, 10)))
    with pytest.raises(ValueError, match="RGB"):
        color_histogram(np.zeros((10, 10, 4)))
    with pytest.raises(ValueError, match="empty"):
        color_histogram(np.zeros((0, 0, 3)))
    with pytest.raises(ValueError, match="finite"):
        color_histogram(np.full((4, 4, 3), np.nan))


def test_tracker_follows_moving_square():
    tracker = MeanShiftTracker()
    center = np.array([30.0, 40.0])
    step = np.array([2.0, 3.0])
    size = 13
    frame = moving_square_frame(center=tuple(center), size=size, seed=0)
    tracker.init(frame, (24.0, 34.0, size, size))
    for _ in range(5):
        center = center + step
        frame = moving_square_frame(center=tuple(center), size=size, seed=0)
        tracker.update(frame)
        error = float(np.hypot(*(np.array(tracker.center_) - center)))
        assert error < 1.5


def test_tracker_stays_put_on_static_frame():
    tracker = MeanShiftTracker()
    frame = moving_square_frame(center=(30, 40), size=13, seed=0)
    tracker.init(frame, (24.0, 34.0, 13, 13))
    before = tracker.center_
    tracker.update(frame)
    assert np.allclose(tracker.center_, before, atol=0.01)


def test_tracker_follows_negative_motion():
    tracker = MeanShiftTracker()
    center = np.array([40.0, 50.0])
    step = np.array([-2.0, -3.0])
    frame = moving_square_frame(center=tuple(center), size=13, seed=0)
    tracker.init(frame, (34.0, 44.0, 13, 13))
    for _ in range(3):
        center = center + step
        frame = moving_square_frame(center=tuple(center), size=13, seed=0)
        tracker.update(frame)
        error = float(np.hypot(*(np.array(tracker.center_) - center)))
        assert error < 1.5


def test_mean_shift_step_moves_toward_target():
    frame = moving_square_frame(center=(30, 40), size=13, seed=0)
    target = color_histogram(frame[24:37, 34:47], bins=8)
    # Start the window 4 px off-centre; one step must move it closer.
    new_y, new_x = mean_shift_step(frame, (34.0, 44.0), (13, 13), target)
    assert np.hypot(new_y - 30.0, new_x - 40.0) < np.hypot(34.0 - 30.0, 44.0 - 40.0)


def test_mean_shift_step_on_static_target_is_stable():
    frame = moving_square_frame(center=(30, 40), size=13, seed=0)
    target = color_histogram(frame[24:37, 34:47], bins=8)
    new_y, new_x = mean_shift_step(frame, (30.0, 40.0), (13, 13), target)
    assert np.hypot(new_y - 30.0, new_x - 40.0) < 1.0


def test_init_rejects_bad_bbox():
    tracker = MeanShiftTracker()
    frame = moving_square_frame()
    with pytest.raises(ValueError, match="bbox"):
        tracker.init(frame, (24.0, 34.0, 12))  # not 4 elements
    with pytest.raises(ValueError, match="positive"):
        tracker.init(frame, (24.0, 34.0, 0, 12))
    with pytest.raises(ValueError, match="does not fit"):
        tracker.init(frame, (55.0, 34.0, 12, 12))
    with pytest.raises(ValueError, match="does not fit"):
        tracker.init(frame, (-1.0, 34.0, 12, 12))


def test_update_before_init_raises():
    tracker = MeanShiftTracker()
    with pytest.raises(RuntimeError, match="not initialised"):
        tracker.update(moving_square_frame())
    with pytest.raises(RuntimeError, match="not initialised"):
        _ = tracker.bbox


def test_init_returns_bbox_and_sets_state():
    tracker = MeanShiftTracker()
    frame = moving_square_frame(center=(30, 40), size=13, seed=0)
    bbox = tracker.init(frame, (24.0, 34.0, 13, 13))
    assert bbox == (24.0, 34.0, 13, 13)
    assert tracker.center_ == (30.5, 40.5)
    assert tracker.size_ == (13, 13)
    assert tracker.target_hist_.sum() == pytest.approx(1.0)


def test_constructor_validation():
    with pytest.raises(ValueError, match="bins"):
        MeanShiftTracker(bins=1)
    with pytest.raises(ValueError, match="max_iterations"):
        MeanShiftTracker(max_iterations=0)
    with pytest.raises(ValueError, match="epsilon"):
        MeanShiftTracker(epsilon=0.0)


def test_moving_square_frame_validation():
    with pytest.raises(ValueError, match="shape"):
        moving_square_frame(shape=(0, 80))
    with pytest.raises(ValueError, match="size"):
        moving_square_frame(size=0)
    with pytest.raises(ValueError, match="does not fit"):
        moving_square_frame(center=(2, 2), size=13)
    frame = moving_square_frame()
    assert frame.shape == (60, 80, 3)
    assert frame.min() >= 0 and frame.max() <= 255


def test_update_rejects_grayscale_frame():
    tracker = MeanShiftTracker()
    tracker.init(moving_square_frame(), (24.0, 34.0, 13, 13))
    with pytest.raises(ValueError, match="RGB"):
        tracker.update(np.zeros((60, 80)))


def test_demo_runs():
    max_error = demo_main()
    assert max_error < 1.5
