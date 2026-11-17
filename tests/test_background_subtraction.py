"""Tests for background_subtraction."""

import numpy as np
import pytest

from background_subtraction import (
    BackgroundSubtractor,
    foreground_mask,
    running_average_update,
)
from background_subtraction.demo import make_video


def _static_frames(n=5, value=50.0, shape=(20, 30)):
    return [np.full(shape, value) for _ in range(n)]


def test_first_frame_gives_empty_mask():
    subtractor = BackgroundSubtractor()
    mask = subtractor.apply(np.full((20, 30), 50.0))
    assert mask.dtype == bool
    assert not mask.any()


def test_static_frames_stay_empty():
    subtractor = BackgroundSubtractor()
    for frame in _static_frames():
        mask = subtractor.apply(frame)
    assert not mask.any()


def test_new_object_is_detected():
    subtractor = BackgroundSubtractor(learning_rate=0.1, threshold=25.0)
    for frame in _static_frames():
        subtractor.apply(frame)
    frame = np.full((20, 30), 50.0)
    frame[5:15, 10:20] = 200.0  # 10x10 bright square
    mask = subtractor.apply(frame)
    # The whole square is detected (interior pixels have >= 2 neighbours).
    assert mask[5:15, 10:20].all()
    assert not mask[0:5, :].any()
    assert int(mask.sum()) == 100


def test_object_leaving_fades_away():
    subtractor = BackgroundSubtractor(learning_rate=0.5, threshold=25.0)
    for frame in _static_frames():
        subtractor.apply(frame)
    frame = np.full((20, 30), 50.0)
    frame[5:15, 10:20] = 200.0
    subtractor.apply(frame)
    empty = np.full((20, 30), 50.0)
    for _ in range(5):
        mask = subtractor.apply(empty)
    assert not mask.any()


def test_small_change_below_threshold_ignored():
    subtractor = BackgroundSubtractor(threshold=25.0)
    for frame in _static_frames():
        subtractor.apply(frame)
    mask = subtractor.apply(np.full((20, 30), 60.0))  # diff 10 < 25
    assert not mask.any()


def test_isolated_pixel_is_cleaned_up():
    subtractor = BackgroundSubtractor(threshold=25.0, min_neighbors=2)
    for frame in _static_frames():
        subtractor.apply(frame)
    frame = np.full((20, 30), 50.0)
    frame[10, 15] = 200.0  # single bright pixel: sensor noise
    mask = subtractor.apply(frame)
    assert not mask.any()


def test_constructor_validation():
    with pytest.raises(ValueError):
        BackgroundSubtractor(learning_rate=0)
    with pytest.raises(ValueError):
        BackgroundSubtractor(learning_rate=1.5)
    with pytest.raises(ValueError):
        BackgroundSubtractor(learning_rate=-0.1)
    with pytest.raises(ValueError):
        BackgroundSubtractor(threshold=-1)
    with pytest.raises(ValueError):
        BackgroundSubtractor(min_neighbors=9)
    with pytest.raises(ValueError):
        BackgroundSubtractor(min_neighbors=-1)


def test_frame_validation_and_shape_mismatch():
    subtractor = BackgroundSubtractor()
    with pytest.raises(ValueError):
        subtractor.apply(np.zeros(10))
    with pytest.raises(ValueError):
        subtractor.apply(np.zeros((5, 5, 3)))
    with pytest.raises(ValueError):
        subtractor.apply(np.zeros((0, 5)))
    subtractor.apply(np.full((20, 30), 50.0))
    with pytest.raises(ValueError):
        subtractor.apply(np.full((10, 30), 50.0))


def test_get_background_and_reset():
    subtractor = BackgroundSubtractor()
    with pytest.raises(ValueError):
        subtractor.get_background()
    subtractor.apply(np.full((20, 30), 50.0))
    bg = subtractor.get_background()
    assert bg.shape == (20, 30)
    assert np.allclose(bg, 50.0)
    subtractor.reset()
    with pytest.raises(ValueError):
        subtractor.get_background()
    mask = subtractor.apply(np.full((20, 30), 80.0))
    assert not mask.any()
    assert np.allclose(subtractor.get_background(), 80.0)


def test_running_average_update_math():
    bg = np.full((2, 2), 100.0)
    updated = running_average_update(bg, np.full((2, 2), 200.0), 0.25)
    assert np.allclose(updated, 125.0)
    with pytest.raises(ValueError):
        running_average_update(bg, np.zeros((3, 3)), 0.5)
    with pytest.raises(ValueError):
        running_average_update(bg, np.zeros((2, 2)), 0)


def test_foreground_mask_function():
    bg = np.full((10, 10), 50.0)
    frame = np.full((10, 10), 50.0)
    frame[2:5, 2:5] = 200.0
    mask = foreground_mask(frame, bg, threshold=25.0)
    assert int(mask.sum()) == 9
    assert foreground_mask(frame, bg, threshold=200.0).sum() == 0
    with pytest.raises(ValueError):
        foreground_mask(frame, bg, threshold=-1)
    with pytest.raises(ValueError):
        foreground_mask(frame, np.zeros((5, 5)), threshold=25.0)


def test_demo_video_detection():
    frames = make_video()
    subtractor = BackgroundSubtractor(learning_rate=0.1, threshold=25.0)
    counts = [int(subtractor.apply(f).sum()) for f in frames]
    assert counts[:5] == [0, 0, 0, 0, 0]
    assert counts[5] == 100
    assert counts[6] == 100
