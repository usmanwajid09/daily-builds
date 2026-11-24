import numpy as np
import pytest

from camshift_tracker.camshift import (
    backproject,
    build_histogram,
    camshift_track,
    mean_shift,
    quantise_colour,
    track_sequence,
)
from camshift_tracker.demo import make_frames


def _frame_with_disc(height=80, width=100, centre=(50, 40), radius=10):
    """Green background, solid red disc."""
    frame = np.zeros((height, width, 3), dtype=np.float64)
    frame[:, :, 1] = 120.0
    rows, cols = np.mgrid[0:height, 0:width]
    disc = (rows - centre[1]) ** 2 + (cols - centre[0]) ** 2 <= radius ** 2
    frame[disc] = [220.0, 30.0, 30.0]
    return frame


def _red_histogram(frame=None):
    frame = frame if frame is not None else _frame_with_disc()
    return build_histogram(frame, window=(45, 35, 10, 10), bins=16)


# --- quantise_colour ------------------------------------------------------


def test_quantise_black_is_zero_and_white_is_top_bin():
    frame = np.zeros((2, 2, 3))
    assert np.all(quantise_colour(frame, 16) == 0)
    frame = np.full((2, 2, 3), 255.0)
    assert np.all(quantise_colour(frame, 16) == 15)


def test_quantise_bins_out_of_range_raise():
    frame = np.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="bins"):
        quantise_colour(frame, 1)
    with pytest.raises(ValueError, match="bins"):
        quantise_colour(frame, 65)


# --- build_histogram ------------------------------------------------------


def test_histogram_sums_to_one_and_has_shape():
    hist = _red_histogram()
    assert hist.shape == (16, 16, 16)
    assert hist.sum() == pytest.approx(1.0)


def test_histogram_of_pure_region_is_a_single_bin():
    hist = _red_histogram()
    assert hist.max() == pytest.approx(1.0)
    assert np.count_nonzero(hist) == 1


def test_histogram_window_outside_frame_raises():
    frame = _frame_with_disc()
    with pytest.raises(ValueError, match="outside"):
        build_histogram(frame, window=(90, 70, 20, 20), bins=16)
    with pytest.raises(ValueError, match="positive"):
        build_histogram(frame, window=(0, 0, 0, 10), bins=16)


# --- backproject ----------------------------------------------------------


def test_backproject_red_pixel_high_green_low():
    frame = _frame_with_disc()
    prob = backproject(frame, _red_histogram(frame))
    assert prob[40, 50] == pytest.approx(1.0)  # centre of the red disc
    assert prob[0, 0] == pytest.approx(0.0)  # green background


def test_backproject_returns_probability_range():
    prob = backproject(_frame_with_disc(), _red_histogram())
    assert prob.min() >= 0.0
    assert prob.max() <= 1.0


def test_backproject_bad_histogram_raises():
    frame = _frame_with_disc()
    with pytest.raises(ValueError):
        backproject(frame, np.zeros((4, 4)))
    with pytest.raises(ValueError):
        backproject(frame, np.full((4, 4, 4), -1.0))


# --- mean_shift -----------------------------------------------------------


def test_mean_shift_moves_window_toward_blob():
    prob = np.zeros((40, 40))
    rows, cols = np.mgrid[0:40, 0:40]
    prob += np.exp(-((rows - 25) ** 2 + (cols - 25) ** 2) / (2 * 3.0 ** 2))
    window = mean_shift(prob, (10, 10, 10, 10), max_iter=20)
    cx, cy = window[0] + window[2] / 2, window[1] + window[3] / 2
    assert abs(cx - 25) <= 2
    assert abs(cy - 25) <= 2


def test_mean_shift_no_mass_returns_window_unchanged():
    prob = np.zeros((20, 20))
    assert mean_shift(prob, (5, 5, 5, 5)) == (5, 5, 5, 5)


def test_mean_shift_invalid_args_raise():
    prob = np.zeros((20, 20))
    with pytest.raises(ValueError):
        mean_shift(prob, (5, 5, 5, 5), max_iter=0)
    with pytest.raises(ValueError):
        mean_shift(prob, (5, 5, 5, 5), epsilon=0)
    with pytest.raises(ValueError):
        mean_shift(np.array([1.0, 2.0]), (0, 0, 1, 1))


# --- camshift_track / track_sequence --------------------------------------


def test_camshift_follows_disc_from_offset_window():
    frame = _frame_with_disc()
    hist = _red_histogram(frame)
    window, mass = camshift_track(frame, (35, 25, 20, 20), hist)
    cx, cy = window[0] + window[2] / 2, window[1] + window[3] / 2
    assert abs(cx - 50) <= 3
    assert abs(cy - 40) <= 3
    assert mass > 0


def test_camshift_returns_mass_zero_when_target_absent():
    frame = np.zeros((40, 40, 3))
    frame[:, :, 1] = 120.0  # all green, no red anywhere
    hist = _red_histogram()
    window, mass = camshift_track(frame, (10, 10, 10, 10), hist)
    assert mass == 0.0
    assert window == (10, 10, 10, 10)


def test_track_sequence_follows_moving_disc():
    frames, centres = make_frames(n_frames=8)
    hist = build_histogram(frames[0], (20, 15, 10, 10), bins=16)
    windows = track_sequence(frames, (20, 15, 10, 10), hist)
    assert len(windows) == 8
    final = windows[-1]
    fcx, fcy = final[0] + final[2] / 2, final[1] + final[3] / 2
    tx, ty = centres[-1]
    assert np.hypot(fcx - tx, fcy - ty) <= 5.0


def test_track_sequence_empty_frames_raises():
    with pytest.raises(ValueError, match="non-empty"):
        track_sequence([], (0, 0, 5, 5), np.ones((2, 2, 2)))


def test_invalid_frame_raises():
    with pytest.raises(ValueError):
        build_histogram(np.zeros((10, 10)), window=(0, 0, 5, 5))
    with pytest.raises(ValueError):
        build_histogram(np.full((10, 10, 3), 300.0), window=(0, 0, 5, 5))
