import numpy as np
import pytest

from object_tracker import (
    ObjectTracker,
    bbox_centre,
    make_frame,
    make_sequence,
    mean_drift,
    ncc,
)


def _tracked_sequence(n_frames=15, velocity=(3.0, 2.0), noise=0.0, seed=0):
    frames, true_centres = make_sequence(
        n_frames=n_frames, start=(30.0, 30.0), velocity=velocity,
        radius=10, noise=noise, seed=seed)
    first = true_centres[0]
    bbox = (int(first[0]) - 10, int(first[1]) - 10, 21, 21)
    tracker = ObjectTracker(search_radius=15).init(frames[0], bbox)
    results = tracker.track_sequence(frames[1:])
    tracked = [bbox] + [b for b, _ in results]
    scores = [s for _, s in results]
    return tracked, scores, true_centres, tracker


def test_ncc_identical_patches_score_one():
    frame = make_frame()
    assert ncc(frame, frame) == pytest.approx(1.0)


def test_ncc_anticorrelated_patches_score_minus_one():
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = 1.0 - a
    assert ncc(a, b) == pytest.approx(-1.0)


def test_ncc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ncc(np.zeros((3, 3)), np.zeros((4, 4)))


def test_tracks_moving_disc_with_small_drift():
    tracked, scores, true_centres, _ = _tracked_sequence()
    assert mean_drift(tracked, true_centres) < 1.5
    assert min(scores) > 0.9


def test_tracks_noisy_sequence_with_small_drift():
    tracked, scores, true_centres, _ = _tracked_sequence(noise=5.0)
    assert mean_drift(tracked, true_centres) < 2.0
    assert min(scores) > 0.7


def test_static_object_has_zero_drift():
    tracked, _, true_centres, _ = _tracked_sequence(velocity=(0.0, 0.0))
    assert mean_drift(tracked, true_centres) == pytest.approx(0.0)


def test_track_returns_bbox_and_score():
    frames, true_centres = make_sequence(n_frames=3)
    first = true_centres[0]
    tracker = ObjectTracker().init(
        frames[0], (int(first[0]) - 10, int(first[1]) - 10, 21, 21))
    bbox, score = tracker.track(frames[1])
    assert len(bbox) == 4
    assert -1.0 <= score <= 1.0


def test_track_before_init_raises():
    with pytest.raises(ValueError):
        ObjectTracker().track(make_frame())


def test_init_out_of_bounds_bbox_raises():
    with pytest.raises(ValueError):
        ObjectTracker().init(make_frame(), (110, 150, 21, 21))


def test_init_constant_template_raises():
    frame = np.full((120, 160), 20.0)
    with pytest.raises(ValueError):
        ObjectTracker().init(frame, (10, 10, 21, 21))


def test_init_non_2d_frame_raises():
    with pytest.raises(ValueError):
        ObjectTracker().init(np.zeros((10, 10, 3)), (0, 0, 5, 5))


def test_lost_when_object_jumps_outside_window():
    frames, true_centres = make_sequence(n_frames=2, velocity=(40.0, 0.0))
    first = true_centres[0]
    tracker = ObjectTracker(search_radius=5, min_score=0.9).init(
        frames[0], (int(first[0]) - 10, int(first[1]) - 10, 21, 21))
    _, score = tracker.track(frames[1])
    assert tracker.is_lost


def test_track_sequence_length_matches_frames():
    frames, _ = make_sequence(n_frames=8)
    tracker = ObjectTracker().init(frames[0], (20, 20, 21, 21))
    results = tracker.track_sequence(frames[1:])
    assert len(results) == 7


def test_track_sequence_empty_raises():
    frames, _ = make_sequence(n_frames=2)
    tracker = ObjectTracker().init(frames[0], (20, 20, 21, 21))
    with pytest.raises(ValueError):
        tracker.track_sequence([])


def test_mean_drift_length_mismatch_raises():
    with pytest.raises(ValueError):
        mean_drift([(0, 0, 5, 5)], np.zeros((2, 2)))


def test_bbox_centre():
    assert np.allclose(bbox_centre((10, 20, 6, 8)), [12.5, 23.5])
    assert np.allclose(bbox_centre((20, 20, 21, 21)), [30.0, 30.0])


def test_make_frame_invalid_radius_raises():
    with pytest.raises(ValueError):
        make_frame(radius=0)


def test_invalid_search_radius_raises():
    with pytest.raises(ValueError):
        ObjectTracker(search_radius=-1)
