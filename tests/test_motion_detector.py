"""Tests for motion_detector."""

import numpy as np
import pytest

from motion_detector import (
    MotionDetector,
    connected_components,
    detect_motion,
)

SHAPE = (64, 64)
BG = 0.2
SQ = 1.0
SIZE = 10


def _background(shape=SHAPE, value=BG):
    return np.full(shape, value)


def _frame_with_square(x0, y0=27, size=SIZE, value=SQ, shape=SHAPE):
    frame = _background(shape)
    frame[y0:y0 + size, x0:x0 + size] = value
    return frame


def _square_centre(x0, y0=27, size=SIZE):
    return (y0 + (size - 1) / 2, x0 + (size - 1) / 2)


def _warm_up(detector, n=5):
    for _ in range(n):
        detector.detect(_background())


def test_first_frame_reports_no_motion():
    detector = MotionDetector()
    result = detector.detect(_background())
    assert result["bounding_boxes"] == []
    assert result["motion_pixels"] == 0
    assert result["motion_fraction"] == 0.0


def test_static_frames_give_no_boxes_after_warm_up():
    detector = MotionDetector()
    for _ in range(10):
        result = detector.detect(_background())
    assert result["bounding_boxes"] == []
    assert result["motion_pixels"] == 0


def test_moving_square_detected_with_exactly_one_box():
    detector = MotionDetector()
    _warm_up(detector)
    result = detector.detect(_frame_with_square(8))
    assert len(result["bounding_boxes"]) == 1
    assert result["bounding_boxes"][0] == (27, 8, 36, 17)
    # The pure function agrees with the class, and the labelling
    # underneath finds one component of exactly the square's area.
    pure = detect_motion(_frame_with_square(8), _background())
    assert pure["bounding_boxes"] == result["bounding_boxes"]
    comps = connected_components(pure["mask"])
    assert len(comps) == 1
    assert comps[0]["area"] == 100


def test_box_centre_tracks_true_square_position():
    # Slow background update: the square overlaps its previous
    # position, and a fast update would leave a detectable "ghost".
    detector = MotionDetector(learning_rate=0.01)
    _warm_up(detector)
    for k, x0 in enumerate([8, 12, 16, 20, 24]):
        result = detector.detect(_frame_with_square(x0))
        assert len(result["bounding_boxes"]) == 1
        y_min, x_min, y_max, x_max = result["bounding_boxes"][0]
        centre = ((y_min + y_max) / 2, (x_min + x_max) / 2)
        true_y, true_x = _square_centre(x0)
        assert abs(centre[0] - true_y) <= 2
        assert abs(centre[1] - true_x) <= 2


def test_motion_pixel_count_matches_square_area_uint8_input():
    detector = MotionDetector()
    _warm_up(detector)
    # uint8 frames (0-255 scale) must be accepted and normalised.
    frame = (np.full(SHAPE, 51, dtype=np.uint8))
    frame[27:37, 8:18] = 255
    result = detector.detect(frame)
    assert len(result["bounding_boxes"]) == 1
    assert 50 <= result["motion_pixels"] <= 150  # true area is 100


def test_small_noise_filtered_by_min_area():
    detector = MotionDetector(min_area=20)
    _warm_up(detector)
    frame = _background()
    rng = np.random.default_rng(0)
    ys = rng.integers(0, SHAPE[0], size=5)
    xs = rng.integers(0, SHAPE[1], size=5)
    frame[ys, xs] = SQ  # 5 isolated single pixels
    result = detector.detect(frame)
    assert result["bounding_boxes"] == []
    assert result["motion_pixels"] == 5


def test_threshold_effect_on_subtle_change():
    # A square only 0.05 brighter than the background.
    subtle = _frame_with_square(8, value=BG + 0.05)
    high = MotionDetector(threshold=0.1)
    _warm_up(high)
    assert high.detect(subtle)["bounding_boxes"] == []
    low = MotionDetector(threshold=0.02)
    _warm_up(low)
    assert len(low.detect(subtle)["bounding_boxes"]) == 1


def test_learning_rate_adapts_to_permanent_object():
    detector = MotionDetector(learning_rate=0.05)
    _warm_up(detector)
    moved = _frame_with_square(8)
    first = detector.detect(moved)
    assert len(first["bounding_boxes"]) == 1
    # The square stays put: the background absorbs it and motion stops.
    for _ in range(150):
        result = detector.detect(moved)
    assert result["bounding_boxes"] == []
    assert result["motion_pixels"] == 0


def test_1d_frame_raises():
    detector = MotionDetector()
    with pytest.raises(ValueError):
        detector.detect(np.zeros(64))


def test_empty_frame_raises():
    detector = MotionDetector()
    with pytest.raises(ValueError):
        detector.detect(np.zeros((0, 0)))


def test_nan_frame_raises():
    detector = MotionDetector()
    frame = _background()
    frame[10, 10] = np.nan
    with pytest.raises(ValueError):
        detector.detect(frame)


def test_shape_change_between_frames_raises():
    detector = MotionDetector()
    detector.detect(_background())
    with pytest.raises(ValueError):
        detector.detect(np.full((32, 32), BG))


def test_negative_threshold_raises():
    with pytest.raises(ValueError):
        MotionDetector(threshold=-0.1)


def test_learning_rate_outside_0_1_raises():
    with pytest.raises(ValueError):
        MotionDetector(learning_rate=-0.1)
    with pytest.raises(ValueError):
        MotionDetector(learning_rate=1.5)


def test_min_area_below_1_raises():
    with pytest.raises(ValueError):
        MotionDetector(min_area=0)


def test_mask_shape_and_boolean_dtype():
    detector = MotionDetector()
    _warm_up(detector)
    result = detector.detect(_frame_with_square(8))
    assert result["mask"].shape == SHAPE
    assert result["mask"].dtype == bool
    assert result["mask"].sum() == result["motion_pixels"]
    assert 0.0 <= result["motion_fraction"] <= 1.0


def test_reset_clears_background():
    detector = MotionDetector()
    _warm_up(detector)
    detector.detect(_frame_with_square(8))
    detector.reset()
    result = detector.detect(_background())
    assert result["bounding_boxes"] == []
    assert result["motion_pixels"] == 0
    # After reset the next frame is the new background, so a square
    # placed on it is detected again from a clean slate.
    result = detector.detect(_frame_with_square(8))
    assert len(result["bounding_boxes"]) == 1


def test_determinism():
    def run():
        detector = MotionDetector()
        out = []
        for frame in [_background()] * 5 + [_frame_with_square(x) for x in (8, 12, 16)]:
            out.append(detector.detect(frame))
        return out

    first, second = run(), run()
    for a, b in zip(first, second):
        assert a["bounding_boxes"] == b["bounding_boxes"]
        assert a["motion_pixels"] == b["motion_pixels"]
        assert np.array_equal(a["mask"], b["mask"])
