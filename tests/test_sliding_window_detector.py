"""Tests for sliding_window_detector."""
import numpy as np
import pytest

from sliding_window_detector.detector import (
    LogisticClassifier,
    SlidingWindowDetector,
    feature_length,
    hog_features,
    iou,
    make_circle_window,
    make_negative_window,
    make_scene,
    non_max_suppression,
)


def _trained_detector(seed=0):
    rng = np.random.default_rng(seed)
    positives = []
    for _ in range(60):
        cy = 8 + int(rng.integers(-2, 3))
        cx = 8 + int(rng.integers(-2, 3))
        positives.append(make_circle_window(16, centre=(cy, cx),
                                            radius=int(rng.integers(4, 6)), rng=rng))
    negatives = [make_negative_window(16, kind=k, rng=rng)
                 for k in ("blank", "noise", "square") for _ in range(20)]
    det = SlidingWindowDetector(window_size=16, stride=4, threshold=0.7)
    det.fit(positives, negatives)
    return det


# --- HOG features ---------------------------------------------------------
def test_hog_feature_length():
    feats = hog_features(np.zeros((16, 16)))
    assert len(feats) == feature_length((16, 16))
    assert len(feats) == 324  # 3x3 blocks * 4 * 9 bins


def test_hog_blank_image_is_zero():
    feats = hog_features(np.zeros((16, 16)))
    assert np.all(feats == 0)


def test_hog_vertical_vs_horizontal_differ():
    v = np.zeros((16, 16))
    v[:, 7:9] = 1.0
    h = np.zeros((16, 16))
    h[7:9, :] = 1.0
    assert np.linalg.norm(hog_features(v) - hog_features(h)) > 0.5


def test_hog_rejects_bad_input():
    with pytest.raises(ValueError):
        hog_features(np.zeros((8, 8, 3)))
    with pytest.raises(ValueError):
        hog_features(np.zeros((0, 8)))
    with pytest.raises(ValueError):
        hog_features(np.zeros((2, 2)), cell_size=4)
    with pytest.raises(ValueError):
        hog_features(np.zeros((8, 8)), n_bins=0)


def test_hog_features_are_nonnegative_and_bounded():
    rng = np.random.default_rng(0)
    feats = hog_features(rng.random((16, 16)))
    assert np.all(feats >= 0)
    assert np.all(feats <= 1.0 + 1e-9)


# --- classifier -----------------------------------------------------------
def test_logistic_classifier_separates():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, size=(20, 4)),
                   rng.normal(2, 0.5, size=(20, 4))])
    y = np.array([0] * 20 + [1] * 20)
    clf = LogisticClassifier(4, lr=0.5, epochs=200, seed=0)
    clf.fit(X, y)
    assert (clf.predict(X) == y).mean() >= 0.95


def test_logistic_classifier_validates():
    clf = LogisticClassifier(4)
    with pytest.raises(ValueError):
        clf.decision_scores(np.zeros(4))  # not fitted yet
    with pytest.raises(ValueError):
        clf.fit(np.zeros((5, 3)), np.zeros(5, dtype=int))
    with pytest.raises(ValueError):
        LogisticClassifier(0)
    with pytest.raises(ValueError):
        LogisticClassifier(4, lr=-1)


# --- IoU / NMS ------------------------------------------------------------
def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_iou_partial_overlap():
    assert abs(iou((0, 0, 10, 10), (5, 0, 10, 10)) - 1 / 3) < 1e-9


def test_iou_rejects_degenerate_box():
    with pytest.raises(ValueError):
        iou((0, 0, 0, 10), (0, 0, 10, 10))


def test_nms_keeps_best_and_drops_overlap():
    dets = [
        {"x": 0, "y": 0, "w": 16, "h": 16, "score": 0.9},
        {"x": 2, "y": 2, "w": 16, "h": 16, "score": 0.7},
        {"x": 40, "y": 40, "w": 16, "h": 16, "score": 0.8},
    ]
    kept = non_max_suppression(dets, iou_threshold=0.3)
    assert len(kept) == 2
    assert kept[0]["score"] == 0.9
    assert kept[1]["score"] == 0.8


def test_nms_empty_input():
    assert non_max_suppression([]) == []


def test_nms_rejects_bad_threshold():
    with pytest.raises(ValueError):
        non_max_suppression([{"x": 0, "y": 0, "w": 4, "h": 4, "score": 0.5}],
                            iou_threshold=1.5)


# --- detector -------------------------------------------------------------
def test_detector_finds_both_circles():
    det = _trained_detector()
    scene = make_scene(size=48, circles=((16, 16), (34, 32)), radius=5)
    detections = det.detect(scene)
    assert len(detections) == 2
    for d in detections:
        assert d["score"] >= 0.5
        assert set(d) == {"x", "y", "w", "h", "score"}


def test_detector_blank_scene_is_empty():
    det = _trained_detector()
    assert det.detect(np.zeros((48, 48))) == []


def test_detector_requires_fit():
    det = SlidingWindowDetector()
    with pytest.raises(ValueError):
        det.detect(np.zeros((48, 48)))


def test_detector_validates_image():
    det = _trained_detector()
    with pytest.raises(ValueError):
        det.detect(np.zeros((10, 10)))  # smaller than the window
    with pytest.raises(ValueError):
        det.detect(np.zeros((48, 48, 3)))  # not grayscale
    with pytest.raises(ValueError):
        det.detect(np.array([]))


def test_detector_validates_fit_inputs():
    det = SlidingWindowDetector()
    with pytest.raises(ValueError):
        det.fit([], [np.zeros((16, 16))])
    with pytest.raises(ValueError):
        det.fit([np.zeros((16, 16))], [])
    with pytest.raises(ValueError):
        det.fit([np.zeros((8, 8))], [np.zeros((16, 16))])


def test_detector_rejects_bad_params():
    with pytest.raises(ValueError):
        SlidingWindowDetector(window_size=0)
    with pytest.raises(ValueError):
        SlidingWindowDetector(stride=0)
    with pytest.raises(ValueError):
        SlidingWindowDetector(threshold=1.0)


# --- synthetic data -------------------------------------------------------
def test_make_scene_contains_circles():
    scene = make_scene(size=32, circles=((16, 16),), radius=4)
    assert scene.shape == (32, 32)
    assert scene[16, 16] == 1.0
    assert scene[0, 0] == 0.0


def test_make_negative_window_kinds():
    for kind in ("blank", "noise", "square"):
        assert make_negative_window(16, kind=kind).shape == (16, 16)
    with pytest.raises(ValueError):
        make_negative_window(16, kind="triangle")


def test_make_circle_window_validation():
    with pytest.raises(ValueError):
        make_circle_window(size=16, radius=0)
