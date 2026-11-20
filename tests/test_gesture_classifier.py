import numpy as np
import pytest

from gesture_classifier import (
    GESTURES,
    GestureClassifier,
    accuracy,
    extract_features,
    hu_moments,
    radial_spread,
    solidity,
    make_circle,
    make_gesture,
    make_square,
    make_star,
    make_triangle,
)


def _dataset(rng, n_train=20, n_test=10):
    train_X, train_y, test_X, test_y = [], [], [], []
    for name in GESTURES:
        for _ in range(n_train):
            train_X.append(make_gesture(name, rng=rng))
            train_y.append(name)
        for _ in range(n_test):
            test_X.append(make_gesture(name, rng=rng))
            test_y.append(name)
    return train_X, np.array(train_y), test_X, np.array(test_y)


def test_hu_moments_have_seven_finite_values():
    feats = hu_moments(make_circle())
    assert feats.shape == (7,)
    assert np.isfinite(feats).all()


def test_hu_moments_translation_invariant():
    small = make_circle(size=64, radius=18, centre=(32.0, 32.0))
    shifted = make_circle(size=64, radius=18, centre=(40.0, 24.0))
    assert np.abs(hu_moments(small) - hu_moments(shifted)).max() < 0.75


def test_hu_moments_scale_invariant():
    small = make_circle(size=64, radius=12, centre=(32.0, 32.0))
    big = make_circle(size=64, radius=22, centre=(32.0, 32.0))
    assert np.abs(hu_moments(small) - hu_moments(big)).max() < 0.75


def test_hu_moments_rotation_invariant_for_square():
    upright = make_square(size=64, side=32, angle_deg=0.0)
    rotated = make_square(size=64, side=32, angle_deg=30.0)
    assert np.abs(hu_moments(upright) - hu_moments(rotated)).max() < 0.75


def test_different_gestures_have_different_features():
    diff = np.abs(hu_moments(make_circle()) - hu_moments(make_star()))
    assert diff.max() > 0.5


def test_classifier_accuracy_on_synthetic_gestures():
    rng = np.random.default_rng(5)
    train_X, train_y, test_X, test_y = _dataset(rng)
    clf = GestureClassifier(k=3).fit(extract_features(train_X), train_y)
    assert accuracy(clf, test_X, test_y) >= 0.85


def test_classify_single_image():
    rng = np.random.default_rng(9)
    train_X, train_y, _, _ = _dataset(rng)
    clf = GestureClassifier(k=3).fit(extract_features(train_X), train_y)
    assert clf.classify(make_circle()) == "circle"
    assert clf.classify(make_triangle()) == "triangle"


def test_predict_proba_rows_sum_to_one():
    rng = np.random.default_rng(5)
    train_X, train_y, test_X, _ = _dataset(rng)
    clf = GestureClassifier(k=3).fit(extract_features(train_X), train_y)
    proba = clf.predict_proba(extract_features(test_X))
    assert proba.shape == (len(test_X), len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_proba_top_class_matches_predict():
    rng = np.random.default_rng(5)
    train_X, train_y, test_X, _ = _dataset(rng)
    clf = GestureClassifier(k=3).fit(extract_features(train_X), train_y)
    feats = extract_features(test_X)
    top = clf.classes_[np.argmax(clf.predict_proba(feats), axis=1)]
    assert (top == clf.predict(feats)).all()


def test_make_gesture_unknown_class_raises():
    with pytest.raises(ValueError):
        make_gesture("hexagon")


def test_hu_moments_empty_image_raises():
    with pytest.raises(ValueError):
        hu_moments(np.zeros((64, 64)))


def test_hu_moments_non_2d_raises():
    with pytest.raises(ValueError):
        hu_moments(np.zeros((64, 64, 3)))


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        GestureClassifier().predict(np.zeros((1, 4)))


def test_fit_unknown_label_raises():
    with pytest.raises(ValueError):
        GestureClassifier().fit(np.zeros((3, 4)),
                                ["circle", "hexagon", "square"])


def test_extract_features_returns_three_columns():
    feats = extract_features([make_circle(), make_square()])
    assert feats.shape == (2, 3)
    assert np.isfinite(feats).all()


def test_solidity_near_one_for_convex_shapes():
    for maker in (make_circle, make_square, make_triangle):
        assert solidity(maker()) == pytest.approx(1.0, abs=0.02)


def test_solidity_well_below_one_for_star():
    assert solidity(make_star()) < 0.8


def test_radial_spread_orders_shapes():
    assert radial_spread(make_circle()) < 0.05
    assert 0.08 < radial_spread(make_square()) < 0.15
    assert 0.18 < radial_spread(make_triangle()) < 0.28
    assert radial_spread(make_star()) > 0.15


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        GestureClassifier(k=0)


def test_each_gesture_generator_produces_foreground():
    for maker in (make_circle, make_square, make_triangle, make_star):
        img = maker()
        assert img.shape == (64, 64)
        assert 0 < img.sum() < 64 * 64
