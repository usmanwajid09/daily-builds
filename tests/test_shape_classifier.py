import numpy as np
import pytest

from shape_classifier import (
    CLASSES,
    ShapeKNN,
    accuracy,
    extract_features,
    hu_moments,
    make_circle,
    make_shape,
    make_square,
    make_star,
    make_triangle,
)


def _dataset(rng, n_train=20, n_test=10):
    train_X, train_y, test_X, test_y = [], [], [], []
    for name in CLASSES:
        for _ in range(n_train):
            train_X.append(make_shape(name, rng=rng))
            train_y.append(name)
        for _ in range(n_test):
            test_X.append(make_shape(name, rng=rng))
            test_y.append(name)
    return train_X, np.array(train_y), test_X, np.array(test_y)


def test_hu_moments_have_seven_finite_values():
    feats = hu_moments(make_circle())
    assert feats.shape == (7,)
    assert np.isfinite(feats).all()


def test_hu_moments_translation_invariant():
    small = make_circle(size=64, radius=18, centre=(32.0, 32.0))
    shifted = make_circle(size=64, radius=18, centre=(40.0, 24.0))
    diff = np.abs(hu_moments(small) - hu_moments(shifted))
    assert diff.max() < 0.75


def test_hu_moments_scale_invariant():
    small = make_circle(size=64, radius=12, centre=(32.0, 32.0))
    big = make_circle(size=64, radius=22, centre=(32.0, 32.0))
    diff = np.abs(hu_moments(small) - hu_moments(big))
    assert diff.max() < 0.75


def test_hu_moments_rotation_invariant_for_square():
    upright = make_square(size=64, side=32, angle_deg=0.0)
    rotated = make_square(size=64, side=32, angle_deg=30.0)
    diff = np.abs(hu_moments(upright) - hu_moments(rotated))
    assert diff.max() < 0.75


def test_different_shapes_have_different_features():
    diff = np.abs(hu_moments(make_circle()) - hu_moments(make_star()))
    assert diff.max() > 0.5


def test_classifier_accuracy_on_synthetic_shapes():
    rng = np.random.default_rng(5)
    train_X, train_y, test_X, test_y = _dataset(rng)
    clf = ShapeKNN(k=3).fit(extract_features(train_X), train_y)
    assert accuracy(clf, test_X, test_y) >= 0.85


def test_make_shape_unknown_class_raises():
    with pytest.raises(ValueError):
        make_shape("hexagon")


def test_hu_moments_empty_image_raises():
    with pytest.raises(ValueError):
        hu_moments(np.zeros((64, 64)))


def test_hu_moments_non_2d_raises():
    with pytest.raises(ValueError):
        hu_moments(np.zeros((64, 64, 3)))


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        ShapeKNN().predict(np.zeros((1, 4)))


def test_extract_features_uses_first_four_invariants():
    feats = extract_features([make_circle(), make_square()])
    assert feats.shape == (2, 4)
    assert np.isfinite(feats).all()


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        ShapeKNN(k=0)


def test_each_shape_generator_produces_foreground():
    for maker in (make_circle, make_square, make_triangle, make_star):
        img = maker()
        assert img.shape == (64, 64)
        assert 0 < img.sum() < 64 * 64
