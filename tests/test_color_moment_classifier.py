"""Tests for the colour-moment classifier (colour moments + kNN, numpy only)."""

import numpy as np
import pytest

from color_moment_classifier.moments import (
    SCENE_KINDS,
    ColorMomentClassifier,
    color_moments,
    extract_features,
    generate_scene,
    predict,
    train_classifier,
)


def _solid(color, shape=(32, 32)):
    return np.tile(np.asarray(color, dtype=np.float64), (*shape, 1))


def _make_dataset(n_train=6, n_test=3):
    """Small synthetic train/test split, fixed seeds."""
    X_train, y_train, X_test, y_test = [], [], [], []
    for kind in SCENE_KINDS:
        for s in range(n_train):
            X_train.append(extract_features(generate_scene(kind, seed=s)))
            y_train.append(kind)
        for s in range(100, 100 + n_test):
            X_test.append(extract_features(generate_scene(kind, seed=s)))
            y_test.append(kind)
    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test),
    )


def test_color_moments_of_solid_colour_image():
    # Solid colour: mean = the colour, std = 0, skew = 0 in every channel.
    feats = color_moments(_solid((0.8, 0.4, 0.2)))
    assert feats == pytest.approx(
        [0.8, 0.0, 0.0, 0.4, 0.0, 0.0, 0.2, 0.0, 0.0]
    )


def test_color_moments_shape_is_nine():
    feats = color_moments(generate_scene("forest", seed=0))
    assert feats.shape == (9,)
    assert np.isfinite(feats).all()


def test_channel_means_match_known_values():
    # Half the pixels red (1, 0, 0), half blue (0, 0, 1): channel means
    # are R=0.5, G=0.0, B=0.5.
    img = np.zeros((8, 8, 3))
    img[:4] = [1.0, 0.0, 0.0]
    img[4:] = [0.0, 0.0, 1.0]
    feats = color_moments(img)
    assert feats[0] == pytest.approx(0.5)  # R mean
    assert feats[3] == pytest.approx(0.0)  # G mean
    assert feats[6] == pytest.approx(0.5)  # B mean
    # Each half/half channel has std 0.5 and zero skew (symmetric).
    assert feats[1] == pytest.approx(0.5)  # R std
    assert feats[2] == pytest.approx(0.0)  # R skew


def test_generate_scene_shape_and_range():
    for kind in SCENE_KINDS:
        img = generate_scene(kind, size=32, seed=3)
        assert img.shape == (32, 32, 3)
        assert img.min() >= 0.0 and img.max() <= 1.0
    # Same seed -> identical image; different seed -> different image.
    a = generate_scene("sunset", seed=7)
    b = generate_scene("sunset", seed=7)
    c = generate_scene("sunset", seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generate_scene_dominant_channel_per_kind():
    sunset = generate_scene("sunset", seed=0).mean(axis=(0, 1))
    forest = generate_scene("forest", seed=0).mean(axis=(0, 1))
    ocean = generate_scene("ocean", seed=0).mean(axis=(0, 1))
    assert sunset[0] > sunset[1] > sunset[2]  # warm red/orange dominant
    assert forest[1] > forest[0] and forest[1] > forest[2]  # green dominant
    assert ocean[2] > ocean[1] > ocean[0]  # blue dominant


def test_generate_scene_unknown_kind_raises():
    with pytest.raises(ValueError):
        generate_scene("desert", seed=0)


def test_classifier_separates_three_kinds():
    X_train, y_train, X_test, y_test = _make_dataset()
    clf = ColorMomentClassifier(k=3).fit(X_train, y_train)
    assert clf.score(X_test, y_test) >= 0.95


def test_k_equals_one_works():
    X_train, y_train, X_test, y_test = _make_dataset()
    clf = ColorMomentClassifier(k=1).fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert preds.shape == (len(y_test),)
    assert set(preds) <= set(SCENE_KINDS)
    assert clf.score(X_test, y_test) >= 0.95


def test_predict_before_fit_raises():
    clf = ColorMomentClassifier(k=3)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((2, 9)))


def test_invalid_k_raises():
    X_train, y_train, _, _ = _make_dataset()
    with pytest.raises(ValueError):
        ColorMomentClassifier(k=0)
    # k larger than the number of training samples.
    with pytest.raises(ValueError):
        ColorMomentClassifier(k=len(y_train) + 1).fit(X_train, y_train)


def test_fit_mismatched_lengths_raises():
    X_train, y_train, _, _ = _make_dataset()
    with pytest.raises(ValueError):
        ColorMomentClassifier(k=3).fit(X_train, y_train[:-1])


def test_score_returns_float_in_zero_to_one():
    X_train, y_train, X_test, y_test = _make_dataset()
    clf = ColorMomentClassifier(k=3).fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_sunset_and_ocean_features_differ():
    sunset = extract_features(generate_scene("sunset", seed=0))
    ocean = extract_features(generate_scene("ocean", seed=0))
    assert not np.allclose(sunset, ocean)
    # Sunset is red-dominant, ocean is blue-dominant in the means.
    assert sunset[0] > ocean[0]
    assert ocean[6] > sunset[6]
    # The convenience helpers agree with the classifier path.
    images = [generate_scene(kind, seed=5) for kind in SCENE_KINDS]
    clf = train_classifier(images, list(SCENE_KINDS), k=1)
    assert list(predict(images, clf)) == list(SCENE_KINDS)
