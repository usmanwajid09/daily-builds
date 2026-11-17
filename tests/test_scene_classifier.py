"""Tests for the scene classifier (colour histograms + kNN, numpy only)."""

import numpy as np
import pytest

from scene_classifier.scene_classifier import (
    SCENE_KINDS,
    SceneClassifier,
    extract_features,
    generate_scene,
    predict,
    train_classifier,
)


def make_dataset(n_train=6, n_test=3, seed_offset=0):
    """Small synthetic train/test split, fixed seeds."""
    X_train, y_train, X_test, y_test = [], [], [], []
    for kind in SCENE_KINDS:
        for s in range(seed_offset, seed_offset + n_train):
            X_train.append(extract_features(generate_scene(kind, seed=s)))
            y_train.append(kind)
        for s in range(100 + seed_offset, 100 + seed_offset + n_test):
            X_test.append(extract_features(generate_scene(kind, seed=s)))
            y_test.append(kind)
    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test),
    )


def test_extract_features_shape_and_normalisation():
    img = generate_scene("beach", seed=0)
    feats = extract_features(img, bins=8)
    assert feats.shape == (24,)
    assert feats.sum() == pytest.approx(1.0)
    assert (feats >= 0).all()


def test_extract_features_bins():
    img = generate_scene("forest", seed=1)
    assert extract_features(img, bins=4).shape == (12,)
    assert extract_features(img, bins=16).shape == (48,)


def test_extract_features_uint8_matches_float():
    img = generate_scene("city", seed=2)
    uint8 = (img * 255).astype(np.uint8)
    np.testing.assert_allclose(
        extract_features(img), extract_features(uint8), atol=0.02
    )


def test_generate_scene_shapes_and_kinds():
    for kind in SCENE_KINDS:
        img = generate_scene(kind, seed=3, size=32)
        assert img.shape == (32, 32, 3)
        assert img.min() >= 0.0 and img.max() <= 1.0
    # Same seed -> identical image; different seed -> different image.
    a = generate_scene("beach", seed=7)
    b = generate_scene("beach", seed=7)
    c = generate_scene("beach", seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generate_scene_colours_differ_by_kind():
    # Beach is blue/sandy, forest is green, city is grey: mean colours differ.
    means = {
        kind: generate_scene(kind, seed=0).mean(axis=(0, 1))
        for kind in SCENE_KINDS
    }
    assert means["beach"][2] > means["forest"][2]  # beach bluer than forest
    # Forest is green-dominant (G well above R); city's pale sky is near-grey.
    assert means["forest"][1] - means["forest"][0] > \
        means["city"][1] - means["city"][0]


def test_classifier_accuracy_on_synthetic_scenes():
    X_train, y_train, X_test, y_test = make_dataset()
    clf = SceneClassifier(k=3).fit(X_train, y_train)
    assert clf.score(X_test, y_test) == pytest.approx(1.0)


def test_classifier_predicts_raw_images():
    X_train, y_train, _, _ = make_dataset()
    clf = SceneClassifier(k=1).fit(X_train, y_train)
    img = generate_scene("forest", seed=200)
    assert clf.predict(img)[0] == "forest"


def test_predict_proba_rows_sum_to_one():
    X_train, y_train, X_test, _ = make_dataset()
    clf = SceneClassifier(k=3).fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_train_classifier_and_predict_helpers():
    X_train, y_train, X_test, y_test = make_dataset()
    clf = train_classifier(X_train, y_train, k=3)
    assert isinstance(clf, SceneClassifier)
    assert (predict(clf, X_test) == y_test).all()


def test_predict_before_fit_raises():
    clf = SceneClassifier(k=3)
    with pytest.raises(ValueError, match="not fitted"):
        clf.predict(np.zeros((1, 24)))


def test_invalid_image_shapes_raise():
    with pytest.raises(ValueError, match="3D"):
        extract_features(np.zeros((32, 32)))  # 2D, no channels
    with pytest.raises(ValueError, match="3 colour channels"):
        extract_features(np.zeros((32, 32, 4)))  # RGBA
    with pytest.raises(ValueError, match="numpy array"):
        extract_features([[0.5, 0.5, 0.5]])  # plain list


def test_invalid_image_values_raise():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        extract_features(np.full((8, 8, 3), 1.5))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        extract_features(np.full((8, 8, 3), -0.1))
    with pytest.raises(ValueError, match="finite"):
        extract_features(np.full((8, 8, 3), np.nan))


def test_invalid_bins_and_k_raise():
    img = generate_scene("beach", seed=0)
    with pytest.raises(ValueError, match="bins must be >= 1"):
        extract_features(img, bins=0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        SceneClassifier(k=0)
    X_train, y_train, _, _ = make_dataset()
    with pytest.raises(ValueError, match="larger than the number of training"):
        SceneClassifier(k=100).fit(X_train, y_train)


def test_unknown_scene_kind_raises():
    with pytest.raises(ValueError, match="unknown scene kind"):
        generate_scene("desert", seed=0)


def test_fit_mismatched_labels_raises():
    X_train, y_train, _, _ = make_dataset()
    with pytest.raises(ValueError, match="labels"):
        SceneClassifier(k=3).fit(X_train, y_train[:5])
