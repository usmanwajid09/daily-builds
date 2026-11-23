import numpy as np
import pytest

from traffic_sign_classifier import (
    CLASSES,
    TrafficSignClassifier,
    accuracy,
    extract_features,
    make_dataset,
    make_sign,
)


def _trained(k=3, n_per_class=12, seed=0):
    images, labels = make_dataset(n_per_class, seed=seed)
    clf = TrafficSignClassifier(k=k)
    clf.fit(images, labels)
    return clf, images, labels


def test_all_classes_generate_correct_shape_and_dtype():
    assert len(CLASSES) == 6
    for sign_type in CLASSES:
        img = make_sign(sign_type, rng=42)
        assert img.shape == (32, 32, 3)
        assert img.dtype == np.uint8
        assert img.min() >= 0 and img.max() <= 255


def test_make_sign_is_deterministic_for_same_seed():
    a = make_sign("stop", rng=7)
    b = make_sign("stop", rng=7)
    assert np.array_equal(a, b)


def test_make_sign_variants_differ_with_different_seeds():
    a = make_sign("stop", rng=1)
    b = make_sign("stop", rng=2)
    assert not np.array_equal(a, b)


def test_each_class_has_its_dominant_colour():
    # stop/no_entry are mostly red, go is blue, warning is yellow.
    red_frac = extract_features(make_sign("stop", rng=3))[0]
    assert red_frac > 0.15
    blue_frac = extract_features(make_sign("go", rng=3))[1]
    assert blue_frac > 0.15
    yellow_frac = extract_features(make_sign("warning", rng=3))[2]
    # Yellow triangle occupies less area than a full circle (black
    # border + exclamation cut into it); 0.06 was measured for seed 3.
    assert yellow_frac > 0.05
    # speed_limit is mostly white inside its red ring.
    white_frac = extract_features(make_sign("speed_limit", rng=3))[3]
    assert white_frac > 0.10


def test_custom_size():
    img = make_sign("yield", rng=0, size=48)
    assert img.shape == (48, 48, 3)


def test_invalid_sign_type_raises():
    with pytest.raises(ValueError):
        make_sign("roundabout", rng=0)
    with pytest.raises(ValueError):
        make_sign("", rng=0)
    with pytest.raises(ValueError):
        make_sign(123, rng=0)


def test_invalid_size_raises():
    with pytest.raises(ValueError):
        make_sign("stop", rng=0, size=8)
    with pytest.raises(ValueError):
        make_sign("stop", rng=0, size="32")


def test_feature_vector_length_and_determinism():
    img = make_sign("no_entry", rng=5)
    feats = extract_features(img)
    assert len(feats) == 62
    assert np.all(np.isfinite(feats))
    assert np.array_equal(feats, extract_features(img))
    # Colour proportions (first 6) are fractions that sum to ~1.
    assert feats[:6].sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(feats[:6] >= 0) and np.all(feats[:6] <= 1)


def test_features_differ_between_classes():
    a = extract_features(make_sign("stop", rng=0))
    b = extract_features(make_sign("go", rng=0))
    assert np.linalg.norm(a - b) > 1.0


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        extract_features(np.zeros((32, 32)))  # not 3D
    with pytest.raises(ValueError):
        extract_features(np.zeros((32, 32, 4), dtype=np.uint8))  # not RGB
    with pytest.raises(ValueError):
        extract_features(np.zeros((4, 4, 3), dtype=np.uint8))  # too small
    with pytest.raises(ValueError):
        extract_features(np.full((32, 32, 3), 300))  # out of range


def test_make_dataset_counts_and_shuffling():
    images, labels = make_dataset(5, seed=11)
    assert len(images) == 30 and len(labels) == 30
    for sign_type in CLASSES:
        assert labels.count(sign_type) == 5
    # Same seed -> same dataset; different seed -> different order/content.
    images2, labels2 = make_dataset(5, seed=11)
    assert labels2 == labels
    assert all(np.array_equal(a, b) for a, b in zip(images, images2))
    _, labels3 = make_dataset(5, seed=12)
    assert labels3 != labels


def test_make_dataset_invalid_n_raises():
    with pytest.raises(ValueError):
        make_dataset(0)
    with pytest.raises(ValueError):
        make_dataset(-3)


def test_classifier_fits_and_predicts_training_signs():
    clf, images, labels = _trained()
    preds = clf.predict(images[:12])
    assert len(preds) == 12
    assert all(p in CLASSES for p in preds)
    assert clf.score(images, labels) >= 0.95


def test_classifier_accuracy_on_held_out_variants():
    clf, _, _ = _trained(n_per_class=30, seed=0)
    test_images, test_labels = make_dataset(20, seed=123)
    assert accuracy(test_images, test_labels, clf) >= 0.90
    assert clf.score(test_images, test_labels) >= 0.90


def test_classifier_per_class_accuracy():
    clf, _, _ = _trained(n_per_class=30, seed=0)
    test_images, test_labels = make_dataset(20, seed=123)
    preds = clf.predict(test_images)
    for sign_type in CLASSES:
        idx = [i for i, lab in enumerate(test_labels) if lab == sign_type]
        acc = np.mean([preds[i] == sign_type for i in idx])
        assert acc >= 0.80, f"{sign_type} accuracy {acc:.2f} too low"


def test_predict_one_matches_predict():
    clf, images, _ = _trained()
    assert clf.predict_one(images[0]) == clf.predict([images[0]])[0]


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        TrafficSignClassifier(k=0)
    with pytest.raises(ValueError):
        TrafficSignClassifier(k=-1)


def test_fit_empty_and_mismatched_raises():
    with pytest.raises(ValueError):
        TrafficSignClassifier().fit([], [])
    with pytest.raises(ValueError):
        TrafficSignClassifier().fit([make_sign("stop", rng=0)], ["stop", "go"])
    with pytest.raises(ValueError):
        TrafficSignClassifier(k=5).fit(
            [make_sign("stop", rng=0)], ["stop"]
        )  # k > n_train


def test_fit_invalid_label_raises():
    with pytest.raises(ValueError):
        TrafficSignClassifier().fit([make_sign("stop", rng=0)], ["roundabout"])


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        TrafficSignClassifier().predict([make_sign("stop", rng=0)])
    with pytest.raises(ValueError):
        TrafficSignClassifier().predict([])


def test_accuracy_label_lists_and_errors():
    assert accuracy(["stop", "go", "go"], ["stop", "go", "warning"]) == pytest.approx(2 / 3)
    assert accuracy(["stop"], ["stop"]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        accuracy(["stop"], ["stop", "go"])
