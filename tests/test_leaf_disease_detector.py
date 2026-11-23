import numpy as np
import pytest

from leaf_disease_detector import (
    CLASSES,
    N_FEATURES,
    LeafDiseaseClassifier,
    accuracy,
    extract_features,
    make_dataset,
    make_leaf,
)


def _trained_clf(n_per_class=15, seed=0):
    images, labels = make_dataset(n_per_class, seed=seed)
    clf = LeafDiseaseClassifier(k=3)
    clf.fit(images, labels)
    return clf


def test_make_leaf_shape_dtype_and_value_range():
    leaf = make_leaf("healthy", rng=0)
    assert leaf.shape == (48, 48, 3)
    assert leaf.dtype == np.float64
    assert leaf.min() >= 0.0 and leaf.max() <= 1.0


def test_make_leaf_custom_size():
    leaf = make_leaf("spot_disease", rng=1, size=32)
    assert leaf.shape == (32, 32, 3)


def test_make_leaf_same_seed_reproducible():
    a = make_leaf("yellowing", rng=42)
    b = make_leaf("yellowing", rng=42)
    assert np.array_equal(a, b)


def test_make_leaf_invalid_label_raises():
    with pytest.raises(ValueError, match="label must be one of"):
        make_leaf("rust", rng=0)


def test_make_leaf_invalid_size_raises():
    with pytest.raises(ValueError, match="size must be an integer >= 16"):
        make_leaf("healthy", rng=0, size=8)


def test_classes_are_distinct_by_brown_and_yellow_fractions():
    rng = np.random.default_rng(3)
    healthy = extract_features(make_leaf("healthy", rng=rng))
    spotted = extract_features(make_leaf("spot_disease", rng=rng))
    yellow = extract_features(make_leaf("yellowing", rng=rng))
    # Feature indices: 9 = brown fraction, 10 = yellow fraction.
    assert spotted[9] > healthy[9]
    assert spotted[9] > yellow[9]
    assert yellow[10] > healthy[10]
    assert yellow[10] > spotted[10]
    assert healthy[12] > spotted[12]  # green fraction highest when healthy


def test_extract_features_length_and_determinism():
    leaf = make_leaf("spot_disease", rng=5)
    feats = extract_features(leaf)
    assert feats.shape == (N_FEATURES,)
    assert feats.shape == (13,)
    assert np.array_equal(feats, extract_features(leaf))
    assert np.isfinite(feats).all()


def test_extract_features_invalid_inputs_raise():
    with pytest.raises(ValueError, match="3D RGB"):
        extract_features(np.zeros((48, 48)))  # grayscale, no channel axis
    with pytest.raises(ValueError, match="3D RGB"):
        extract_features(np.zeros((48, 48, 4)))  # RGBA, not RGB
    with pytest.raises(ValueError, match="lie in \\[0, 1\\]"):
        extract_features(np.full((48, 48, 3), 1.5))
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros((48, 48, 3))
        bad[0, 0, 0] = np.nan
        extract_features(bad)


def test_spot_count_feature_positive_for_spotted_leaf():
    rng = np.random.default_rng(11)
    spotted = extract_features(make_leaf("spot_disease", rng=rng))
    healthy = extract_features(make_leaf("healthy", rng=rng))
    # Feature index 11 = brown-blob count.
    assert spotted[11] >= 3
    assert healthy[11] == 0


def test_classifier_k_validation():
    with pytest.raises(ValueError, match="k must be an integer >= 1"):
        LeafDiseaseClassifier(k=0)


def test_classifier_fit_empty_training_set_raises():
    clf = LeafDiseaseClassifier()
    with pytest.raises(ValueError, match="at least one training image"):
        clf.fit([], [])


def test_classifier_fit_mismatched_lengths_raises():
    clf = LeafDiseaseClassifier()
    with pytest.raises(ValueError, match="images but"):
        clf.fit([make_leaf("healthy", rng=0)], ["healthy", "yellowing"])


def test_classifier_fit_invalid_label_raises():
    clf = LeafDiseaseClassifier()
    with pytest.raises(ValueError, match="label must be one of"):
        clf.fit([make_leaf("healthy", rng=0)], ["rust"])


def test_classifier_predict_before_fit_raises():
    clf = LeafDiseaseClassifier()
    with pytest.raises(ValueError, match="not been fitted"):
        clf.predict(make_leaf("healthy", rng=0))


def test_classifier_predicts_each_class():
    clf = _trained_clf()
    rng = np.random.default_rng(123)
    for label in CLASSES:
        assert clf.predict(make_leaf(label, rng=rng)) == label


def test_classifier_predict_many_length_and_labels():
    clf = _trained_clf()
    images, labels = make_dataset(3, seed=77)
    preds = clf.predict_many(images)
    assert len(preds) == len(images)
    assert set(preds) <= set(CLASSES)


def test_make_dataset_counts_and_order():
    images, labels = make_dataset(4, seed=2)
    assert len(images) == 12 and len(labels) == 12
    assert labels == ["healthy"] * 4 + ["spot_disease"] * 4 + ["yellowing"] * 4
    assert all(img.shape == (48, 48, 3) for img in images)


def test_make_dataset_invalid_n_raises():
    with pytest.raises(ValueError, match="n_per_class must be an integer >= 1"):
        make_dataset(0)


def test_accuracy_above_threshold_on_held_out():
    clf = _trained_clf(n_per_class=30, seed=7)
    test_images, test_labels = make_dataset(15, seed=99)
    acc = accuracy(test_images, test_labels, clf)
    assert acc >= 0.90


def test_accuracy_empty_and_mismatched_raise():
    clf = _trained_clf(n_per_class=5, seed=0)
    with pytest.raises(ValueError, match="at least one image"):
        accuracy([], [], clf)
    with pytest.raises(ValueError, match="images but"):
        accuracy([make_leaf("healthy", rng=0)], ["healthy", "yellowing"], clf)


def test_accuracy_perfect_on_training_subset():
    clf = _trained_clf(n_per_class=10, seed=0)
    images, labels = make_dataset(5, seed=0)
    # Training images (same seed) are in the training set, so kNN
    # with k=3 still votes correctly on these familiar examples.
    assert accuracy(images, labels, clf) >= 0.90
