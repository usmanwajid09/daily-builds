import numpy as np
import pytest

from digit_classifier_hog import (
    HOGKNNClassifier,
    accuracy,
    hog_features,
    hog_vector_length,
    make_dataset,
    make_images,
    predict,
    template_match_accuracy,
    train,
)
from digit_recognizer.digits import template


def _train_test(cell_size=2, seed_train=1, seed_test=99):
    train_f, train_y = make_dataset(10, 0.05, seed_train, cell_size=cell_size)
    test_f, test_y = make_dataset(10, 0.05, seed_test, cell_size=cell_size)
    return train_f, train_y, test_f, test_y


def test_hog_vector_length_defaults():
    assert hog_vector_length(4, 9) == 36
    assert hog_vector_length(2, 9) == 324
    assert len(hog_features(template(0))) == 36
    assert len(hog_features(template(0), cell_size=2)) == 324


def test_hog_features_are_finite_and_normalised():
    for digit in range(10):
        vec = hog_features(template(digit), cell_size=2)
        assert np.isfinite(vec).all()
        assert vec.min() >= 0.0
        # 9 blocks, each L2-normalised to ~1 (a block with no edges
        # at all stays zero, so the total norm is at most 3.0).
        assert 2.5 <= np.linalg.norm(vec) <= 3.0


def test_hog_features_deterministic_and_digit_specific():
    a = hog_features(template(3), cell_size=2)
    b = hog_features(template(3), cell_size=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, hog_features(template(8), cell_size=2))
    # A blank image has no gradients, so its HOG vector is all zeros.
    assert np.all(hog_features(np.zeros((8, 8))) == 0.0)


def test_hog_invalid_image_shape_raises():
    with pytest.raises(ValueError):
        hog_features(np.zeros((7, 8)))
    with pytest.raises(ValueError):
        hog_features(np.zeros(64))
    with pytest.raises(ValueError):
        hog_features(np.zeros((8, 8, 1)))


def test_hog_invalid_values_raise():
    with pytest.raises(ValueError):
        hog_features(np.full((8, 8), np.nan))
    with pytest.raises(ValueError):
        hog_features(np.full((8, 8), 2.0))
    with pytest.raises(ValueError):
        hog_features(np.full((8, 8), -0.5))


def test_hog_invalid_settings_raise():
    with pytest.raises(ValueError):
        hog_features(template(0), cell_size=3)  # 3 does not divide 8
    with pytest.raises(ValueError):
        hog_features(template(0), cell_size=0)
    with pytest.raises(ValueError):
        hog_features(template(0), n_bins=0)
    with pytest.raises(ValueError):
        hog_vector_length(cell_size=5)


def test_make_dataset_shape_and_labels():
    features, labels = make_dataset(5, 0.05, seed=7, cell_size=2)
    assert features.shape == (50, 324)
    assert labels.shape == (50,)
    assert set(np.unique(labels)) == set(range(10))
    assert np.all(labels[:5] == 0) and np.all(labels[5:10] == 1)
    features4, _ = make_dataset(5, 0.05, seed=7, cell_size=4)
    assert features4.shape == (50, 36)


def test_make_dataset_deterministic_same_seed():
    f1, y1 = make_dataset(5, 0.05, seed=42, cell_size=2)
    f2, y2 = make_dataset(5, 0.05, seed=42, cell_size=2)
    assert np.array_equal(f1, f2) and np.array_equal(y1, y2)
    f3, _ = make_dataset(5, 0.05, seed=43, cell_size=2)
    assert not np.array_equal(f1, f3)


def test_make_dataset_append_pixels():
    features, _ = make_dataset(2, 0.05, seed=1, cell_size=2, append_pixels=True)
    assert features.shape == (20, 324 + 64)
    # Last 64 entries are the raw 0/1 pixels.
    assert set(np.unique(features[:, 324:])).issubset({0.0, 1.0})


def test_make_dataset_invalid_args_raise():
    with pytest.raises(ValueError):
        make_dataset(0, 0.05)
    with pytest.raises(ValueError):
        make_dataset(5, 1.5)
    with pytest.raises(ValueError):
        make_images(0, 0.05)
    with pytest.raises(ValueError):
        make_images(5, -0.1)


def test_knn_fits_and_predicts_clean_templates():
    train_f, train_y = make_dataset(10, 0.0, seed=1, cell_size=2)
    clf = train(train_f, train_y, k=3)
    for digit in range(10):
        assert clf.predict_one(hog_features(template(digit), cell_size=2)) == digit


def test_knn_noisy_accuracy_threshold():
    train_f, train_y, test_f, test_y = _train_test()
    clf = train(train_f, train_y, k=3)
    assert accuracy(test_f, test_y, clf) >= 0.85
    assert clf.score(test_f, test_y) >= 0.85


def test_knn_predict_single_vector_and_batch_agree():
    train_f, train_y, test_f, _ = _train_test()
    clf = train(train_f, train_y, k=3)
    batch = clf.predict(test_f)
    assert batch.shape == (100,)
    assert set(np.unique(batch)).issubset(set(range(10)))
    assert batch[0] == clf.predict_one(test_f[0])
    assert np.array_equal(predict(clf, test_f), batch)


def test_knn_tie_break_is_deterministic():
    # Two identical training vectors with different labels: the
    # smallest digit must win the tie every time.
    vec = hog_features(template(5), cell_size=2)
    clf = train(np.array([vec, vec]), np.array([5, 3]), k=2)
    assert clf.predict_one(vec) == 3
    assert clf.predict_one(vec) == 3


def test_knn_invalid_k_raises():
    with pytest.raises(ValueError):
        HOGKNNClassifier(k=0)
    with pytest.raises(ValueError):
        HOGKNNClassifier(k=-2)
    with pytest.raises(ValueError):
        HOGKNNClassifier(k=1.5)
    with pytest.raises(ValueError):
        train(np.zeros((2, 36)), np.array([0, 1]), k=3)  # k > n_train


def test_knn_fit_validation_raises():
    clf = HOGKNNClassifier(k=3)
    with pytest.raises(ValueError):
        clf.fit(np.zeros((0, 36)), np.array([], dtype=int))
    with pytest.raises(ValueError):
        clf.fit(np.zeros((2, 36)), np.array([0]))  # labels mismatch
    with pytest.raises(ValueError):
        clf.fit(np.zeros((2, 36)), np.array([0, 10]))  # label out of range
    with pytest.raises(ValueError):
        clf.fit(np.full((2, 36), np.inf), np.array([0, 1]))


def test_knn_predict_before_fit_and_wrong_length_raise():
    clf = HOGKNNClassifier(k=3)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((1, 36)))
    train_f, train_y = make_dataset(5, 0.05, seed=1, cell_size=2)
    clf.fit(train_f, train_y)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((1, 36)))  # wrong feature length
    with pytest.raises(ValueError):
        clf.predict(np.zeros((0, 324)))
    with pytest.raises(ValueError):
        predict("not a classifier", train_f)
    with pytest.raises(ValueError):
        accuracy(train_f, train_y, "not a classifier")


def test_template_match_accuracy_baseline():
    images, labels = make_images(20, 0.05, seed=99)
    acc = template_match_accuracy(images, labels)
    assert acc >= 0.95
    # Clean images are matched perfectly.
    clean = [template(d) for d in range(10)]
    assert template_match_accuracy(clean, list(range(10))) == 1.0


def test_template_match_accuracy_validation():
    with pytest.raises(ValueError):
        template_match_accuracy([template(0)], [0, 1])
    with pytest.raises(ValueError):
        template_match_accuracy([], [])


def test_hog_knn_beats_raw_chance_and_reports_honestly():
    # HOG + kNN must do far better than chance (0.10) on noisy digits;
    # the demo reports the honest gap to template matching separately.
    train_f, train_y, test_f, test_y = _train_test()
    clf = train(train_f, train_y, k=3)
    hog_acc = accuracy(test_f, test_y, clf)
    images, labels = make_images(10, 0.05, seed=99)
    tmpl_acc = template_match_accuracy(images, labels)
    assert hog_acc >= 0.85
    assert tmpl_acc >= hog_acc  # template matching wins on synthetic digits
