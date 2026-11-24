import numpy as np
import pytest

from document_classifier import (
    CLASSES,
    FEATURE_NAMES,
    DocumentClassifier,
    accuracy,
    confusion_matrix,
    extract_features,
    make_document,
    train_classifier,
)


def _dataset(n_train=10, n_test=5, train_start=0, test_start=1000):
    train_X, train_y, test_X, test_y = [], [], [], []
    for cls in CLASSES:
        for seed in range(train_start, train_start + n_train):
            train_X.append(make_document(cls, seed))
            train_y.append(cls)
        for seed in range(test_start, test_start + n_test):
            test_X.append(make_document(cls, seed))
            test_y.append(cls)
    return train_X, train_y, test_X, test_y


def _fitted(n_train=10):
    train_X, train_y, _, _ = _dataset(n_train=n_train)
    return train_classifier(train_X, train_y)


# --- generation ------------------------------------------------------------

def test_make_document_shape_dtype_and_values():
    img = make_document("invoice", seed=0)
    assert img.shape == (150, 200)
    assert img.dtype == np.uint8
    assert set(np.unique(img)).issubset({0, 255})
    assert 0 < img.mean() < 255  # has both ink and paper


def test_make_document_deterministic_same_seed():
    a = make_document("letter", seed=7)
    b = make_document("letter", seed=7)
    assert np.array_equal(a, b)


def test_make_document_differs_across_seeds_and_types():
    assert not np.array_equal(make_document("form", seed=1),
                              make_document("form", seed=2))
    assert not np.array_equal(make_document("invoice", seed=1),
                              make_document("receipt", seed=1))


def test_make_document_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown document type"):
        make_document("passport", seed=0)


def test_make_document_non_integer_seed_raises():
    with pytest.raises(ValueError, match="seed must be an integer"):
        make_document("letter", seed="abc")


def test_each_class_has_distinct_layout():
    # Receipt is narrow (wide side margins); letter is dense with ink.
    receipt = extract_features(make_document("receipt", seed=0))
    letter = extract_features(make_document("letter", seed=0))
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert receipt[idx["left_margin"]] > letter[idx["left_margin"]]
    assert receipt[idx["right_margin"]] > letter[idx["right_margin"]]
    assert letter[idx["ink_density"]] > receipt[idx["ink_density"]]
    # Invoice has the table grid: many long horizontal/vertical lines.
    invoice = extract_features(make_document("invoice", seed=0))
    assert invoice[idx["h_line_count"]] > letter[idx["h_line_count"]]
    assert invoice[idx["v_line_count"]] > letter[idx["v_line_count"]]
    # Form has enclosed boxes; letter has (almost) none.
    form = extract_features(make_document("form", seed=0))
    assert form[idx["box_count"]] >= 1
    assert letter[idx["box_count"]] == 0


# --- features --------------------------------------------------------------

def test_extract_features_length_and_finiteness():
    feats = extract_features(make_document("form", seed=3))
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(feats).all()
    assert (feats >= 0).all()


def test_extract_features_blank_page_is_all_zeros():
    blank = np.full((150, 200), 255, dtype=np.uint8)
    assert np.all(extract_features(blank) == 0)


def test_extract_features_rejects_empty_and_non_2d():
    with pytest.raises(ValueError, match="empty"):
        extract_features(np.zeros((0, 0), dtype=np.uint8))
    with pytest.raises(ValueError, match="2-D"):
        extract_features(np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="numpy array"):
        extract_features([[0, 1], [1, 0]])


# --- classifier ------------------------------------------------------------

def test_classifier_accuracy_on_held_out_docs():
    train_X, train_y, test_X, test_y = _dataset(n_train=20, n_test=10)
    clf = train_classifier(train_X, train_y)
    assert accuracy(clf, test_X, test_y) >= 0.85


def test_predict_returns_a_known_class_and_proba_sums_to_one():
    clf = _fitted()
    img = make_document("receipt", seed=999)
    pred = clf.predict(img)
    assert pred in CLASSES
    proba = clf.predict_proba(img)
    assert set(proba) == set(CLASSES)
    assert sum(proba.values()) == pytest.approx(1.0)
    assert proba[pred] == max(proba.values())


def test_predict_before_fit_raises():
    clf = DocumentClassifier()
    with pytest.raises(ValueError, match="not fitted"):
        clf.predict(make_document("letter", seed=0))
    with pytest.raises(ValueError, match="not fitted"):
        clf.predict_proba(make_document("letter", seed=0))


def test_fit_rejects_empty_data_mismatched_labels_and_unknown_label():
    with pytest.raises(ValueError, match="empty training set"):
        train_classifier([], [])
    with pytest.raises(ValueError, match="counts must match"):
        train_classifier([make_document("letter", 0)], ["letter", "form"])
    with pytest.raises(ValueError, match="unknown label"):
        train_classifier([make_document("letter", 0)], ["passport"])


def test_fit_predict_roundtrip_on_training_data():
    train_X, train_y, _, _ = _dataset(n_train=5)
    clf = train_classifier(train_X, train_y)
    assert clf.score(train_X, train_y) == pytest.approx(1.0)


def test_score_rejects_empty_and_mismatched():
    clf = _fitted()
    with pytest.raises(ValueError, match="empty"):
        clf.score([], [])
    with pytest.raises(ValueError, match="counts must match"):
        clf.score([make_document("letter", 0)], [])


def test_confusion_matrix_counts_add_up():
    train_X, train_y, test_X, test_y = _dataset(n_train=10, n_test=5)
    clf = train_classifier(train_X, train_y)
    cm = confusion_matrix(clf, test_X, test_y)
    assert sum(sum(row.values()) for row in cm.values()) == len(test_y)
    for true in CLASSES:
        assert sum(cm[true].values()) == 5
    # Diagonal dominates: most docs classified correctly.
    assert sum(cm[c][c] for c in CLASSES) >= 0.85 * len(test_y)


def test_confusion_matrix_rejects_unknown_label():
    clf = _fitted()
    with pytest.raises(ValueError, match="unknown label"):
        confusion_matrix(clf, [make_document("letter", 0)], ["passport"])


def test_classifier_is_deterministic():
    train_X, train_y, test_X, _ = _dataset(n_train=10, n_test=5)
    clf1 = train_classifier(train_X, train_y)
    clf2 = train_classifier(train_X, train_y)
    preds1 = [clf1.predict(img) for img in test_X]
    preds2 = [clf2.predict(img) for img in test_X]
    assert preds1 == preds2
