import numpy as np
import pytest

from label_propagation.model import LabelPropagation


def make_blobs(seed=0, n_per_class=100, spread=0.8):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 5.0]])
    X = np.vstack([rng.normal(c, spread, size=(n_per_class, 2)) for c in centres])
    y_true = np.repeat([0, 1], n_per_class)
    return X, y_true


def sparsely_label(y_true, per_class=10, seed=1):
    rng = np.random.default_rng(seed)
    y = np.full(len(y_true), -1)
    for cls in np.unique(y_true):
        idx = np.where(y_true == cls)[0]
        y[rng.choice(idx, size=per_class, replace=False)] = cls
    return y


def test_happy_path_blobs_accuracy_on_unlabeled():
    X, y_true = make_blobs()
    y = sparsely_label(y_true)
    model = LabelPropagation(n_neighbors=7, alpha=0.8).fit(X, y)
    pred = model.predict(X)
    acc = np.mean(pred[y == -1] == y_true[y == -1])
    assert acc > 0.9


def test_label_distributions_rows_sum_to_one():
    X, y_true = make_blobs()
    model = LabelPropagation().fit(X, sparsely_label(y_true))
    assert model.label_distributions_.shape == (len(X), 2)
    np.testing.assert_allclose(
        model.label_distributions_.sum(axis=1), np.ones(len(X)), atol=1e-8
    )
    assert (model.label_distributions_ >= 0).all()


def test_labelled_points_keep_their_labels():
    X, y_true = make_blobs()
    y = sparsely_label(y_true)
    model = LabelPropagation().fit(X, y)
    np.testing.assert_array_equal(model.labels_[y != -1], y[y != -1])


def test_fully_labelled_data_near_perfect():
    X, y_true = make_blobs()
    model = LabelPropagation().fit(X, y_true)
    assert np.mean(model.predict(X) == y_true) > 0.99


def test_all_unlabeled_raises():
    X, _ = make_blobs()
    with pytest.raises(ValueError, match="no labelled"):
        LabelPropagation().fit(X, np.full(len(X), -1))


def test_single_class_raises():
    X, y_true = make_blobs()
    y = np.full(len(X), -1)
    y[:10] = 0  # only class 0 labelled
    with pytest.raises(ValueError, match="at least 2 distinct"):
        LabelPropagation().fit(X, y)


def test_length_mismatch_raises():
    X, y_true = make_blobs()
    with pytest.raises(ValueError, match="inconsistent lengths"):
        LabelPropagation().fit(X, y_true[:-5])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        LabelPropagation().fit(np.empty((0, 2)), np.empty(0, dtype=int))


def test_1d_X_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="2D"):
        LabelPropagation().fit(rng.normal(size=20), np.array([0, 1] * 10))


def test_nan_raises():
    X, y_true = make_blobs()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        LabelPropagation().fit(X, sparsely_label(y_true))


def test_predict_before_fit_raises():
    X, _ = make_blobs()
    with pytest.raises(RuntimeError, match="not fitted"):
        LabelPropagation().predict(X)


def test_n_neighbors_too_large_raises():
    X, y_true = make_blobs(n_per_class=5)
    with pytest.raises(ValueError, match="n_neighbors"):
        LabelPropagation(n_neighbors=10).fit(X, sparsely_label(y_true, per_class=2))


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError, match="alpha"):
        LabelPropagation(alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        LabelPropagation(alpha=0.0)


def test_predict_new_points_near_centres():
    X, y_true = make_blobs()
    model = LabelPropagation().fit(X, sparsely_label(y_true))
    new_points = np.array([[0.0, 0.0], [5.0, 5.0], [0.2, -0.1], [4.9, 5.1]])
    np.testing.assert_array_equal(model.predict(new_points), [0, 1, 0, 1])


def test_deterministic():
    X, y_true = make_blobs()
    y = sparsely_label(y_true)
    a = LabelPropagation().fit(X, y).labels_
    b = LabelPropagation().fit(X, y).labels_
    np.testing.assert_array_equal(a, b)
