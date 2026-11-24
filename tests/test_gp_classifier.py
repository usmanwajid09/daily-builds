"""Tests for gp_classifier.GaussianProcessClassifier."""
import numpy as np
import pytest

from gp_classifier.model import GaussianProcessClassifier


def make_blobs(seed=0, n=200, sep=3.0, noise=1.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-sep, 0], noise, size=(n // 2, 2))
    x1 = rng.normal([sep, 0], noise, size=(n // 2, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    perm = rng.permutation(n)
    return x[perm], y[perm]


def test_accuracy_above_0_9_on_separable_blobs():
    x, y = make_blobs(seed=0)
    model = GaussianProcessClassifier().fit(x[:160], y[:160])
    assert model.score(x[160:], y[160:]) > 0.9


def test_predict_proba_in_range_and_rows_sum_to_one():
    x, y = make_blobs(seed=1)
    model = GaussianProcessClassifier().fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_proba_near_zero_and_one_far_inside_each_class():
    x, y = make_blobs(seed=2)
    model = GaussianProcessClassifier().fit(x, y)
    # Deep inside each class (at the blob centres, where data is dense).
    p0 = model.predict_proba(np.array([[-3.0, 0.0]]))[0, 1]
    p1 = model.predict_proba(np.array([[3.0, 0.0]]))[0, 1]
    assert p0 < 0.1
    assert p1 > 0.9


def test_predict_shape_and_dtype():
    x, y = make_blobs(seed=3)
    model = GaussianProcessClassifier().fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (len(y),)
    assert pred.dtype == np.int64 or pred.dtype == np.int32
    assert set(np.unique(pred)) <= {0, 1}


def test_score_in_zero_to_one():
    x, y = make_blobs(seed=4)
    model = GaussianProcessClassifier().fit(x[:160], y[:160])
    s = model.score(x[160:], y[160:])
    assert 0.0 <= s <= 1.0
    assert s > 0.9


def test_fit_raises_on_1d_x():
    x, y = make_blobs(seed=5)
    with pytest.raises(ValueError):
        GaussianProcessClassifier().fit(x[:, 0], y)


def test_fit_raises_on_mismatched_lengths():
    x, y = make_blobs(seed=6)
    with pytest.raises(ValueError):
        GaussianProcessClassifier().fit(x, y[:-1])


def test_fit_raises_on_empty_input():
    with pytest.raises(ValueError):
        GaussianProcessClassifier().fit(np.empty((0, 2)), np.empty((0,)))


def test_fit_raises_on_nan():
    x, y = make_blobs(seed=7)
    x = x.copy()
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        GaussianProcessClassifier().fit(x, y)


def test_fit_raises_on_non_binary_labels():
    x, y = make_blobs(seed=8)
    y = y.copy()
    y[0] = 2
    with pytest.raises(ValueError):
        GaussianProcessClassifier().fit(x, y)


def test_fit_raises_on_single_sample():
    with pytest.raises(ValueError):
        GaussianProcessClassifier().fit(np.array([[0.0, 0.0]]), np.array([1]))


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        GaussianProcessClassifier(length_scale=0.0)
    with pytest.raises(ValueError):
        GaussianProcessClassifier(length_scale=-1.0)
    with pytest.raises(ValueError):
        GaussianProcessClassifier(signal_variance=0.0)
    with pytest.raises(ValueError):
        GaussianProcessClassifier(signal_variance=-2.0)


def test_predict_before_fit_raises():
    x, _ = make_blobs(seed=9)
    model = GaussianProcessClassifier()
    with pytest.raises(ValueError):
        model.predict(x)
    with pytest.raises(ValueError):
        model.predict_proba(x)
    with pytest.raises(ValueError):
        model.score(x, np.zeros(len(x), dtype=int))


def test_determinism_same_fit_twice():
    x, y = make_blobs(seed=10)
    m1 = GaussianProcessClassifier().fit(x, y)
    m2 = GaussianProcessClassifier().fit(x, y)
    assert np.array_equal(m1.predict(x), m2.predict(x))
    assert np.allclose(m1.predict_proba(x), m2.predict_proba(x))


def test_predict_matches_proba_threshold():
    x, y = make_blobs(seed=11)
    model = GaussianProcessClassifier().fit(x, y)
    pred = model.predict(x)
    assert np.array_equal(pred, (model.predict_proba(x)[:, 1] >= 0.5).astype(int))
