import numpy as np
import pytest

from ridge_classifier.model import RidgeClassifier


def _blobs(seed=0, n=40):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[0.0, 0.0], scale=0.8, size=(n, 2))
    x1 = rng.normal(loc=[5.0, 4.0], scale=0.8, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_binary_accuracy_high():
    x, y = _blobs()
    model = RidgeClassifier(alpha=1.0).fit(x, y)
    assert model.score(x, y) > 0.95


def test_predicts_nearby_class():
    x, y = _blobs()
    model = RidgeClassifier(alpha=1.0).fit(x, y)
    assert model.predict(np.array([[0.1, 0.1]]))[0] == 0
    assert model.predict(np.array([[5.1, 4.1]]))[0] == 1


def test_multiclass():
    rng = np.random.default_rng(1)
    xs, ys = [], []
    for i, c in enumerate([[0, 0], [6, 0], [3, 6]]):
        xs.append(rng.normal(loc=c, scale=0.8, size=(30, 2)))
        ys.append(np.full(30, i))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    model = RidgeClassifier(alpha=1.0).fit(x, y)
    assert model.coef_.shape == (2, 3)
    assert model.score(x, y) > 0.95


def test_string_labels():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array(["cat", "cat", "cat", "dog", "dog", "dog"])
    model = RidgeClassifier(alpha=1.0).fit(x, y)
    assert set(model.predict(x)) == {"cat", "dog"}
    assert list(model.classes_) == ["cat", "dog"]


def test_decision_function_shape():
    x, y = _blobs()
    model = RidgeClassifier(alpha=1.0).fit(x, y)
    scores = model.decision_function(x[:5])
    assert scores.shape == (5, 2)


def test_larger_alpha_shrinks_coef():
    x, y = _blobs()
    small = RidgeClassifier(alpha=0.01).fit(x, y)
    big = RidgeClassifier(alpha=100.0).fit(x, y)
    assert np.linalg.norm(big.coef_) < np.linalg.norm(small.coef_)


def test_no_intercept():
    x, y = _blobs()
    model = RidgeClassifier(alpha=1.0, fit_intercept=False).fit(x, y)
    assert np.all(model.intercept_ == 0)
    # Without an intercept the boundary must pass through the origin,
    # so accuracy on uncentred blobs is lower but still well above chance.
    assert model.score(x, y) > 0.6


def test_fit_returns_self():
    x, y = _blobs()
    model = RidgeClassifier()
    assert model.fit(x, y) is model


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        RidgeClassifier().predict(np.array([[0.0, 0.0]]))


def test_decision_function_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        RidgeClassifier().decision_function(np.array([[0.0, 0.0]]))


def test_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        RidgeClassifier(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        RidgeClassifier(alpha=-1.0)


def test_mismatched_xy_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="samples"):
        RidgeClassifier().fit(x, y[:-1])


def test_single_class_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="at least 2 classes"):
        RidgeClassifier().fit(x, np.array([1, 1]))


def test_wrong_feature_count_raises():
    x, y = _blobs()
    model = RidgeClassifier().fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.array([[0.0, 0.0, 0.0]]))


def test_nonfinite_X_raises():
    x = np.array([[0.0, np.nan], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        RidgeClassifier().fit(x, y)


def test_1d_X_raises():
    with pytest.raises(ValueError, match="2D"):
        RidgeClassifier().fit(np.array([0.0, 1.0, 2.0]), np.array([0, 1, 1]))
