import numpy as np
import pytest

from pls_classifier.model import PLSClassifier


def _collinear(seed=0, n=40):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i, c in enumerate([[-3.0, -3.0], [3.0, -3.0], [0.0, 3.0]]):
        latent = rng.normal(loc=c, scale=1.0, size=(n, 2))
        mixing = rng.normal(size=(2, 8))
        xs.append(latent @ mixing + rng.normal(scale=0.3, size=(n, 8)))
        ys.append(np.full(n, i))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def _blobs(seed=0, n=40):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[0.0, 0.0], scale=0.8, size=(n, 2))
    x1 = rng.normal(loc=[5.0, 4.0], scale=0.8, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_collinear_accuracy_high():
    x, y = _collinear()
    model = PLSClassifier(n_components=2).fit(x, y)
    assert model.score(x, y) > 0.9


def test_binary_accuracy_high():
    x, y = _blobs()
    model = PLSClassifier(n_components=1).fit(x, y)
    assert model.score(x, y) > 0.95


def test_more_components_help_collinear():
    x, y = _collinear()
    acc1 = PLSClassifier(n_components=1).fit(x, y).score(x, y)
    acc2 = PLSClassifier(n_components=2).fit(x, y).score(x, y)
    assert acc2 >= acc1


def test_predicts_nearby_class():
    x, y = _blobs()
    model = PLSClassifier(n_components=1).fit(x, y)
    assert model.predict(np.array([[0.1, 0.1]]))[0] == 0
    assert model.predict(np.array([[5.1, 4.1]]))[0] == 1


def test_decision_function_shape():
    x, y = _collinear()
    model = PLSClassifier(n_components=2).fit(x, y)
    assert model.decision_function(x[:5]).shape == (5, 3)


def test_explained_variance_sums_below_one():
    x, y = _collinear()
    model = PLSClassifier(n_components=2).fit(x, y)
    assert model.explained_variance_ratio_.shape == (2,)
    assert 0 < model.explained_variance_ratio_.sum() <= 1.0


def test_weights_shape():
    x, y = _collinear()
    model = PLSClassifier(n_components=2).fit(x, y)
    assert model.x_weights_.shape == (8, 2)
    assert model.x_loadings_.shape == (8, 2)


def test_no_scale_option():
    x, y = _blobs()
    model = PLSClassifier(n_components=1, scale=False).fit(x, y)
    assert model.score(x, y) > 0.95


def test_fit_returns_self():
    x, y = _blobs()
    model = PLSClassifier()
    assert model.fit(x, y) is model


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        PLSClassifier().predict(np.array([[0.0, 0.0]]))


def test_bad_n_components_raises():
    with pytest.raises(ValueError, match="n_components"):
        PLSClassifier(n_components=0)


def test_too_many_components_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="exceeds the maximum"):
        PLSClassifier(n_components=5).fit(x, y)


def test_mismatched_xy_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="samples"):
        PLSClassifier().fit(x, y[:-1])


def test_single_class_raises():
    with pytest.raises(ValueError, match="at least 2 classes"):
        PLSClassifier().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1, 1]))


def test_wrong_feature_count_raises():
    x, y = _blobs()
    model = PLSClassifier(n_components=1).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.array([[0.0, 0.0, 0.0]]))


def test_nonfinite_X_raises():
    x = np.array([[0.0, np.nan], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        PLSClassifier().fit(x, y)


def test_string_labels():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array(["a", "a", "a", "b", "b", "b"])
    model = PLSClassifier(n_components=1).fit(x, y)
    assert set(model.predict(x)) == {"a", "b"}
