import numpy as np
import pytest

from elastic_net_classifier.model import ElasticNetClassifier


def _blobs(seed=0, n=50):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[0.0, 0.0], scale=0.8, size=(n, 2))
    x1 = rng.normal(loc=[4.0, 3.0], scale=0.8, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_binary_accuracy_high():
    x, y = _blobs()
    model = ElasticNetClassifier(alpha=0.1, l1_ratio=0.5).fit(x, y)
    assert model.score(x, y) > 0.95


def test_predicts_nearby_class():
    x, y = _blobs()
    model = ElasticNetClassifier(alpha=0.1, l1_ratio=0.5).fit(x, y)
    assert model.predict(np.array([[0.1, 0.1]]))[0] == 0
    assert model.predict(np.array([[4.1, 3.1]]))[0] == 1


def test_multiclass():
    rng = np.random.default_rng(2)
    xs, ys = [], []
    for i, c in enumerate([[0, 0], [6, 0], [3, 6]]):
        xs.append(rng.normal(loc=c, scale=0.8, size=(30, 2)))
        ys.append(np.full(30, i))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    model = ElasticNetClassifier(alpha=0.1, l1_ratio=0.5).fit(x, y)
    assert model.coef_.shape == (3, 2)
    assert model.score(x, y) > 0.95


def test_proba_rows_sum_to_one():
    x, y = _blobs()
    model = ElasticNetClassifier(alpha=0.1, l1_ratio=0.5).fit(x, y)
    probs = model.predict_proba(x[:10])
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_pure_l1_zeroes_noise_features():
    rng = np.random.default_rng(3)
    n = 100
    x0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    x1 = rng.normal(loc=[3.0, 2.5], scale=1.0, size=(n, 2))
    noise = rng.normal(size=(2 * n, 4))
    x = np.vstack([np.hstack([x0, noise[:n]]), np.hstack([x1, noise[n:]])])
    y = np.array([0] * n + [1] * n)
    model = ElasticNetClassifier(alpha=0.1, l1_ratio=1.0).fit(x, y)
    # Noise features (columns 2-5) should be zeroed for at least one class.
    n_noise_zero = int(np.sum(model.coef_[:, 2:] == 0.0))
    assert n_noise_zero >= 4
    assert model.score(x, y) > 0.9


def test_pure_l2_keeps_all_nonzero():
    x, y = _blobs()
    model = ElasticNetClassifier(alpha=0.1, l1_ratio=0.0).fit(x, y)
    assert np.all(model.coef_ != 0.0)


def test_alpha_zero_is_plain_logistic():
    x, y = _blobs()
    model = ElasticNetClassifier(alpha=0.0, l1_ratio=0.5).fit(x, y)
    assert model.score(x, y) > 0.95


def test_fit_returns_self():
    x, y = _blobs()
    model = ElasticNetClassifier()
    assert model.fit(x, y) is model


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ElasticNetClassifier().predict(np.array([[0.0, 0.0]]))


def test_bad_params_raise():
    with pytest.raises(ValueError, match="alpha"):
        ElasticNetClassifier(alpha=-0.5)
    with pytest.raises(ValueError, match="l1_ratio"):
        ElasticNetClassifier(l1_ratio=1.5)
    with pytest.raises(ValueError, match="l1_ratio"):
        ElasticNetClassifier(l1_ratio=-0.1)
    with pytest.raises(ValueError, match="max_iter"):
        ElasticNetClassifier(max_iter=0)
    with pytest.raises(ValueError, match="learning_rate"):
        ElasticNetClassifier(learning_rate=0.0)


def test_mismatched_xy_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="samples"):
        ElasticNetClassifier().fit(x, y[:-1])


def test_single_class_raises():
    with pytest.raises(ValueError, match="at least 2 classes"):
        ElasticNetClassifier().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1, 1]))


def test_wrong_feature_count_raises():
    x, y = _blobs()
    model = ElasticNetClassifier().fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.array([[0.0, 0.0, 0.0]]))


def test_nonfinite_X_raises():
    x = np.array([[0.0, np.nan], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        ElasticNetClassifier().fit(x, y)


def test_string_labels():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array(["a", "a", "a", "b", "b", "b"])
    model = ElasticNetClassifier(alpha=0.1).fit(x, y)
    assert set(model.predict(x)) == {"a", "b"}


def test_n_iter_recorded():
    x, y = _blobs()
    model = ElasticNetClassifier(alpha=0.1).fit(x, y)
    assert model.n_iter_.shape == (2,)
    assert np.all(model.n_iter_ >= 1)
