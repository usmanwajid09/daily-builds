import numpy as np
import pytest

from mlp_classifier.model import MLPClassifier


def _moons(seed=42, n=100):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, np.pi, size=n)
    outer = np.c_[np.cos(t), np.sin(t)]
    inner = np.c_[1.0 - np.cos(t), 0.5 - np.sin(t)]
    x = np.vstack([outer, inner]) + rng.normal(scale=0.1, size=(2 * n, 2))
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def _blobs(seed=42, n=60):
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(loc=[0, 0], scale=1, size=(n, 2)),
                   rng.normal(loc=[4, 3], scale=1, size=(n, 2))])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_learns_moons():
    x, y = _moons()
    model = MLPClassifier(hidden_size=16, epochs=300, learning_rate=0.1,
                          random_state=42).fit(x, y)
    assert model.score(x, y) > 0.9


def test_learns_blobs():
    x, y = _blobs()
    model = MLPClassifier(hidden_size=8, epochs=200, learning_rate=0.1,
                          random_state=0).fit(x, y)
    assert model.score(x, y) > 0.95


def test_loss_decreases():
    x, y = _moons()
    model = MLPClassifier(hidden_size=16, epochs=100, learning_rate=0.1,
                          random_state=42).fit(x, y)
    assert len(model.loss_history_) == 100
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_predict_proba_sums_to_one():
    x, y = _blobs()
    model = MLPClassifier(hidden_size=8, epochs=50, random_state=0).fit(x, y)
    probs = model.predict_proba(x)
    assert probs.shape == (len(y), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_reproducible_with_same_seed():
    x, y = _blobs()
    m1 = MLPClassifier(hidden_size=8, epochs=50, random_state=7).fit(x, y)
    m2 = MLPClassifier(hidden_size=8, epochs=50, random_state=7).fit(x, y)
    assert np.array_equal(m1.predict(x), m2.predict(x))


def test_three_classes():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(loc=c, scale=1, size=(40, 2))
                   for c in [[0, 0], [4, 3], [0, 5]]])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    model = MLPClassifier(hidden_size=16, epochs=200, learning_rate=0.1,
                          random_state=0).fit(x, y)
    assert model.score(x, y) > 0.9
    assert model.predict_proba(x).shape == (120, 3)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        MLPClassifier().predict(np.zeros((2, 2)))


def test_bad_hyperparams_raise():
    with pytest.raises(ValueError, match="hidden_size"):
        MLPClassifier(hidden_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        MLPClassifier(learning_rate=0.0)
    with pytest.raises(ValueError, match="epochs"):
        MLPClassifier(epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        MLPClassifier(batch_size=0)
    with pytest.raises(ValueError, match="l2"):
        MLPClassifier(l2=-0.1)


def test_one_class_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="at least 2 classes"):
        MLPClassifier().fit(x, np.zeros_like(y))


def test_mismatched_shapes_raise():
    x, y = _blobs()
    with pytest.raises(ValueError, match="same number of samples"):
        MLPClassifier().fit(x, y[:10])
    with pytest.raises(ValueError, match="2D"):
        MLPClassifier().fit(x.ravel(), y)


def test_nan_input_raises():
    x, y = _blobs()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        MLPClassifier().fit(x, y)


def test_fit_returns_self():
    x, y = _blobs()
    model = MLPClassifier(epochs=5, random_state=0)
    assert model.fit(x, y) is model
