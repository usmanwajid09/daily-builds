import numpy as np
import pytest

from kernel_logistic_regression.model import KernelLogisticRegression


def _moons(seed=0, n=60):
    rng = np.random.default_rng(seed)
    t0 = rng.uniform(0, np.pi, n)
    t1 = rng.uniform(0, np.pi, n)
    x0 = np.column_stack([np.cos(t0), np.sin(t0)])
    x1 = np.column_stack([1.0 - np.cos(t1), 1.0 - np.sin(t1) - 0.5])
    x = np.vstack([x0, x1]) + rng.normal(scale=0.1, size=(2 * n, 2))
    y = np.array([0] * n + [1] * n)
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


def test_moons_accuracy_high():
    x, y = _moons()
    model = KernelLogisticRegression(gamma=2.0, C=1.0).fit(x, y)
    assert model.score(x, y) > 0.9


def test_blobs_accuracy_high():
    x, y = _blobs()
    model = KernelLogisticRegression(gamma=0.5, C=1.0).fit(x, y)
    assert model.score(x, y) > 0.95


def test_predicts_nearby_class():
    x, y = _blobs()
    model = KernelLogisticRegression(gamma=0.5, C=1.0).fit(x, y)
    assert model.predict(np.array([[0.1, 0.1]]))[0] == 0
    assert model.predict(np.array([[5.1, 4.1]]))[0] == 1


def test_proba_rows_sum_to_one():
    x, y = _moons()
    model = KernelLogisticRegression(gamma=2.0, C=1.0).fit(x, y)
    probs = model.predict_proba(x[:10])
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_proba_agrees_with_predict():
    x, y = _moons()
    model = KernelLogisticRegression(gamma=2.0, C=1.0).fit(x, y)
    probs = model.predict_proba(x)
    from_predict = model.predict(x)
    from_proba = model.classes_[np.argmax(probs, axis=1)]
    assert np.array_equal(from_predict, from_proba)


def test_decision_function_sign_matches_predict():
    x, y = _blobs()
    model = KernelLogisticRegression(gamma=0.5, C=1.0).fit(x, y)
    scores = model.decision_function(x)
    assert scores.shape == (len(y),)
    assert np.array_equal(
        model.classes_[(scores >= 0).astype(int)], model.predict(x)
    )


def test_larger_C_fits_training_harder():
    x, y = _moons()
    loose = KernelLogisticRegression(gamma=2.0, C=0.01).fit(x, y)
    tight = KernelLogisticRegression(gamma=2.0, C=100.0).fit(x, y)
    assert tight.score(x, y) >= loose.score(x, y)


def test_fit_returns_self():
    x, y = _blobs()
    model = KernelLogisticRegression()
    assert model.fit(x, y) is model


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        KernelLogisticRegression().predict(np.array([[0.0, 0.0]]))


def test_bad_params_raise():
    with pytest.raises(ValueError, match="gamma"):
        KernelLogisticRegression(gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        KernelLogisticRegression(gamma=-1.0)
    with pytest.raises(ValueError, match="C"):
        KernelLogisticRegression(C=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        KernelLogisticRegression(max_iter=0)
    with pytest.raises(ValueError, match="learning_rate"):
        KernelLogisticRegression(learning_rate=-0.5)


def test_multiclass_raises():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2))
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    with pytest.raises(ValueError, match="binary only"):
        KernelLogisticRegression().fit(x, y)


def test_single_class_raises():
    with pytest.raises(ValueError, match="binary only"):
        KernelLogisticRegression().fit(
            np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1, 1])
        )


def test_mismatched_xy_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="samples"):
        KernelLogisticRegression().fit(x, y[:-1])


def test_wrong_feature_count_raises():
    x, y = _blobs()
    model = KernelLogisticRegression(gamma=0.5).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.array([[0.0, 0.0, 0.0]]))


def test_nonfinite_X_raises():
    x = np.array([[0.0, np.nan], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        KernelLogisticRegression().fit(x, y)


def test_string_labels():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array(["a", "a", "a", "b", "b", "b"])
    model = KernelLogisticRegression(gamma=0.5).fit(x, y)
    assert set(model.predict(x)) == {"a", "b"}


def test_n_iter_recorded():
    x, y = _blobs()
    model = KernelLogisticRegression(gamma=0.5).fit(x, y)
    assert 1 <= model.n_iter_ <= model.max_iter
