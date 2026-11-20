"""Tests for the RBF-kernel SVM (svm_rbf)."""
import numpy as np
import pytest

from svm_rbf import RBFKernelSVM
from svm_rbf.model import RBFKernelSVM as RBFKernelSVMFromModel


def make_circles(seed=42, n=100, noise=0.1):
    rng = np.random.default_rng(seed)
    a1 = rng.uniform(0, 2 * np.pi, n)
    a2 = rng.uniform(0, 2 * np.pi, n)
    inner = np.column_stack([np.cos(a1), np.sin(a1)]) + rng.normal(0, noise, (n, 2))
    outer = 3.0 * np.column_stack([np.cos(a2), np.sin(a2)]) + rng.normal(0, noise, (n, 2))
    X = np.vstack([inner, outer])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_moons(seed=7, n=100, noise=0.1):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n)
    upper = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, noise, (n, 2))
    t2 = rng.uniform(0, np.pi, n)
    lower = np.column_stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5]) + rng.normal(0, noise, (n, 2))
    X = np.vstack([upper, lower])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_blobs(seed=42, n=60):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    c1 = rng.normal(loc=[4.0, 3.0], scale=1.0, size=(n, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def split(X, y, frac=0.8):
    cut = int(frac * len(y))
    return X[:cut], y[:cut], X[cut:], y[cut:]


# -- accuracy -----------------------------------------------------------

def test_circles_accuracy():
    X, y = make_circles()
    Xtr, ytr, Xte, yte = split(X, y)
    model = RBFKernelSVM(C=1.0, random_state=42).fit(Xtr, ytr)
    assert model.score(Xte, yte) >= 0.9


def test_moons_accuracy():
    X, y = make_moons()
    Xtr, ytr, Xte, yte = split(X, y)
    model = RBFKernelSVM(C=1.0, random_state=7).fit(Xtr, ytr)
    assert model.score(Xte, yte) >= 0.9


def test_linearly_separable_blobs_accuracy():
    X, y = make_blobs()
    Xtr, ytr, Xte, yte = split(X, y)
    model = RBFKernelSVM(C=1.0, random_state=42).fit(Xtr, ytr)
    assert model.score(Xte, yte) >= 0.9


def test_fit_returns_self_and_train_accuracy_high():
    X, y = make_blobs()
    model = RBFKernelSVM(random_state=42)
    assert model.fit(X, y) is model
    assert model.score(X, y) >= 0.95


# -- decision function / predict agreement ------------------------------

def test_decision_function_sign_agrees_with_predict():
    X, y = make_circles()
    Xtr, ytr, Xte, yte = split(X, y)
    model = RBFKernelSVM(C=1.0, random_state=42).fit(Xtr, ytr)
    scores = model.decision_function(Xte)
    assert scores.ndim == 1 and scores.shape[0] == len(yte)
    expected = np.where(scores >= 0, model.classes_[1], model.classes_[0])
    np.testing.assert_array_equal(model.predict(Xte), expected)


def test_string_labels_roundtrip():
    X, y = make_blobs()
    y_str = np.where(y == 0, "cat", "dog")
    model = RBFKernelSVM(random_state=42).fit(X, y_str)
    preds = model.predict(X)
    assert set(preds.tolist()) <= {"cat", "dog"}
    assert model.score(X, y_str) >= 0.9
    assert list(model.classes_) == ["cat", "dog"]


# -- support vectors / dual coefficients --------------------------------

def test_support_vectors_exist_and_dual_coef_in_range():
    X, y = make_circles()
    C = 1.0
    model = RBFKernelSVM(C=C, random_state=42).fit(X, y)
    assert model.n_support_ > 0
    assert model.n_support_ <= len(y)
    assert model.support_vectors_.shape[1] == X.shape[1]
    assert model.support_vectors_.shape[0] == model.n_support_
    assert np.all(model.dual_coef_ >= 0) and np.all(model.dual_coef_ <= C)
    assert model.support_vector_labels_.shape[0] == model.n_support_
    assert set(np.unique(model.support_vector_labels_).tolist()) <= {-1.0, 1.0}


def test_gamma_auto_and_float_work():
    X, y = make_circles()
    Xtr, ytr, Xte, yte = split(X, y)
    for gamma in ("auto", 0.1, 0.5):
        model = RBFKernelSVM(C=1.0, gamma=gamma, random_state=42).fit(Xtr, ytr)
        assert model.score(Xte, yte) >= 0.9


def test_init_exports_from_package():
    assert RBFKernelSVM is RBFKernelSVMFromModel


# -- validation errors --------------------------------------------------

def test_single_class_y_raises():
    X, _ = make_blobs()
    with pytest.raises(ValueError, match="exactly 2"):
        RBFKernelSVM().fit(X, np.zeros(len(X), dtype=int))


def test_three_class_y_raises():
    X, _ = make_blobs()
    y = np.array([0, 1, 2] * 40)
    with pytest.raises(ValueError, match="exactly 2"):
        RBFKernelSVM().fit(X, y)


def test_mismatched_X_y_raises():
    X, y = make_blobs()
    with pytest.raises(ValueError, match="same number of samples"):
        RBFKernelSVM().fit(X, y[:-1])


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        RBFKernelSVM().fit(np.empty((0, 2)), np.empty((0,), dtype=int))


def test_nan_input_raises():
    X, y = make_blobs()
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RBFKernelSVM().fit(X, y)


def test_bad_C_raises():
    for bad in (0, -1.0, "high"):
        with pytest.raises(ValueError, match="C must be"):
            RBFKernelSVM(C=bad)


def test_bad_gamma_raises():
    for bad in (0, -0.5, "rbf", None):
        with pytest.raises(ValueError, match="gamma must be"):
            RBFKernelSVM(gamma=bad)


def test_predict_before_fit_raises():
    X, _ = make_blobs()
    with pytest.raises(RuntimeError, match="not fitted"):
        RBFKernelSVM().predict(X)


def test_decision_function_before_fit_raises():
    X, _ = make_blobs()
    with pytest.raises(RuntimeError, match="not fitted"):
        RBFKernelSVM().decision_function(X)


def test_wrong_feature_count_raises():
    X, y = make_blobs()
    model = RBFKernelSVM(random_state=42).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="features"):
        model.decision_function(np.zeros((5, 3)))
