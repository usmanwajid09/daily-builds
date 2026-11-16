"""Tests for QDAClassifier."""

import numpy as np
import pytest

from qda_classifier import QDAClassifier


def make_three_classes(seed=42, n_per_class=100):
    """3 Gaussian classes with clearly different covariances."""
    rng = np.random.default_rng(seed)
    specs = [
        (np.array([-3.0, -3.0]), np.array([[2.5, 0.8], [0.8, 0.4]])),
        (np.array([3.0, -3.0]), np.array([[0.4, 0.8], [0.8, 2.5]])),
        (np.array([0.0, 4.0]), np.array([[0.6, 0.0], [0.0, 0.6]])),
    ]
    Xs, ys = [], []
    for label, (mean, cov) in enumerate(specs):
        Xs.append(rng.multivariate_normal(mean, cov, size=n_per_class))
        ys.append(np.full(n_per_class, label))
    return np.vstack(Xs), np.concatenate(ys)


def shared_covariance_predict(X_train, y_train, X_test):
    """Quick LDA-style baseline: one pooled covariance for all classes."""
    classes = np.unique(y_train)
    means = np.stack([X_train[y_train == c].mean(axis=0) for c in classes])
    pooled = np.zeros((X_train.shape[1], X_train.shape[1]))
    for c in classes:
        Xc = X_train[y_train == c] - means[list(classes).index(c)]
        pooled += Xc.T @ Xc
    pooled /= len(X_train) - len(classes)
    inv = np.linalg.pinv(pooled)
    scores = np.empty((len(X_test), len(classes)))
    for k, mu in enumerate(means):
        diff = X_test - mu
        scores[:, k] = -0.5 * np.einsum("ij,jk,ik->i", diff, inv, diff)
    return classes[np.argmax(scores, axis=1)]


def test_accuracy_on_different_covariances():
    X, y = make_three_classes()
    clf = QDAClassifier().fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    assert acc >= 0.90, f"accuracy {acc:.3f} below 0.90"
    assert clf.classes_.tolist() == [0, 1, 2]
    assert clf.means_.shape == (3, 2)
    assert clf.covariances_.shape == (3, 2, 2)
    assert clf.priors_.shape == (3,)
    assert np.isclose(clf.priors_.sum(), 1.0)


def test_predict_proba_shape_and_row_sums():
    X, y = make_three_classes()
    clf = QDAClassifier().fit(X, y)
    probs = clf.predict_proba(X[:10])
    assert probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    # predict and predict_proba must agree on the argmax.
    assert np.array_equal(clf.predict(X[:10]), clf.classes_[probs.argmax(axis=1)])


def test_beats_shared_covariance_on_same_mean_different_variance():
    # One tight class sitting inside a wide class, same mean: no linear
    # boundary can separate them, but QDA's per-class covariance can.
    rng = np.random.default_rng(7)
    n = 200
    X_tight = rng.multivariate_normal([0, 0], [[0.25, 0], [0, 0.25]], size=n)
    X_wide = rng.multivariate_normal([0, 0], [[4.0, 0], [0, 4.0]], size=n)
    X = np.vstack([X_tight, X_wide])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    clf = QDAClassifier().fit(X, y)
    qda_acc = np.mean(clf.predict(X) == y)
    baseline_acc = np.mean(shared_covariance_predict(X, y, X) == y)

    assert qda_acc > 0.7, f"QDA accuracy {qda_acc:.3f} not above 0.7"
    assert qda_acc > baseline_acc + 0.1, (
        f"QDA {qda_acc:.3f} should clearly beat shared-covariance "
        f"baseline {baseline_acc:.3f}"
    )


def test_singular_covariance_does_not_crash():
    # Duplicate feature column -> every class covariance is singular.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 2))
    X = np.column_stack([X, X[:, 0]])  # third column == first column
    y = np.array([0] * 30 + [1] * 30)
    clf = QDAClassifier().fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    assert preds.shape == (60,)
    assert set(np.unique(preds)) <= {0, 1}
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_predict_before_fit_raises():
    clf = QDAClassifier()
    X = np.zeros((2, 2))
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.predict_proba(X)


def test_invalid_reg_raises():
    with pytest.raises(ValueError):
        QDAClassifier(reg=-1e-6)
    with pytest.raises(ValueError):
        QDAClassifier(reg="bad")


def test_invalid_priors_raise():
    with pytest.raises(ValueError):
        QDAClassifier(priors=[0.5, 0.5, 0.5])  # sums to 1.5
    with pytest.raises(ValueError):
        QDAClassifier(priors=[0.0, 1.0])  # not positive
    with pytest.raises(ValueError):
        QDAClassifier(priors=[-0.2, 1.2])  # negative entry
    # Wrong length for the number of classes: caught at fit time.
    X, y = make_three_classes()
    with pytest.raises(ValueError):
        QDAClassifier(priors=[0.5, 0.5]).fit(X, y)


def test_mismatched_xy_raises():
    X, y = make_three_classes()
    with pytest.raises(ValueError):
        QDAClassifier().fit(X, y[:-1])
    with pytest.raises(ValueError):
        QDAClassifier().fit(X.reshape(-1), y)  # X not 2D
    with pytest.raises(ValueError):
        QDAClassifier().fit(X, y.reshape(-1, 1))  # y not 1D
    with pytest.raises(ValueError):
        QDAClassifier().fit(np.array([[1.0]]), np.array([0]))  # 1 class only


def test_string_labels():
    rng = np.random.default_rng(11)
    X = np.vstack([
        rng.multivariate_normal([-2, 0], np.eye(2) * 0.5, size=50),
        rng.multivariate_normal([2, 0], np.eye(2) * 0.5, size=50),
    ])
    y = np.array(["cat"] * 50 + ["dog"] * 50)
    clf = QDAClassifier().fit(X, y)
    preds = clf.predict(X)
    assert set(np.unique(preds)) <= {"cat", "dog"}
    assert np.mean(preds == y) >= 0.90
    assert list(clf.classes_) == ["cat", "dog"]
