import numpy as np
import pytest

from ai_slop_detector.classifier import (
    LogisticRegression,
    StandardScaler,
    classification_metrics,
    confusion_matrix,
)


def test_standard_scaler_zero_mean_unit_variance():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)


def test_standard_scaler_handles_constant_column_without_nan():
    # A column with zero variance would divide by zero if unguarded.
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.all(np.isfinite(X_scaled))
    assert np.allclose(X_scaled[:, 0], 0.0)


def test_scaler_transform_before_fit_raises():
    scaler = StandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_logistic_regression_learns_linearly_separable_data():
    rng = np.random.default_rng(0)
    X_pos = rng.normal(loc=3.0, scale=0.5, size=(50, 2))
    X_neg = rng.normal(loc=-3.0, scale=0.5, size=(50, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [0] * 50)

    model = LogisticRegression(lr=0.5, epochs=500, l2=0.0)
    model.fit(X, y)
    preds = model.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.95


def test_logistic_regression_loss_decreases():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(lr=0.3, epochs=300, l2=0.01)
    model.fit(X, y)
    assert model.loss_history[-1] < model.loss_history[0]


def test_predict_proba_before_fit_raises():
    model = LogisticRegression()
    with pytest.raises(RuntimeError):
        model.predict_proba(np.array([[1.0, 2.0]]))


def test_fit_rejects_non_binary_labels():
    model = LogisticRegression()
    X = np.zeros((3, 2))
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_rejects_mismatched_shapes():
    model = LogisticRegression()
    X = np.zeros((3, 2))
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_confusion_matrix_values():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    cm = confusion_matrix(y_true, y_pred)
    assert cm == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_classification_metrics_perfect_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_classification_metrics_handles_zero_denominator_gracefully():
    # All-negative predictions -> precision denominator (tp+fp) is 0.
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
