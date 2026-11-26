import numpy as np
import pytest

from passive_aggressive_classifier.model import PassiveAggressiveClassifier


def _stream(seed=42, n=300):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x, y


def test_fit_reaches_high_accuracy_on_separable_data():
    x, y = _stream()
    model = PassiveAggressiveClassifier(C=1.0, n_epochs=5, random_state=0).fit(x, y)
    assert np.mean(model.predict(x) == y) > 0.95


def test_partial_fit_online_reaches_high_accuracy():
    x, y = _stream()
    model = PassiveAggressiveClassifier(C=1.0)
    model.partial_fit(x[:1], y[:1], classes=np.array([0, 1]))
    for xi, yi in zip(x[1:], y[1:]):
        model.partial_fit(xi.reshape(1, -1), np.array([yi]))
    assert np.mean(model.predict(x) == y) >= 0.90


def test_updates_counted_and_bounded_by_examples_seen():
    x, y = _stream()
    model = PassiveAggressiveClassifier(C=1.0)
    model.partial_fit(x[:1], y[:1], classes=np.array([0, 1]))
    for xi, yi in zip(x[1:], y[1:]):
        model.partial_fit(xi.reshape(1, -1), np.array([yi]))
    assert model.n_seen_ == len(x)
    assert 0 < model.n_updates_ <= len(x)


def test_no_update_when_already_correct_with_margin():
    model = PassiveAggressiveClassifier(C=1.0)
    model.partial_fit(np.array([[10.0, 10.0]]), np.array([1]), classes=np.array([0, 1]))
    updates_after_first = model.n_updates_
    model.partial_fit(np.array([[10.0, 10.0]]), np.array([1]))
    assert model.n_updates_ == updates_after_first  # passive: no further update


def test_wrong_prediction_triggers_update():
    model = PassiveAggressiveClassifier(C=1.0)
    model.partial_fit(np.array([[10.0, 10.0]]), np.array([1]), classes=np.array([0, 1]))
    before = model.n_updates_
    model.partial_fit(np.array([[10.0, 10.0]]), np.array([0]))
    assert model.n_updates_ == before + 1


def test_smaller_c_limits_weight_movement():
    x, y = _stream()
    small = PassiveAggressiveClassifier(C=0.001, n_epochs=1, random_state=0).fit(x, y)
    large = PassiveAggressiveClassifier(C=100.0, n_epochs=1, random_state=0).fit(x, y)
    assert np.linalg.norm(small.coef_) < np.linalg.norm(large.coef_)


def test_decision_function_sign_matches_predict():
    x, y = _stream()
    model = PassiveAggressiveClassifier(C=1.0, n_epochs=3, random_state=0).fit(x, y)
    scores = model.decision_function(x)
    preds = model.predict(x)
    assert np.array_equal(preds, np.where(scores >= 0, 1, 0))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        PassiveAggressiveClassifier().predict(np.array([[0.0, 0.0]]))


def test_partial_fit_without_classes_on_first_call_raises():
    with pytest.raises(ValueError, match="classes must be given"):
        PassiveAggressiveClassifier().partial_fit(np.array([[0.0, 0.0]]), np.array([1]))


def test_partial_fit_with_wrong_class_label_raises():
    model = PassiveAggressiveClassifier(C=1.0)
    model.partial_fit(np.array([[0.0, 0.0]]), np.array([1]), classes=np.array([0, 1]))
    with pytest.raises(ValueError, match="unknown class"):
        model.partial_fit(np.array([[0.0, 0.0]]), np.array([2]))


def test_fit_with_three_classes_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        PassiveAggressiveClassifier().fit(x, y)


def test_fit_with_one_class_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0])
    with pytest.raises(ValueError, match="binary"):
        PassiveAggressiveClassifier().fit(x, y)


def test_empty_fit_raises():
    with pytest.raises(ValueError, match="empty"):
        PassiveAggressiveClassifier().fit(np.empty((0, 2)), np.array([]))


def test_mismatched_xy_raises():
    with pytest.raises(ValueError, match="same number of samples"):
        PassiveAggressiveClassifier().fit(np.zeros((5, 2)), np.zeros(4))


def test_bad_c_raises():
    with pytest.raises(ValueError, match="C must be positive"):
        PassiveAggressiveClassifier(C=0.0)


def test_bad_n_epochs_raises():
    with pytest.raises(ValueError, match="n_epochs"):
        PassiveAggressiveClassifier(n_epochs=0)


def test_wrong_feature_count_raises():
    x, y = _stream()
    model = PassiveAggressiveClassifier(C=1.0, n_epochs=2, random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((1, 3)))


def test_non_binary_string_labels_work():
    x, y = _stream()
    y_str = np.where(y == 1, "yes", "no")
    model = PassiveAggressiveClassifier(C=1.0, n_epochs=3, random_state=0).fit(x, y_str)
    assert set(model.predict(x).tolist()) <= {"yes", "no"}
    assert np.mean(model.predict(x) == y_str) > 0.95


def test_demo_runs():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "passive_aggressive_classifier/demo.py"],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "Test accuracy" in result.stdout
