import numpy as np
import pytest

from bagging_classifier.model import BaggingClassifier, _DecisionTree


def _moons(seed=42, n=200, noise=0.3):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta = rng.uniform(0, np.pi, size=n_half)
    upper = np.column_stack([np.cos(theta), np.sin(theta)])
    lower = np.column_stack([1.0 - np.cos(theta), 0.5 - np.sin(theta)])
    x = np.vstack([upper, lower]) + rng.normal(0, noise, size=(n, 2))
    y = np.array([0] * n_half + [1] * n_half)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def _blobs(seed=1, n=120):
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal([0, 0], 0.7, size=(n // 2, 2)),
                   rng.normal([4, 4], 0.7, size=(n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return x, y


def test_fits_and_predicts_blobs():
    x, y = _blobs()
    model = BaggingClassifier(n_estimators=5, random_state=0).fit(x, y)
    preds = model.predict(x)
    assert preds.shape == y.shape
    assert np.mean(preds == y) > 0.95


def test_bagged_beats_or_matches_single_tree_on_moons():
    x, y = _moons()
    split = int(0.8 * len(y))
    tree = _DecisionTree().fit(x[:split], y[:split])
    single = np.mean(tree.predict(x[split:]) == y[split:])
    bagged = BaggingClassifier(n_estimators=25, random_state=42)
    bagged.fit(x[:split], y[:split])
    score = np.mean(bagged.predict(x[split:]) == y[split:])
    assert score >= single - 0.05
    assert score > 0.85


def test_n_estimators_is_respected():
    x, y = _blobs()
    model = BaggingClassifier(n_estimators=7, random_state=0).fit(x, y)
    assert len(model.estimators_) == 7


def test_predict_proba_rows_sum_to_one():
    x, y = _blobs()
    model = BaggingClassifier(n_estimators=5, random_state=0).fit(x, y)
    proba = model.predict_proba(x[:10])
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_proba_agrees_with_predict():
    x, y = _blobs()
    model = BaggingClassifier(n_estimators=9, random_state=0).fit(x, y)
    preds = model.predict(x)
    proba = model.predict_proba(x)
    assert np.array_equal(preds, model.classes_[np.argmax(proba, axis=1)])


def test_reproducible_with_same_seed():
    x, y = _blobs()
    a = BaggingClassifier(n_estimators=5, random_state=3).fit(x, y)
    b = BaggingClassifier(n_estimators=5, random_state=3).fit(x, y)
    assert np.array_equal(a.predict(x), b.predict(x))


def test_single_sample_predict_shape():
    x, y = _blobs()
    model = BaggingClassifier(n_estimators=3, random_state=0).fit(x, y)
    assert model.predict(np.array([0.1, 0.1])).shape == (1,)


def test_max_features_subset_works():
    x, y = _blobs()
    model = BaggingClassifier(n_estimators=5, max_features=1, random_state=0)
    model.fit(x, y)
    assert np.mean(model.predict(x) == y) > 0.9


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        BaggingClassifier().predict(np.array([[0.0, 0.0]]))


def test_empty_fit_raises():
    with pytest.raises(ValueError, match="empty"):
        BaggingClassifier().fit(np.empty((0, 2)), np.array([]))


def test_mismatched_xy_raises():
    with pytest.raises(ValueError, match="same number of samples"):
        BaggingClassifier().fit(np.zeros((5, 2)), np.zeros(4))


def test_1d_x_raises():
    with pytest.raises(ValueError, match="2D"):
        BaggingClassifier().fit(np.zeros(5), np.zeros(5))


def test_single_class_raises():
    with pytest.raises(ValueError, match="at least 2 classes"):
        BaggingClassifier().fit(np.zeros((5, 2)), np.zeros(5))


def test_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        BaggingClassifier().fit(np.array([[np.nan, 0.0]]), np.array([0]))


def test_bad_n_estimators_raises():
    with pytest.raises(ValueError, match="n_estimators"):
        BaggingClassifier(n_estimators=0)


def test_bad_max_depth_raises():
    with pytest.raises(ValueError, match="max_depth"):
        BaggingClassifier(max_depth=0)


def test_wrong_feature_count_raises():
    x, y = _blobs()
    model = BaggingClassifier(n_estimators=3, random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 3)))


def test_demo_runs():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "bagging_classifier/demo.py"],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "Bagged (25 trees)" in result.stdout
