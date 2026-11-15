import numpy as np
import pytest

from pca_anomaly_detector.model import PCAAnomalyDetector


def make_normal(n=200, seed=0):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=n)
    return np.column_stack([
        3.0 * factor + rng.normal(scale=0.2, size=n),
        3.0 * factor + rng.normal(scale=0.2, size=n),
    ])


OUTLIERS = np.array([[8.0, -8.0], [-8.0, 8.0], [0.0, 8.0], [8.0, 0.0]])


@pytest.fixture
def detector():
    return PCAAnomalyDetector(n_components=1, threshold="std", k=3.0).fit(make_normal())


def test_fit_sets_threshold(detector):
    assert detector.threshold_ is not None
    assert detector.threshold_ > 0
    assert detector.train_errors_.shape == (200,)


def test_outliers_have_higher_error_than_normal(detector):
    normal = make_normal(n=20, seed=1)
    assert detector.score_samples(OUTLIERS).min() > detector.score_samples(normal).max()


def test_outliers_flagged(detector):
    assert detector.is_anomaly(OUTLIERS).all()


def test_most_normal_points_not_flagged(detector):
    normal = make_normal(n=200, seed=2)
    assert detector.is_anomaly(normal).mean() < 0.05


def test_predict_returns_one_and_minus_one(detector):
    normal = make_normal(n=20, seed=3)
    preds = detector.predict(np.vstack([normal, OUTLIERS]))
    assert set(np.unique(preds)) <= {1, -1}
    # The std threshold has a small tail: allow at most 2/20 false positives.
    assert (preds[:20] == 1).sum() >= 18
    assert (preds[20:] == -1).all()


def test_quantile_threshold_rule():
    det = PCAAnomalyDetector(n_components=1, threshold="quantile", quantile=0.99).fit(make_normal())
    assert np.isclose(det.threshold_, np.quantile(det.train_errors_, 0.99))


def test_explicit_threshold():
    det = PCAAnomalyDetector(n_components=1, threshold=0.5).fit(make_normal())
    assert det.threshold_ == 0.5
    assert det.is_anomaly(OUTLIERS).all()


def test_score_shape(detector):
    assert detector.score_samples(make_normal(n=7, seed=4)).shape == (7,)


def test_train_errors_small_for_correlated_data(detector):
    assert detector.train_errors_.mean() < 0.1


def test_invalid_n_components_zero():
    with pytest.raises(ValueError):
        PCAAnomalyDetector(n_components=0)


def test_n_components_equals_features_raises():
    with pytest.raises(ValueError):
        PCAAnomalyDetector(n_components=2).fit(make_normal())


def test_invalid_threshold_string_raises():
    with pytest.raises(ValueError):
        PCAAnomalyDetector(threshold="median")


def test_negative_explicit_threshold_raises():
    with pytest.raises(ValueError):
        PCAAnomalyDetector(threshold=-1.0)


def test_invalid_quantile_raises():
    with pytest.raises(ValueError):
        PCAAnomalyDetector(threshold="quantile", quantile=1.5)


def test_score_before_fit_raises():
    with pytest.raises(ValueError):
        PCAAnomalyDetector().score_samples(make_normal(n=5, seed=5))


def test_wrong_feature_count_raises(detector):
    with pytest.raises(ValueError):
        detector.score_samples(np.zeros((5, 3)))


def test_nonfinite_input_raises():
    X = make_normal()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        PCAAnomalyDetector(n_components=1).fit(X)
