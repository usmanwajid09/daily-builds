"""Tests for one_class_svm.OneClassSVM."""
import numpy as np
import pytest

from one_class_svm.model import OneClassSVM


def make_blob(seed=0, n=200, dim=2, scale=1.0, shift=0.0):
    rng = np.random.default_rng(seed)
    return rng.normal(shift, scale, size=(n, dim))


def test_detects_far_outliers():
    x_train = make_blob(seed=0)
    rng = np.random.default_rng(1)
    x_outlier = rng.normal(0, 1, size=(30, 2)) + np.array([8.0, 8.0])
    model = OneClassSVM(nu=0.1, random_state=0).fit(x_train)
    assert float(np.mean(model.predict(x_outlier) == -1)) >= 0.9


def test_most_train_points_are_inliers():
    x_train = make_blob(seed=2)
    model = OneClassSVM(nu=0.1, random_state=0).fit(x_train)
    rate = float(np.mean(model.predict(x_train) == 1))
    assert 0.8 <= rate <= 0.98


def test_fresh_inliers_mostly_normal():
    x_train = make_blob(seed=3)
    x_fresh = make_blob(seed=4, n=100)
    model = OneClassSVM(nu=0.1, random_state=0).fit(x_train)
    assert float(np.mean(model.predict(x_fresh) == 1)) >= 0.8


def test_decision_scores_ordered():
    x_train = make_blob(seed=5)
    rng = np.random.default_rng(6)
    x_normal = make_blob(seed=7, n=50)
    x_outlier = rng.normal(0, 1, size=(50, 2)) + np.array([8.0, 8.0])
    model = OneClassSVM(nu=0.1, random_state=0).fit(x_train)
    assert model.decision_function(x_normal).mean() > model.decision_function(x_outlier).mean()
    assert model.score_samples(x_normal).mean() > model.score_samples(x_outlier).mean()


def test_predict_returns_only_plus_minus_one():
    x_train = make_blob(seed=8)
    model = OneClassSVM(nu=0.1, random_state=0).fit(x_train)
    pred = model.predict(x_train)
    assert set(np.unique(pred)) <= {1, -1}
    assert pred.shape == (len(x_train),)


def test_higher_nu_flags_more():
    x_train = make_blob(seed=9)
    low = OneClassSVM(nu=0.05, random_state=0).fit(x_train)
    high = OneClassSVM(nu=0.3, random_state=0).fit(x_train)
    assert float(np.mean(low.predict(x_train) == -1)) < float(np.mean(high.predict(x_train) == -1))


def test_dual_coef_sums_near_one():
    x_train = make_blob(seed=10)
    model = OneClassSVM(nu=0.1, random_state=0).fit(x_train)
    assert abs(float(model.dual_coef_.sum()) - 1.0) < 0.25
    assert np.all(model.dual_coef_ >= 0)
    assert np.all(model.dual_coef_ <= 1.0 / (0.1 * len(x_train)) + 1e-9)


def test_gamma_float_and_auto():
    x_train = make_blob(seed=11)
    for gamma in (0.5, "auto"):
        model = OneClassSVM(nu=0.1, gamma=gamma, random_state=0).fit(x_train)
        assert model.gamma_ > 0
        assert model.predict(x_train).shape == (len(x_train),)


def test_reproducible_with_random_state():
    x_train = make_blob(seed=12)
    m1 = OneClassSVM(nu=0.1, random_state=5).fit(x_train)
    m2 = OneClassSVM(nu=0.1, random_state=5).fit(x_train)
    assert np.allclose(m1.decision_function(x_train), m2.decision_function(x_train))


def test_1d_data():
    x_train = make_blob(seed=13, dim=1, n=150)
    rng = np.random.default_rng(14)
    x_outlier = rng.normal(0, 1, size=(20, 1)) + 8.0
    model = OneClassSVM(nu=0.1, random_state=0).fit(x_train)
    assert float(np.mean(model.predict(x_outlier) == -1)) >= 0.9


def test_near_outliers_partly_flagged():
    x_train = make_blob(seed=15)
    rng = np.random.default_rng(16)
    x_near = rng.normal(0, 1, size=(50, 2)) + np.array([3.0, 3.0])
    x_far = rng.normal(0, 1, size=(50, 2)) + np.array([8.0, 8.0])
    model = OneClassSVM(nu=0.1, random_state=0).fit(x_train)
    near_rate = float(np.mean(model.predict(x_near) == -1))
    far_rate = float(np.mean(model.predict(x_far) == -1))
    assert far_rate >= near_rate


def test_predict_before_fit_raises():
    model = OneClassSVM()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.zeros((2, 2)))


def test_value_error_1d_X():
    with pytest.raises(ValueError, match="2D"):
        OneClassSVM().fit(np.zeros(10))


def test_value_error_empty_input():
    with pytest.raises(ValueError, match="empty"):
        OneClassSVM().fit(np.zeros((0, 2)))


def test_value_error_nonfinite():
    x = make_blob(seed=17)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        OneClassSVM().fit(x)


def test_value_error_bad_params():
    with pytest.raises(ValueError):
        OneClassSVM(nu=0)
    with pytest.raises(ValueError):
        OneClassSVM(nu=1.5)
    with pytest.raises(ValueError):
        OneClassSVM(gamma=-1.0)
    with pytest.raises(ValueError):
        OneClassSVM(gamma="bogus")
    with pytest.raises(ValueError):
        OneClassSVM(learning_rate=0)
    with pytest.raises(ValueError):
        OneClassSVM(epochs=0)
    with pytest.raises(ValueError):
        OneClassSVM(batch_size=0)


def test_value_error_constant_data_gamma_scale():
    x = np.ones((20, 2))
    with pytest.raises(ValueError, match="non-constant"):
        OneClassSVM().fit(x)


def test_value_error_wrong_feature_count_at_predict():
    x_train = make_blob(seed=18)
    model = OneClassSVM(nu=0.1, random_state=0).fit(x_train)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 5)))
