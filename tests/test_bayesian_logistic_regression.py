"""Tests for bayesian_logistic_regression.BayesianLogisticRegression."""
import numpy as np
import pytest

from bayesian_logistic_regression.model import BayesianLogisticRegression


def make_blobs(seed=0, n=200, sep=2.0, noise=1.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-sep, 0], noise, size=(n // 2, 2))
    x1 = rng.normal([sep, 0], noise, size=(n // 2, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    perm = rng.permutation(n)
    return x[perm], y[perm]


def test_accuracy_above_0_9():
    x, y = make_blobs(seed=0)
    model = BayesianLogisticRegression().fit(x[:160], y[:160])
    assert model.score(x[160:], y[160:]) > 0.9


def test_predict_proba_rows_sum_to_one_and_in_range():
    x, y = make_blobs(seed=1)
    model = BayesianLogisticRegression().fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_matches_proba_threshold():
    x, y = make_blobs(seed=2)
    model = BayesianLogisticRegression().fit(x, y)
    pred = model.predict(x)
    assert np.array_equal(pred, (model.predict_proba(x)[:, 1] >= 0.5).astype(int))
    assert set(np.unique(pred)) <= {0, 1}


def test_posterior_covariance_symmetric_positive_definite():
    x, y = make_blobs(seed=3)
    model = BayesianLogisticRegression().fit(x, y)
    s = model.posterior_cov_
    assert np.allclose(s, s.T)
    assert np.all(np.linalg.eigvalsh(s) > 0)
    assert s.shape == (3, 3)  # intercept + 2 features


def test_posterior_mean_shape_and_coef():
    x, y = make_blobs(seed=4)
    model = BayesianLogisticRegression().fit(x, y)
    assert model.posterior_mean_.shape == (3,)
    assert model.coef_.shape == (2,)
    assert isinstance(model.intercept_, float)
    # First feature separates the classes, so its weight dominates.
    assert abs(model.coef_[0]) > abs(model.coef_[1])


def test_uncertainty_pulls_far_point_towards_half():
    x, y = make_blobs(seed=5)
    model = BayesianLogisticRegression().fit(x, y)
    far = np.array([[0.0, 50.0]])
    p_laplace = model.predict_proba(far)[0, 1]
    p_plugin = model.predict_proba(far, use_uncertainty=False)[0, 1]
    assert abs(p_laplace - 0.5) < abs(p_plugin - 0.5)


def test_uncertainty_negligible_near_data():
    x, y = make_blobs(seed=6)
    model = BayesianLogisticRegression().fit(x, y)
    near = np.array([[0.0, 0.0]])
    p_laplace = model.predict_proba(near)[0, 1]
    p_plugin = model.predict_proba(near, use_uncertainty=False)[0, 1]
    assert abs(p_laplace - p_plugin) < 0.05


def test_stronger_prior_shrinks_weights():
    x, y = make_blobs(seed=7)
    weak = BayesianLogisticRegression(alpha=0.01).fit(x, y)
    strong = BayesianLogisticRegression(alpha=100.0).fit(x, y)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)


def test_no_intercept_mode():
    x, y = make_blobs(seed=8)
    model = BayesianLogisticRegression(fit_intercept=False).fit(x, y)
    assert model.intercept_ == 0.0
    assert model.posterior_cov_.shape == (2, 2)
    assert model.score(x, y) > 0.85


def test_single_feature():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, size=(200, 1))
    y = (x[:, 0] > 0).astype(int)
    model = BayesianLogisticRegression().fit(x[:160], y[:160])
    assert model.score(x[160:], y[160:]) > 0.9


def test_linearly_separable_converges():
    rng = np.random.default_rng(10)
    x = np.vstack([rng.normal(-3, 0.3, size=(50, 2)),
                   rng.normal(3, 0.3, size=(50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    model = BayesianLogisticRegression().fit(x, y)
    assert model.score(x, y) == 1.0
    assert model.n_iter_ <= 100


def test_predict_before_fit_raises():
    model = BayesianLogisticRegression()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.zeros((2, 2)))


def test_value_error_non_binary_y():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 2))
    with pytest.raises(ValueError, match="only 0 and 1"):
        BayesianLogisticRegression().fit(x, np.array([0, 1, 2] * 6 + [0, 1]))


def test_value_error_single_class_y():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(20, 2))
    with pytest.raises(ValueError, match="both classes"):
        BayesianLogisticRegression().fit(x, np.zeros(20, dtype=int))


def test_value_error_1d_X():
    with pytest.raises(ValueError, match="2D"):
        BayesianLogisticRegression().fit(np.zeros(10), np.zeros(10, dtype=int))


def test_value_error_mismatched_lengths():
    with pytest.raises(ValueError, match="inconsistent lengths"):
        BayesianLogisticRegression().fit(np.zeros((5, 2)), np.zeros(4, dtype=int))


def test_value_error_empty_input():
    with pytest.raises(ValueError, match="empty"):
        BayesianLogisticRegression().fit(np.zeros((0, 2)), np.zeros(0, dtype=int))


def test_value_error_nonfinite():
    x, y = make_blobs(seed=13)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        BayesianLogisticRegression().fit(x, y)


def test_value_error_bad_params():
    with pytest.raises(ValueError):
        BayesianLogisticRegression(alpha=0)
    with pytest.raises(ValueError):
        BayesianLogisticRegression(alpha=-1)
    with pytest.raises(ValueError):
        BayesianLogisticRegression(max_iter=0)
    with pytest.raises(ValueError):
        BayesianLogisticRegression(tol=0)


def test_value_error_wrong_feature_count_at_predict():
    x, y = make_blobs(seed=14)
    model = BayesianLogisticRegression().fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 5)))
