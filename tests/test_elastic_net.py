import numpy as np
import pytest

from elastic_net.model import ElasticNetRegression


def _data(seed=42, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    x[:, 1] = x[:, 0] + rng.normal(scale=0.3, size=n)
    y = 2.0 * x[:, 0] - 1.5 * x[:, 2] + rng.normal(scale=0.5, size=n)
    return x, y


def test_recovers_coefficients_with_small_alpha():
    x, y = _data()
    model = ElasticNetRegression(alpha=0.1, l1_ratio=0.5).fit(x, y)
    assert model.coef_[0] == pytest.approx(2.0, abs=0.3)
    assert model.coef_[2] == pytest.approx(-1.5, abs=0.3)


def test_large_alpha_zeroes_everything():
    x, y = _data()
    model = ElasticNetRegression(alpha=10000.0).fit(x, y)
    assert np.all(model.coef_ == 0.0)
    assert model.predict(x).std() == pytest.approx(0.0, abs=1e-12)


def test_l1_ratio_one_matches_lasso_behaviour():
    x, y = _data()
    model = ElasticNetRegression(alpha=50.0, l1_ratio=1.0).fit(x, y)
    assert np.all(model.coef_[3:] == 0.0)  # noise features zeroed


def test_l1_ratio_zero_keeps_all_features():
    x, y = _data()
    model = ElasticNetRegression(alpha=1.0, l1_ratio=0.0).fit(x, y)
    assert np.all(np.abs(model.coef_) > 0)  # ridge never zeroes exactly


def test_alpha_zero_matches_ols():
    x, y = _data()
    model = ElasticNetRegression(alpha=0.0, tol=1e-8).fit(x, y)
    ols, *_ = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)
    assert np.allclose(model.coef_, ols[:-1], atol=1e-2)


def test_score_is_high_on_signal_data():
    x, y = _data()
    model = ElasticNetRegression(alpha=0.5).fit(x, y)
    assert model.score(x, y) > 0.9


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        ElasticNetRegression().predict(np.zeros((2, 5)))


def test_bad_l1_ratio_raises():
    with pytest.raises(ValueError, match="l1_ratio"):
        ElasticNetRegression(l1_ratio=1.5)
    with pytest.raises(ValueError, match="l1_ratio"):
        ElasticNetRegression(l1_ratio=-0.1)


def test_negative_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        ElasticNetRegression(alpha=-1.0)


def test_mismatched_shapes_raise():
    x, y = _data()
    with pytest.raises(ValueError, match="same number of samples"):
        ElasticNetRegression().fit(x, y[:10])
    with pytest.raises(ValueError, match="2D"):
        ElasticNetRegression().fit(x.ravel(), y)
    with pytest.raises(ValueError, match="1D"):
        ElasticNetRegression().fit(x, y.reshape(-1, 1))


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        ElasticNetRegression().fit(np.zeros((0, 3)), np.zeros(0))


def test_nan_input_raises():
    x, y = _data()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ElasticNetRegression().fit(x, y)


def test_constant_feature_handled():
    x, y = _data()
    x[:, 4] = 7.0  # constant column must not divide by zero
    model = ElasticNetRegression(alpha=0.5).fit(x, y)
    assert np.isfinite(model.coef_).all()
    assert model.score(x, y) > 0.9


def test_fit_returns_self_and_predict_shape():
    x, y = _data()
    model = ElasticNetRegression(alpha=1.0)
    assert model.fit(x, y) is model
    preds = model.predict(x[:5])
    assert preds.shape == (5,)
