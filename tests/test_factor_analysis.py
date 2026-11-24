import numpy as np
import pytest

from factor_analysis.model import FactorAnalysis

TRUE_W = np.array([
    [0.9, 0.1],
    [0.8, 0.2],
    [0.7, -0.1],
    [0.1, 0.9],
    [0.2, 0.8],
    [-0.1, 0.7],
])
TRUE_PSI = np.array([0.2, 0.3, 0.25, 0.2, 0.3, 0.25])
TRUE_MU = np.array([1.0, -0.5, 0.0, 0.5, -1.0, 0.25])


def _data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    eps = rng.normal(size=(n, 6)) * np.sqrt(TRUE_PSI)
    return z @ TRUE_W.T + TRUE_MU + eps


def _fit(**kwargs):
    kwargs.setdefault("n_factors", 2)
    kwargs.setdefault("random_state", 0)
    return FactorAnalysis(**kwargs).fit(_data())


def test_covariance_recovery():
    # Loadings are identified only up to rotation/sign, so compare the
    # implied covariance W W^T + Psi against the truth.
    model = _fit(max_iter=200)
    cov_hat = model.components_ @ model.components_.T + np.diag(model.noise_variance_)
    cov_true = TRUE_W @ TRUE_W.T + np.diag(TRUE_PSI)
    assert np.mean((cov_hat - cov_true) ** 2) < 0.05


def test_factor_scores_correlate_with_true_factors():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(300, 2))
    eps = rng.normal(size=(300, 6)) * np.sqrt(TRUE_PSI)
    x = z @ TRUE_W.T + TRUE_MU + eps
    model = FactorAnalysis(n_factors=2, max_iter=200, random_state=0).fit(x)
    z_hat = model.transform(x)
    # Each recovered factor must correlate strongly (|r| > 0.7) with some
    # true factor (rotation/sign may permute them).
    for j in range(2):
        cors = [abs(np.corrcoef(z_hat[:, j], z[:, t])[0, 1]) for t in range(2)]
        assert max(cors) > 0.7


def test_reconstruction_mse_small():
    x = _data()
    model = _fit(max_iter=200)
    mse = float(np.mean((x - model.reconstruct(x)) ** 2))
    assert mse < 0.3


def test_transform_shape():
    x = _data()
    model = _fit()
    assert model.transform(x).shape == (300, 2)
    assert model.transform(x[:10]).shape == (10, 2)


def test_log_likelihood_increases_with_iterations():
    x = _data()
    short = FactorAnalysis(n_factors=2, max_iter=1, random_state=0).fit(x)
    long = FactorAnalysis(n_factors=2, max_iter=20, random_state=0).fit(x)
    assert long.log_likelihood(x) > short.log_likelihood(x)
    assert long.log_likelihood_history_[-1] >= long.log_likelihood_history_[0]


def test_noise_variance_positive():
    model = _fit(max_iter=200)
    assert model.noise_variance_.shape == (6,)
    assert np.all(model.noise_variance_ > 0)
    assert np.all(np.isfinite(model.noise_variance_))


def test_determinism_same_random_state():
    x = _data()
    a = FactorAnalysis(n_factors=2, max_iter=50, random_state=7).fit(x)
    b = FactorAnalysis(n_factors=2, max_iter=50, random_state=7).fit(x)
    assert np.allclose(a.components_, b.components_)
    assert np.allclose(a.noise_variance_, b.noise_variance_)
    assert np.allclose(a.transform(x), b.transform(x))


def test_fit_rejects_1d_input():
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=2).fit(np.zeros(10))


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=2).fit(np.zeros((0, 6)))


def test_fit_rejects_nan_and_inf():
    x = _data()
    x_bad = x.copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=2).fit(x_bad)
    x_bad = x.copy()
    x_bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=2).fit(x_bad)


def test_fit_rejects_n_factors_ge_n_features():
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=6).fit(_data())
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=10).fit(_data())


def test_rejects_n_factors_below_one():
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=0)
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=-2)


def test_transform_before_fit_raises():
    model = FactorAnalysis(n_factors=2)
    with pytest.raises(ValueError):
        model.transform(_data())
    with pytest.raises(ValueError):
        model.log_likelihood(_data())


def test_transform_rejects_wrong_feature_count():
    model = _fit()
    with pytest.raises(ValueError):
        model.transform(np.zeros((5, 4)))


def test_fit_rejects_too_few_samples():
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=2).fit(np.zeros((2, 6)))


def test_score_is_mean_log_likelihood():
    x = _data()
    model = _fit(max_iter=50)
    assert np.isclose(model.score(x), model.log_likelihood(x) / len(x))
    assert np.isfinite(model.score(x))
