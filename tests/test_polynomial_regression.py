import numpy as np
import pytest

from polynomial_regression.model import PolynomialRegression, select_degree


def quad_data(n=100, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    y = 2 + 3 * x - x**2 + rng.normal(scale=noise, size=n)
    return x, y


def test_recovers_quadratic_coefficients_noiseless():
    x, y = quad_data(noise=0.0)
    model = PolynomialRegression(degree=2, alpha=0.0).fit(x, y)
    assert model.score(x, y) >= 0.99
    np.testing.assert_allclose(model.coef_, [2, 3, -1], atol=1e-6)


def test_degree_one_on_linear_data():
    rng = np.random.default_rng(1)
    x = rng.uniform(-5, 5, size=80)
    y = 1.5 - 2.0 * x + rng.normal(scale=0.1, size=80)
    model = PolynomialRegression(degree=1, alpha=0.0).fit(x, y)
    assert model.score(x, y) >= 0.99


def test_ridge_shrinks_coefficients():
    x, y = quad_data(n=60, noise=1.0, seed=2)
    plain = PolynomialRegression(degree=5, alpha=0.0).fit(x, y)
    shrunk = PolynomialRegression(degree=5, alpha=100.0).fit(x, y)
    assert np.linalg.norm(shrunk.coef_[1:]) < np.linalg.norm(plain.coef_[1:])


def test_select_degree_picks_two_for_quadratic():
    rng = np.random.default_rng(4)
    x = rng.uniform(-3, 3, size=600)
    y = 2 + 3 * x - x**2 + rng.normal(scale=0.5, size=600)
    best, best_mse, results = select_degree(
        x[:400], y[:400], x[400:], y[400:], degrees=range(1, 6), alpha=1.0
    )
    assert best == 2
    assert set(results) == {1, 2, 3, 4, 5}
    assert best_mse == results[2]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        PolynomialRegression().predict([1.0, 2.0])


def test_invalid_degree_and_alpha_raise():
    with pytest.raises(ValueError):
        PolynomialRegression(degree=0)
    with pytest.raises(ValueError):
        PolynomialRegression(degree=2.5)
    with pytest.raises(ValueError):
        PolynomialRegression(alpha=-1.0)


def test_predict_output_shape_matches_input():
    x, y = quad_data(n=50)
    model = PolynomialRegression(degree=2).fit(x, y)
    assert model.predict(x).shape == (50,)
    assert model.predict(x.reshape(-1, 1)).shape == (50,)
    assert model.predict([0.0, 1.0, 2.0]).shape == (3,)


def test_accepts_1d_and_2d_x_with_same_predictions():
    x, y = quad_data(n=50)
    m1 = PolynomialRegression(degree=3).fit(x, y)
    m2 = PolynomialRegression(degree=3).fit(x.reshape(-1, 1), y)
    np.testing.assert_allclose(m1.predict(x), m2.predict(x.reshape(-1, 1)))


def test_input_validation_errors():
    x, y = quad_data(n=20)
    with pytest.raises(ValueError):
        PolynomialRegression().fit(x[:10], y)  # mismatched lengths
    with pytest.raises(ValueError):
        PolynomialRegression().fit(np.array([]), np.array([]))  # empty
    with pytest.raises(ValueError):
        PolynomialRegression().fit(np.array([[1, 2], [3, 4]]), [1, 2])  # 2 features
    bad = x.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError):
        PolynomialRegression().fit(bad, y)  # non-finite
