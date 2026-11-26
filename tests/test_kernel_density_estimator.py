import numpy as np
import pytest

from kernel_density_estimator.model import GaussianKDE


def _bimodal(seed=42, n=200):
    rng = np.random.default_rng(seed)
    half = n // 2
    return np.concatenate([
        rng.normal(-2.0, 0.8, size=half),
        rng.normal(2.5, 1.0, size=n - half),
    ])


def test_density_peaks_near_modes():
    kde = GaussianKDE().fit(_bimodal())
    d = kde.score(np.array([-2.0, 0.0, 2.5]))
    assert d[0] > d[1]
    assert d[2] > d[1]


def test_density_is_positive_and_finite():
    kde = GaussianKDE().fit(_bimodal())
    d = kde.score(np.linspace(-6, 6, 25))
    assert np.all(d > 0)
    assert np.all(np.isfinite(d))


def test_score_samples_is_log_of_score():
    kde = GaussianKDE().fit(_bimodal())
    pts = np.array([-3.0, 0.5, 4.0])
    np.testing.assert_allclose(kde.score_samples(pts), np.log(kde.score(pts)))


def test_density_integrates_to_about_one():
    kde = GaussianKDE().fit(_bimodal())
    grid = np.linspace(-10, 10, 2001)
    integral = float(np.trapezoid(kde.score(grid), grid))
    assert integral == pytest.approx(1.0, abs=0.02)


def test_silverman_bandwidth_is_positive():
    kde = GaussianKDE().fit(_bimodal())
    assert kde.bandwidth_ > 0


def test_fixed_bandwidth_is_respected():
    kde = GaussianKDE(bandwidth=0.25).fit(_bimodal())
    assert kde.bandwidth_ == 0.25


def test_narrower_bandwidth_gives_sharper_peak():
    pts = np.array([-2.0])
    narrow = GaussianKDE(bandwidth=0.1).fit(_bimodal()).score(pts)[0]
    wide = GaussianKDE(bandwidth=2.0).fit(_bimodal()).score(pts)[0]
    assert narrow > wide


def test_sample_returns_requested_shape_and_near_data():
    kde = GaussianKDE().fit(_bimodal())
    samples = kde.sample(50, random_state=0)
    assert samples.shape == (50, 1)
    assert samples.min() > -8 and samples.max() < 9


def test_sample_is_reproducible_with_seed():
    kde = GaussianKDE().fit(_bimodal())
    a = kde.sample(10, random_state=7)
    b = kde.sample(10, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_mode_lands_near_a_true_mode():
    kde = GaussianKDE().fit(_bimodal())
    mode = kde.mode()
    assert mode.shape == (1,)
    assert min(abs(mode[0] - (-2.0)), abs(mode[0] - 2.5)) < 0.5


def test_2d_data_works():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(100, 2))
    kde = GaussianKDE().fit(x)
    d = kde.score(np.array([[0.0, 0.0], [5.0, 5.0]]))
    assert d[0] > d[1]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GaussianKDE().score(np.array([0.0]))


def test_empty_fit_raises():
    with pytest.raises(ValueError, match="empty"):
        GaussianKDE().fit(np.array([]))


def test_single_sample_fit_raises():
    with pytest.raises(ValueError, match="at least 2 samples"):
        GaussianKDE().fit(np.array([1.0]))


def test_nan_fit_raises():
    with pytest.raises(ValueError, match="NaN"):
        GaussianKDE().fit(np.array([0.0, np.nan, 1.0]))


def test_wrong_feature_count_raises():
    kde = GaussianKDE().fit(_bimodal())
    with pytest.raises(ValueError, match="features"):
        kde.score(np.array([[0.0, 0.0]]))


def test_non_positive_bandwidth_raises():
    with pytest.raises(ValueError, match="positive"):
        GaussianKDE(bandwidth=0.0)


def test_bad_bandwidth_string_raises():
    with pytest.raises(ValueError, match="bandwidth"):
        GaussianKDE(bandwidth="scott")


def test_sample_count_validation():
    kde = GaussianKDE().fit(_bimodal())
    with pytest.raises(ValueError, match="at least 1"):
        kde.sample(0)


def test_demo_runs():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "kernel_density_estimator/demo.py"],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "Estimated mode" in result.stdout
