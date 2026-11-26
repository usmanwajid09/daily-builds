import numpy as np
import pytest

from feature_selector.model import (
    correlation_scores,
    mutual_info_scores,
    select_top_k,
    variance_threshold,
)


def _data(seed=42, n=400):
    rng = np.random.default_rng(seed)
    informative = rng.normal(0, 1, size=(n, 2))
    y = (informative[:, 0] + 0.5 * informative[:, 1] > 0).astype(int)
    noise = rng.normal(0, 1, size=(n, 4))
    return np.hstack([informative, noise]), y


def test_mi_top2_are_informative():
    x, y = _data()
    top2 = select_top_k(mutual_info_scores(x, y), 2)
    assert set(top2.tolist()) == {0, 1}


def test_mi_informative_beats_noise():
    x, y = _data()
    scores = mutual_info_scores(x, y)
    assert scores[0] > scores[2:].max()
    assert scores[1] > scores[2:].max()


def test_mi_scores_non_negative():
    x, y = _data()
    assert np.all(mutual_info_scores(x, y) >= 0)


def test_mi_constant_feature_scores_zero():
    rng = np.random.default_rng(0)
    x = np.hstack([rng.normal(0, 1, size=(100, 1)), np.ones((100, 1))])
    y = rng.integers(0, 2, size=100)
    scores = mutual_info_scores(x, y)
    assert scores[1] == 0.0


def test_correlation_top2_are_informative():
    x, y = _data()
    top2 = select_top_k(correlation_scores(x, y.astype(float)), 2)
    assert set(top2.tolist()) == {0, 1}


def test_correlation_scores_in_unit_range():
    x, y = _data()
    scores = correlation_scores(x, y.astype(float))
    assert np.all((scores >= 0) & (scores <= 1))


def test_correlation_perfect_linear_scores_one():
    x = np.column_stack([np.arange(10, dtype=float), np.arange(10, dtype=float) * 2])
    y = np.arange(10, dtype=float)
    scores = correlation_scores(x, y)
    np.testing.assert_allclose(scores, [1.0, 1.0])


def test_variance_threshold_drops_constant():
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    mask = variance_threshold(x, threshold=0.0)
    assert mask.tolist() == [True, False]


def test_variance_threshold_with_positive_threshold():
    x = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    mask = variance_threshold(x, threshold=0.05)
    assert mask.tolist() == [False, False]


def test_select_top_k_returns_ranked_order():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    assert select_top_k(scores, 2).tolist() == [1, 2]


def test_select_top_k_all_features():
    scores = np.array([0.2, 0.8, 0.4])
    assert select_top_k(scores, 3).tolist() == [1, 2, 0]


def test_select_top_k_k_too_large_raises():
    with pytest.raises(ValueError, match="exceeds"):
        select_top_k(np.array([0.1, 0.2]), 3)


def test_select_top_k_k_zero_raises():
    with pytest.raises(ValueError, match="at least 1"):
        select_top_k(np.array([0.1, 0.2]), 0)


def test_select_top_k_non_1d_raises():
    with pytest.raises(ValueError, match="1D"):
        select_top_k(np.zeros((2, 2)), 1)


def test_variance_threshold_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        variance_threshold(np.zeros((3, 2)), threshold=-1.0)


def test_mi_n_bins_validation():
    x, y = _data()
    with pytest.raises(ValueError, match="n_bins"):
        mutual_info_scores(x, y, n_bins=1)


def test_mismatched_xy_raises():
    x, y = _data()
    with pytest.raises(ValueError, match="same number of samples"):
        mutual_info_scores(x, y[:10])


def test_correlation_zero_variance_y_raises():
    x, _ = _data()
    with pytest.raises(ValueError, match="zero variance"):
        correlation_scores(x, np.ones(len(x)))


def test_empty_x_raises():
    with pytest.raises(ValueError, match="empty"):
        variance_threshold(np.empty((0, 2)))


def test_demo_runs():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "feature_selector/demo.py"],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "Top-2 features by MI: [0, 1]" in result.stdout
