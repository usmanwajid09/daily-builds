"""Tests for smote_oversampler (SMOTE from scratch, numpy only)."""
import numpy as np
import pytest

from smote_oversampler import SMOTE, fit_resample


def make_imbalanced(seed=0, n_majority=20, n_minority=6, n_features=2):
    rng = np.random.default_rng(seed)
    X_maj = rng.normal(0.0, 1.0, size=(n_majority, n_features))
    X_min = rng.normal(4.0, 1.0, size=(n_minority, n_features))
    X = np.vstack([X_maj, X_min])
    y = np.array([0] * n_majority + [1] * n_minority)
    return X, y


def counts_of(y):
    labels, counts = np.unique(y, return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))


def on_segment(point, samples, tol=1e-9):
    """True if point lies on the segment between some pair of samples."""
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a, b = samples[i], samples[j]
            ab = b - a
            denom = float(ab @ ab)
            if denom == 0.0:
                continue
            t = float((point - a) @ ab) / denom
            if -tol <= t <= 1 + tol:
                closest = a + t * ab
                if np.linalg.norm(point - closest) <= 1e-9:
                    return True
    return False


# --- core behaviour -------------------------------------------------------

def test_balances_to_majority_count():
    X, y = make_imbalanced()
    X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
    assert counts_of(y_res) == {0: 20, 1: 20}
    assert X_res.shape == (40, 2)
    assert len(y_res) == 40


def test_float_ratio_target():
    X, y = make_imbalanced(n_majority=20, n_minority=6)
    X_res, y_res = SMOTE(random_state=0, sampling_strategy=0.5).fit_resample(X, y)
    # minority target = round(0.5 * 20) = 10
    assert counts_of(y_res) == {0: 20, 1: 10}


def test_float_ratio_one_equals_auto():
    X, y = make_imbalanced()
    X_res, y_res = SMOTE(random_state=0, sampling_strategy=1.0).fit_resample(X, y)
    assert counts_of(y_res) == {0: 20, 1: 20}


def test_dict_strategy_exact_targets():
    X, y = make_imbalanced(n_majority=20, n_minority=6)
    X_res, y_res = SMOTE(random_state=0, sampling_strategy={1: 12}).fit_resample(X, y)
    assert counts_of(y_res) == {0: 20, 1: 12}


def test_dict_strategy_leaves_unlisted_class_alone():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(4, 1, (8, 2)),
                   rng.normal(-4, 1, (5, 2))])
    y = np.array([0] * 30 + [1] * 8 + [2] * 5)
    _, y_res = SMOTE(k_neighbors=3, random_state=0, sampling_strategy={2: 15}).fit_resample(X, y)
    assert counts_of(y_res) == {0: 30, 1: 8, 2: 15}


def test_multi_class_balances_all():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(4, 1, (8, 2)),
                   rng.normal(-4, 1, (5, 2))])
    y = np.array([0] * 30 + [1] * 8 + [2] * 5)
    X_res, y_res = SMOTE(k_neighbors=3, random_state=0).fit_resample(X, y)
    assert counts_of(y_res) == {0: 30, 1: 30, 2: 30}
    assert X_res.shape == (90, 2)


def test_string_labels():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0, 1, (12, 2)), rng.normal(4, 1, (4, 2))])
    y = np.array(["cat"] * 12 + ["dog"] * 4)
    X_res, y_res = SMOTE(k_neighbors=2, random_state=0).fit_resample(X, y)
    assert counts_of(y_res) == {"cat": 12, "dog": 12}
    assert set(np.unique(y_res)) == {"cat", "dog"}


def test_synthetic_points_lie_on_segments():
    X, y = make_imbalanced(n_majority=20, n_minority=8)
    X_res, _ = SMOTE(random_state=0).fit_resample(X, y)
    minority = X[y == 1]
    synthetic = X_res[len(X):]
    assert len(synthetic) == 12
    for point in synthetic:
        assert on_segment(point, minority)


def test_synthetic_points_close_to_real_minority():
    X, y = make_imbalanced(n_majority=20, n_minority=8)
    X_res, _ = SMOTE(random_state=0).fit_resample(X, y)
    minority = X[y == 1]
    synthetic = X_res[len(X):]
    # Max distance between two real minority points bounds every segment.
    max_pairwise = max(np.linalg.norm(a - b) for a in minority for b in minority)
    for point in synthetic:
        nearest = min(np.linalg.norm(point - m) for m in minority)
        assert nearest <= max_pairwise


def test_original_samples_unchanged_and_first():
    X, y = make_imbalanced()
    X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
    np.testing.assert_array_equal(X_res[: len(X)], X)
    np.testing.assert_array_equal(y_res[: len(y)], y)


def test_output_dtype_and_shape():
    X, y = make_imbalanced()
    X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
    assert X_res.dtype == np.float64
    assert X_res.ndim == 2 and y_res.ndim == 1
    assert X_res.shape[1] == X.shape[1]
    assert np.isfinite(X_res).all()


def test_fit_attributes():
    X, y = make_imbalanced(n_majority=20, n_minority=6)
    smote = SMOTE(random_state=0).fit(X, y)
    assert smote.class_counts_ == {0: 20, 1: 6}
    assert smote.n_synthetic_ == 14
    assert smote.classes_ == [0, 1]


def test_already_balanced_returns_copies_unchanged():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (20, 2))
    y = np.array([0] * 10 + [1] * 10)
    X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
    np.testing.assert_array_equal(X_res, X)
    np.testing.assert_array_equal(y_res, y)
    assert X_res is not X and y_res is not y  # copies, not the same objects


def test_reproducible_with_same_seed():
    X, y = make_imbalanced()
    X1, y1 = SMOTE(random_state=42).fit_resample(X, y)
    X2, y2 = SMOTE(random_state=42).fit_resample(X, y)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_differ():
    X, y = make_imbalanced()
    X1, _ = SMOTE(random_state=1).fit_resample(X, y)
    X2, _ = SMOTE(random_state=2).fit_resample(X, y)
    assert not np.array_equal(X1, X2)


def test_module_function_wrapper():
    X, y = make_imbalanced()
    X_res, y_res = fit_resample(X, y, random_state=0)
    assert counts_of(y_res) == {0: 20, 1: 20}
    assert X_res.shape == (40, 2)


# --- validation errors ----------------------------------------------------

def test_empty_input_raises():
    with pytest.raises(ValueError):
        SMOTE().fit_resample(np.empty((0, 2)), np.array([]))


def test_1d_X_raises():
    with pytest.raises(ValueError):
        SMOTE().fit_resample(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))


def test_mismatched_lengths_raises():
    X, y = make_imbalanced()
    with pytest.raises(ValueError):
        SMOTE().fit_resample(X, y[:-1])


def test_nan_raises():
    X, y = make_imbalanced()
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        SMOTE().fit_resample(X, y)


def test_inf_raises():
    X, y = make_imbalanced()
    X = X.copy()
    X[0, 0] = np.inf
    with pytest.raises(ValueError):
        SMOTE().fit_resample(X, y)


def test_k_neighbors_zero_raises():
    with pytest.raises(ValueError):
        SMOTE(k_neighbors=0)


def test_k_neighbors_too_large_raises():
    X, y = make_imbalanced(n_majority=20, n_minority=6)
    with pytest.raises(ValueError):
        SMOTE(k_neighbors=6, random_state=0).fit_resample(X, y)


def test_singleton_minority_raises():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 1, (10, 2)), rng.normal(4, 1, (1, 2))])
    y = np.array([0] * 10 + [1])
    with pytest.raises(ValueError):
        SMOTE(random_state=0).fit_resample(X, y)


def test_dict_target_below_current_raises():
    X, y = make_imbalanced(n_majority=20, n_minority=6)
    with pytest.raises(ValueError):
        SMOTE(sampling_strategy={1: 3}).fit_resample(X, y)


def test_invalid_sampling_strategy_raises():
    X, y = make_imbalanced()
    with pytest.raises(ValueError):
        SMOTE(sampling_strategy="all").fit_resample(X, y)
    with pytest.raises(ValueError):
        SMOTE(sampling_strategy=1.5).fit_resample(X, y)
    with pytest.raises(ValueError):
        SMOTE(sampling_strategy=0).fit_resample(X, y)
    with pytest.raises(ValueError):
        SMOTE(sampling_strategy=[1, 2]).fit_resample(X, y)


def test_resample_before_fit_raises():
    with pytest.raises(RuntimeError):
        SMOTE().resample()
