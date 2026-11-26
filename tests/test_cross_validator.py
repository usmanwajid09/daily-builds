import numpy as np
import pytest

from cross_validator.model import (
    KFold,
    StratifiedKFold,
    accuracy_score,
    cross_val_score,
    r2_score,
)


class NearestCentroid:
    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([x[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, x):
        dists = np.linalg.norm(x[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[np.argmin(dists, axis=1)]


class MeanRegressor:
    def fit(self, x, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


def _blobs(seed=42, n=150):
    rng = np.random.default_rng(seed)
    per_class = n // 3
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
    x = np.vstack([rng.normal(loc=c, scale=0.8, size=(per_class, 2)) for c in centres])
    y = np.array([c for c in range(3) for _ in range(per_class)])
    perm = rng.permutation(n)
    return x[perm], y[perm]


def test_kfold_covers_each_sample_once_as_test():
    x = np.arange(10)
    seen = []
    for _, test_idx in KFold(n_splits=5).split(x):
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(10))


def test_kfold_train_test_are_disjoint_and_cover_all():
    x = np.arange(12)
    for train_idx, test_idx in KFold(n_splits=4).split(x):
        assert set(train_idx).isdisjoint(set(test_idx))
        assert sorted(np.concatenate([train_idx, test_idx]).tolist()) == list(range(12))


def test_kfold_fold_sizes_differ_by_at_most_one():
    x = np.arange(11)
    sizes = [len(t) for _, t in KFold(n_splits=3).split(x)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 11


def test_kfold_shuffle_changes_order_but_covers_all():
    x = np.arange(20)
    folds = list(KFold(n_splits=4, shuffle=True, random_state=0).split(x))
    first_test = folds[0][1]
    assert not np.array_equal(first_test, np.arange(5))
    seen = np.concatenate([t for _, t in folds])
    assert sorted(seen.tolist()) == list(range(20))


def test_kfold_reproducible_with_seed():
    x = np.arange(20)
    a = [t.tolist() for _, t in KFold(n_splits=4, shuffle=True, random_state=5).split(x)]
    b = [t.tolist() for _, t in KFold(n_splits=4, shuffle=True, random_state=5).split(x)]
    assert a == b


def test_kfold_too_many_splits_raises():
    with pytest.raises(ValueError, match="cannot split"):
        list(KFold(n_splits=5).split(np.arange(3)))


def test_kfold_n_splits_validation():
    with pytest.raises(ValueError, match="at least 2"):
        KFold(n_splits=1)


def test_stratified_keeps_class_proportions():
    x, y = _blobs()
    for train_idx, test_idx in StratifiedKFold(n_splits=5).split(x, y):
        counts = np.bincount(y[test_idx], minlength=3)
        assert counts.tolist() == [10, 10, 10]


def test_stratified_covers_each_sample_once():
    x, y = _blobs()
    seen = []
    for _, test_idx in StratifiedKFold(n_splits=5).split(x, y):
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(len(y)))


def test_stratified_rare_class_raises():
    y = np.array([0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="least populated class"):
        list(StratifiedKFold(n_splits=3).split(np.arange(5), y))


def test_cross_val_score_returns_one_score_per_fold():
    x, y = _blobs()
    scores = cross_val_score(NearestCentroid(), x, y, cv=5)
    assert scores.shape == (5,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores.mean() > 0.95


def test_cross_val_score_with_splitter_object():
    x, y = _blobs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(NearestCentroid(), x, y, cv=cv, scoring="accuracy")
    assert scores.shape == (5,)
    assert scores.mean() > 0.95


def test_cross_val_score_regression_r2():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(100, 2))
    y = 3 * x[:, 0] - 2 * x[:, 1] + rng.normal(0, 0.1, size=100)
    scores = cross_val_score(MeanRegressor(), x, y, cv=5, scoring="r2")
    assert scores.shape == (5,)
    assert np.all(scores <= 0.0 + 1e-9)  # predicting the mean gives R² ≈ 0 or below


def test_cross_val_score_default_scorer_is_r2_for_float_y():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(50, 1))
    y = x[:, 0] + rng.normal(0, 0.1, size=50)
    scores = cross_val_score(MeanRegressor(), x, y, cv=5)
    assert np.all(scores <= 0.0 + 1e-9)


def test_cross_val_score_custom_callable_scorer():
    x, y = _blobs()
    scores = cross_val_score(
        NearestCentroid(), x, y, cv=3,
        scoring=lambda yt, yp: float(np.mean(yt == yp)),
    )
    assert scores.shape == (3,)
    assert scores.mean() > 0.95


def test_cross_val_score_with_explicit_fold_list():
    x, y = _blobs()
    folds = list(KFold(n_splits=2).split(x))
    scores = cross_val_score(NearestCentroid(), x, y, cv=folds)
    assert scores.shape == (2,)


def test_accuracy_score_known_value():
    assert accuracy_score([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_r2_score_perfect_is_one():
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_r2_score_zero_variance_raises():
    with pytest.raises(ValueError, match="zero variance"):
        r2_score([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])


def test_cross_val_bad_scoring_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="scoring"):
        cross_val_score(NearestCentroid(), x, y, cv=2, scoring="f1")


def test_cross_val_mismatched_xy_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="same number of samples"):
        cross_val_score(NearestCentroid(), x, y[:10], cv=2)


def test_demo_runs():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "cross_validator/demo.py"],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0
    assert "Mean accuracy" in result.stdout
