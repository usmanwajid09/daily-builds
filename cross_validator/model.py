"""K-fold cross-validation utilities, from scratch with numpy only.

- ``KFold`` — plain k-fold splitter, shuffled or sequential.
- ``StratifiedKFold`` — each fold keeps the class proportions of ``y``.
- ``cross_val_score`` — fits a fresh copy of an estimator on each fold
  and returns one score per fold (accuracy for classifiers, R² for
  regressors, or a custom scorer).
"""
from __future__ import annotations

import copy

import numpy as np


class KFold:
    """K-fold splitter.

    Parameters
    ----------
    n_splits : int
        Number of folds (at least 2).
    shuffle : bool
        Shuffle the samples before splitting.
    random_state : int or None
        Seed used when ``shuffle`` is True.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: int | None = None):
        if not isinstance(n_splits, int) or isinstance(n_splits, bool):
            raise ValueError("n_splits must be an integer")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, x, y=None):
        """Yield ``(train_idx, test_idx)`` pairs, one per fold."""
        n = len(np.asarray(x))
        if n < self.n_splits:
            raise ValueError(
                "cannot split %d samples into %d folds" % (n, self.n_splits)
            )
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        start = 0
        for size in fold_sizes:
            stop = start + size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            start = stop

    def get_n_splits(self) -> int:
        return self.n_splits


class StratifiedKFold:
    """Stratified k-fold splitter: every fold keeps the class proportions.

    Parameters
    ----------
    n_splits : int
        Number of folds (at least 2).
    shuffle : bool
        Shuffle within each class before splitting.
    random_state : int or None
        Seed used when ``shuffle`` is True.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: int | None = None):
        if not isinstance(n_splits, int) or isinstance(n_splits, bool):
            raise ValueError("n_splits must be an integer")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, x, y):
        """Yield ``(train_idx, test_idx)`` pairs, one per fold."""
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        n = len(y)
        if n < self.n_splits:
            raise ValueError(
                "cannot split %d samples into %d folds" % (n, self.n_splits)
            )
        classes, counts = np.unique(y, return_counts=True)
        if counts.min() < self.n_splits:
            raise ValueError(
                "the least populated class has %d samples, fewer than n_splits=%d"
                % (counts.min(), self.n_splits)
            )
        rng = np.random.default_rng(self.random_state)
        # Per-class fold assignments.
        fold_of_sample = np.empty(n, dtype=int)
        for cls in classes:
            idx = np.where(y == cls)[0]
            if self.shuffle:
                idx = rng.permutation(idx)
            fold_of_sample[idx] = np.arange(len(idx)) % self.n_splits
        all_idx = np.arange(n)
        for fold in range(self.n_splits):
            test_idx = all_idx[fold_of_sample == fold]
            train_idx = all_idx[fold_of_sample != fold]
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(y_true == y_pred))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        raise ValueError("r2 is undefined when y_true has zero variance")
    return 1.0 - ss_res / ss_tot


def cross_val_score(estimator, x, y, cv=5, scoring: str | None = None) -> np.ndarray:
    """Cross-validated scores for ``estimator`` on ``(x, y)``.

    ``cv`` is an int (plain ``KFold``), a ``KFold``/``StratifiedKFold``
    instance, or an iterable of ``(train_idx, test_idx)`` pairs. The
    estimator is deep-copied for every fold, so one fitted estimator is
    never reused. ``scoring`` is ``"accuracy"``, ``"r2"``, or a callable
    ``(y_true, y_pred) -> float``; when ``None`` it defaults to accuracy
    for integer/bool targets and R² otherwise.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    if x.shape[0] == 0:
        raise ValueError("cannot cross-validate on empty data")

    if isinstance(cv, int):
        if cv < 2:
            raise ValueError("cv must be at least 2")
        splitter = KFold(n_splits=cv)
        folds = list(splitter.split(x))
    elif hasattr(cv, "split"):
        folds = list(cv.split(x, y))
    else:
        folds = list(cv)

    if scoring is None:
        scorer = accuracy_score if y.dtype.kind in "iu b" else r2_score
    elif scoring == "accuracy":
        scorer = accuracy_score
    elif scoring == "r2":
        scorer = r2_score
    elif callable(scoring):
        scorer = scoring
    else:
        raise ValueError("scoring must be 'accuracy', 'r2', a callable, or None")

    scores = []
    for train_idx, test_idx in folds:
        est = copy.deepcopy(estimator)
        est.fit(x[train_idx], y[train_idx])
        preds = est.predict(x[test_idx])
        scores.append(float(scorer(y[test_idx], preds)))
    return np.array(scores)
