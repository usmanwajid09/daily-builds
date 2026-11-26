"""Feature selection utilities, from scratch with numpy only.

Three scorers, one for each common situation:

- ``variance_threshold`` — drop near-constant features (unsupervised).
- ``mutual_info_scores`` — mutual information between each feature and a
  discrete target, estimated by binning (classification).
- ``correlation_scores`` — absolute Pearson correlation of each feature
  with a continuous target (regression).

``select_top_k`` ranks features by any of these scores and returns the
indices of the ``k`` best.
"""
from __future__ import annotations

import numpy as np


def _check_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
    if x.shape[0] == 0:
        raise ValueError("cannot select features from empty data")
    if not np.isfinite(x).all():
        raise ValueError("x must not contain NaN or infinite values")
    return x


def _check_y(y: np.ndarray, n_samples: int) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array of labels or targets")
    if y.shape[0] != n_samples:
        raise ValueError("x and y must have the same number of samples")
    return y


def variance_threshold(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Boolean mask of features whose variance exceeds ``threshold``.

    Features with (near-)zero variance carry no information and are
    dropped. Returns a mask of length ``n_features``.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a number")
    if float(threshold) < 0:
        raise ValueError("threshold must be non-negative, got %r" % (threshold,))
    x = _check_x(x)
    variances = np.var(x, axis=0)
    return variances > float(threshold)


def mutual_info_scores(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Mutual information (in nats) of each feature with discrete ``y``.

    Each continuous feature is binned into ``n_bins`` equal-width bins;
    MI is computed from the joint histogram of binned feature vs class.
    """
    if not isinstance(n_bins, int) or isinstance(n_bins, bool):
        raise ValueError("n_bins must be an integer")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    x = _check_x(x)
    y = _check_y(y, x.shape[0])
    classes, y_idx = np.unique(y, return_inverse=True)
    n = x.shape[0]
    scores = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.ptp(col) == 0:
            scores[j] = 0.0
            continue
        edges = np.linspace(col.min(), col.max(), n_bins + 1)
        edges[0] -= 1e-12
        edges[-1] += 1e-12
        binned = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1)
        joint = np.zeros((n_bins, len(classes)))
        for b, c in zip(binned, y_idx):
            joint[b, c] += 1
        joint /= n
        p_x = joint.sum(axis=1, keepdims=True)
        p_y = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(joint > 0, joint / (p_x * p_y), 1.0)
            mi = np.sum(np.where(joint > 0, joint * np.log(ratio), 0.0))
        scores[j] = float(mi)
    return scores


def correlation_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation of each feature with continuous ``y``.

    Constant features score 0. Returns values in [0, 1].
    """
    x = _check_x(x)
    y = _check_y(y, x.shape[0])
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("y must not contain NaN or infinite values")
    if np.std(y) == 0:
        raise ValueError("y has zero variance - correlation is undefined")
    y_centred = y - y.mean()
    scores = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.std(col) == 0:
            scores[j] = 0.0
            continue
        scores[j] = abs(float(np.corrcoef(col, y)[0, 1]))
    return scores


def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` highest-scoring features, in ranked order."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array")
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("k must be an integer")
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > len(scores):
        raise ValueError("k=%d exceeds the number of features (%d)" % (k, len(scores)))
    order = np.argsort(-scores, kind="stable")
    return order[:k]
