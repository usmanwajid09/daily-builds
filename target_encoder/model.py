"""Smoothed target encoding for categorical features (numpy only).

Target encoding replaces each category with the mean of the target for that
category, shrunk toward the global target mean so rare categories do not get
extreme values from a handful of samples:

    enc(c) = (count_c * mean_c + smoothing * global_mean) / (count_c + smoothing)

Categories with fewer than ``min_samples_leaf`` samples are encoded with the
global mean exactly.
"""

import numpy as np

__all__ = ["TargetEncoder"]


# Sentinel dict key for missing values (None / NaN), so a real category can
# never collide with the missing-value bucket.
_MISSING = object()


def _is_missing(value):
    """Return True for None or NaN values."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


class TargetEncoder:
    """Smoothed target encoder for categorical features.

    Parameters
    ----------
    smoothing : float, default=1.0
        ``m`` in the encoding formula. 0 gives raw per-category means;
        larger values pull every encoding closer to the global mean.
        Must be >= 0.
    min_samples_leaf : int, default=1
        Categories with fewer than this many samples are encoded with the
        global mean exactly, instead of the smoothed formula.
        Must be >= 1.
    handle_unknown : {"value", "error"}, default="value"
        How to encode categories unseen during ``fit``: "value" encodes
        them with the global mean, "error" raises ``ValueError``.
    handle_missing : {"value", "error"}, default="value"
        How to treat missing values (None / NaN): "value" treats them as
        their own category, "error" raises ``ValueError``.

    Attributes
    ----------
    global_mean_ : float
        Mean of ``y`` over the training data.
    categories_ : list of np.ndarray
        One array of seen categories per fitted column (in order of first
        appearance). Missing values appear as the internal ``_MISSING``
        sentinel when ``handle_missing="value"``.
    encodings_ : list of dict
        One ``{category: encoded value}`` mapping per fitted column.
    n_features_in_ : int
        Number of columns ``fit`` was called with.
    """

    def __init__(self, smoothing=1.0, min_samples_leaf=1,
                 handle_unknown="value", handle_missing="value"):
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing!r}")
        if min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be >= 1, got {min_samples_leaf!r}")
        if handle_unknown not in ("value", "error"):
            raise ValueError(
                f"handle_unknown must be 'value' or 'error', "
                f"got {handle_unknown!r}")
        if handle_missing not in ("value", "error"):
            raise ValueError(
                f"handle_missing must be 'value' or 'error', "
                f"got {handle_missing!r}")
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.global_mean_ = None
        self.categories_ = None
        self.encodings_ = None
        self.n_features_in_ = None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _as_2d(X):
        """Return X as a 2D object array plus a flag for whether it was 1D."""
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            return arr.reshape(-1, 1), True
        if arr.ndim == 2:
            return arr, False
        raise ValueError(f"X must be 1D or 2D, got {arr.ndim}D")

    def _check_fitted(self):
        if self.encodings_ is None:
            raise ValueError(
                "TargetEncoder has not been fitted yet; call fit() first.")

    def _key(self, value):
        """Map a raw value to its dict key, honouring handle_missing."""
        if _is_missing(value):
            if self.handle_missing == "error":
                raise ValueError(
                    "Missing value (None/NaN) found but "
                    "handle_missing='error'.")
            return _MISSING
        return value

    # -- API -------------------------------------------------------------
    def fit(self, X, y):
        """Fit the encoder on categorical features ``X`` and target ``y``."""
        X2d, _ = self._as_2d(X)
        try:
            y_arr = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("y must be numeric.") from None
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
        if X2d.shape[0] == 0 or y_arr.shape[0] == 0:
            raise ValueError("X and y must not be empty.")
        if X2d.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: "
                f"{X2d.shape[0]} != {y_arr.shape[0]}")
        if np.isnan(y_arr).any():
            raise ValueError("y must not contain NaN values.")

        self.global_mean_ = float(np.mean(y_arr))
        self.n_features_in_ = X2d.shape[1]
        self.categories_ = []
        self.encodings_ = []

        for col in range(X2d.shape[1]):
            sums = {}
            counts = {}
            order = []
            for value, target in zip(X2d[:, col], y_arr):
                key = self._key(value)
                if key not in counts:
                    counts[key] = 0
                    sums[key] = 0.0
                    order.append(key)
                counts[key] += 1
                sums[key] += float(target)
            encoding = {}
            for key in order:
                count = counts[key]
                if count < self.min_samples_leaf:
                    encoding[key] = self.global_mean_
                else:
                    raw_mean = sums[key] / count
                    encoding[key] = (
                        (count * raw_mean + self.smoothing * self.global_mean_)
                        / (count + self.smoothing)
                    )
            self.categories_.append(np.array(order, dtype=object))
            self.encodings_.append(encoding)
        return self

    def transform(self, X):
        """Encode ``X`` using the fitted per-category values."""
        self._check_fitted()
        X2d, was_1d = self._as_2d(X)
        if X2d.shape[0] == 0:
            raise ValueError("X must not be empty.")
        if X2d.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} column(s), "
                f"got {X2d.shape[1]}")
        out = np.empty((X2d.shape[0], X2d.shape[1]), dtype=float)
        for col in range(X2d.shape[1]):
            encoding = self.encodings_[col]
            for i, value in enumerate(X2d[:, col]):
                key = self._key(value)
                if key in encoding:
                    out[i, col] = encoding[key]
                elif self.handle_unknown == "error":
                    raise ValueError(
                        f"Unseen category {value!r} in column {col} but "
                        f"handle_unknown='error'.")
                else:
                    out[i, col] = self.global_mean_
        return out.ravel() if was_1d else out

    def fit_transform(self, X, y):
        """Fit on (X, y) and return the encoded X."""
        return self.fit(X, y).transform(X)
