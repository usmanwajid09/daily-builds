"""kNN missing-value imputation, implemented from scratch (numpy only).

For each row with missing entries, the k nearest donor rows are found using
Euclidean distance computed over the features observed in both rows (scaled
by n_features / n_common so rows sharing fewer features stay comparable).
Each missing entry is filled with the (optionally inverse-distance-weighted)
mean of the donors that actually observed that feature. Falls back to the
column mean when no donor observed the feature, or when a row has no
observed features at all.
"""

import numpy as np


class KNNImputer:
    """Impute missing values with the k nearest neighbour rows.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of donor rows used to impute each missing entry.
    weights : {"uniform", "distance"}, default "uniform"
        "uniform" averages the donors equally; "distance" weights each
        donor by 1 / distance (a donor at distance 0 dominates).
    missing_values : float, default np.nan
        The placeholder for missing values.
    """

    def __init__(self, n_neighbors=5, weights="uniform", missing_values=np.nan):
        if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
            raise ValueError("n_neighbors must be an integer >= 1")
        if weights not in ("uniform", "distance"):
            raise ValueError('weights must be "uniform" or "distance"')
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.missing_values = missing_values
        self._fitted = False

    # -- helpers ---------------------------------------------------------
    def _missing_mask(self, X):
        if isinstance(self.missing_values, float) and np.isnan(self.missing_values):
            return np.isnan(X)
        return X == self.missing_values

    @staticmethod
    def _as_float_2d(X, name="X"):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got {X.ndim}D")
        if X.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if np.isinf(X).any():
            raise ValueError(f"{name} must not contain inf values")
        return X

    def _distances_to(self, row, row_observed):
        """Distance from `row` to every stored training row.

        Computed over the features observed in both rows, with the squared
        distance scaled by n_features / n_common. Rows sharing no observed
        feature get distance inf (unusable as donors).
        """
        train = self._X_train
        train_obs = ~self._train_missing
        common = row_observed[None, :] & train_obs
        n_common = common.sum(axis=1)
        diff = np.where(common, row[None, :] - train, 0.0)
        sq = np.einsum("ij,ij->i", diff, diff)
        n_features = train.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.sqrt(sq * n_features / np.where(n_common > 0, n_common, 1))
        return np.where(n_common > 0, dist, np.inf)

    def _impute_value(self, donor_idx, donor_dist, feature):
        """Imputed value for one feature from the nearest donors that saw it."""
        observed = ~self._train_missing[donor_idx, feature] & np.isfinite(donor_dist)
        idx = donor_idx[observed]
        dist = donor_dist[observed]
        if idx.size == 0:
            return self._column_means[feature]
        k = min(self.n_neighbors, idx.size)
        idx, dist = idx[:k], dist[:k]
        values = self._X_train[idx, feature]
        if self.weights == "uniform":
            return float(values.mean())
        # Inverse-distance weighting; a zero-distance donor dominates.
        zero = dist == 0.0
        if zero.any():
            return float(values[zero].mean())
        w = 1.0 / dist
        return float(np.dot(w, values) / w.sum())

    # -- public API ------------------------------------------------------
    def fit(self, X):
        """Store the training data and the observed column means."""
        X = self._as_float_2d(X)
        missing = self._missing_mask(X)
        n_features = X.shape[1]
        column_means = np.empty(n_features, dtype=float)
        for j in range(n_features):
            observed = X[~missing[:, j], j]
            if observed.size == 0:
                raise ValueError(f"column {j} is entirely missing")
            column_means[j] = observed.mean()
        self._X_train = X.copy()
        self._train_missing = missing
        self._column_means = column_means
        self.n_features_in_ = n_features
        self._fitted = True
        return self

    def transform(self, X):
        """Return a copy of X with missing values imputed."""
        if not self._fitted:
            raise ValueError("KNNImputer is not fitted yet; call fit first")
        X = self._as_float_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but KNNImputer was fitted with "
                f"{self.n_features_in_} features"
            )
        missing = self._missing_mask(X)
        if not missing.any():
            return X.copy()
        X_out = X.copy()
        # Pre-sort donors once per row; only rows with missing values need it.
        for i in range(X.shape[0]):
            row_missing = np.flatnonzero(missing[i])
            if row_missing.size == 0:
                continue
            observed = ~missing[i]
            if not observed.any():
                # No observed features: fall back to column means.
                X_out[i, row_missing] = self._column_means[row_missing]
                continue
            dist = self._distances_to(X[i], observed)
            donor_order = np.argsort(dist, kind="stable")
            donor_dist = dist[donor_order]
            for f in row_missing:
                X_out[i, f] = self._impute_value(donor_order, donor_dist, f)
        return X_out

    def fit_transform(self, X):
        """Fit on X and return X with missing values imputed."""
        return self.fit(X).transform(X)
