"""Matrix factorisation recommender via SGD (numpy only).

Learns low-rank user and item factors (plus global, user and item bias
terms) from a sparse ratings matrix by stochastic gradient descent on
the squared error of the observed entries. Missing entries are encoded
as 0 (or NaN) and are ignored during training.
"""

import numpy as np


class MatrixFactorization:
    """SGD matrix factorisation for collaborative filtering.

    Parameters
    ----------
    n_factors : int
        Rank of the factorisation (size of each user/item vector).
    learning_rate : float
        SGD step size.
    reg : float
        L2 regularisation strength on factors and biases.
    n_epochs : int
        Number of passes over the observed ratings.
    random_state : int or None
        Seed for the factor initialisation and SGD shuffling.
    """

    def __init__(self, n_factors=10, learning_rate=0.01, reg=0.02,
                 n_epochs=100, random_state=None):
        if int(n_factors) < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {learning_rate}")
        if reg < 0:
            raise ValueError(f"reg must be >= 0, got {reg}")
        if int(n_epochs) < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
        self.n_factors = int(n_factors)
        self.learning_rate = float(learning_rate)
        self.reg = float(reg)
        self.n_epochs = int(n_epochs)
        self.random_state = random_state
        self._fitted = False

    def fit(self, ratings):
        """Fit factors to ``ratings`` (2D array, 0 or NaN = missing)."""
        ratings = np.asarray(ratings, dtype=np.float64)
        if ratings.ndim != 2:
            raise ValueError(
                f"ratings must be 2D, got shape {ratings.shape}")
        if ratings.shape[0] < 1 or ratings.shape[1] < 1:
            raise ValueError(
                f"ratings must be non-empty, got shape {ratings.shape}")
        observed = ~np.isnan(ratings) & (ratings != 0)
        if not np.isfinite(ratings[observed]).all():
            raise ValueError("ratings must contain only finite values")
        if observed.sum() == 0:
            raise ValueError(
                "ratings has no observed entries (all 0 or NaN)")
        self.n_users, self.n_items = ratings.shape
        self.global_mean_ = float(ratings[observed].mean())
        self._observed_mask = observed
        rng = np.random.default_rng(self.random_state)
        scale = 0.1
        self.user_factors_ = rng.normal(0.0, scale,
                                        (self.n_users, self.n_factors))
        self.item_factors_ = rng.normal(0.0, scale,
                                        (self.n_items, self.n_factors))
        self.user_bias_ = np.zeros(self.n_users)
        self.item_bias_ = np.zeros(self.n_items)
        users, items = np.nonzero(observed)
        values = ratings[observed]
        order = np.arange(len(values))
        for _ in range(self.n_epochs):
            rng.shuffle(order)
            for idx in order:
                u, i, r = users[idx], items[idx], values[idx]
                pred = self._raw_predict(u, i)
                err = r - pred
                lr, reg = self.learning_rate, self.reg
                uf = self.user_factors_[u].copy()
                itf = self.item_factors_[i].copy()
                self.user_bias_[u] += lr * (err - reg * self.user_bias_[u])
                self.item_bias_[i] += lr * (err - reg * self.item_bias_[i])
                self.user_factors_[u] += lr * (err * itf - reg * uf)
                self.item_factors_[i] += lr * (err * uf - reg * itf)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise ValueError("call fit() before predicting")

    def _check_user_item(self, user, item):
        if not isinstance(user, (int, np.integer)):
            raise ValueError(f"user index must be an integer, got {user!r}")
        if not isinstance(item, (int, np.integer)):
            raise ValueError(f"item index must be an integer, got {item!r}")
        if not 0 <= user < self.n_users:
            raise ValueError(
                f"user index {user} out of range [0, {self.n_users})")
        if not 0 <= item < self.n_items:
            raise ValueError(
                f"item index {item} out of range [0, {self.n_items})")

    def _raw_predict(self, user, item):
        return (self.global_mean_ + self.user_bias_[user]
                + self.item_bias_[item]
                + float(self.user_factors_[user]
                        @ self.item_factors_[item]))

    def predict(self, user, item):
        """Predicted rating for one (user, item) pair."""
        self._check_fitted()
        self._check_user_item(user, item)
        return float(self._raw_predict(user, item))

    def predict_all(self):
        """Full (n_users, n_items) matrix of predicted ratings."""
        self._check_fitted()
        return (self.global_mean_ + self.user_bias_[:, None]
                + self.item_bias_[None, :]
                + self.user_factors_ @ self.item_factors_.T)

    def rmse(self, ratings):
        """RMSE over the observed entries of ``ratings``."""
        self._check_fitted()
        ratings = np.asarray(ratings, dtype=np.float64)
        if ratings.ndim != 2 or ratings.shape != (self.n_users,
                                                   self.n_items):
            raise ValueError(
                f"ratings must have shape {(self.n_users, self.n_items)}, "
                f"got {ratings.shape}")
        observed = ~np.isnan(ratings) & (ratings != 0)
        if observed.sum() == 0:
            raise ValueError("ratings has no observed entries")
        errors = ratings[observed] - self.predict_all()[observed]
        return float(np.sqrt(np.mean(errors ** 2)))

    def recommend(self, user, n=5, exclude_rated=True):
        """Top-``n`` item indices for ``user``, best predicted first.

        With ``exclude_rated=True`` (default) items the user already
        rated in the training matrix are skipped.
        """
        self._check_fitted()
        if not isinstance(user, (int, np.integer)):
            raise ValueError(f"user index must be an integer, got {user!r}")
        if not 0 <= user < self.n_users:
            raise ValueError(
                f"user index {user} out of range [0, {self.n_users})")
        if int(n) < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if int(n) > self.n_items:
            raise ValueError(
                f"n={n} exceeds the number of items ({self.n_items})")
        scores = self.predict_all()[user].copy()
        if exclude_rated:
            scores[self._observed_mask[user]] = -np.inf
            if np.isinf(scores).all():
                raise ValueError(
                    f"user {user} has rated every item: nothing to recommend")
        ranked = np.argsort(-scores)
        return [int(i) for i in ranked[: int(n)]]
