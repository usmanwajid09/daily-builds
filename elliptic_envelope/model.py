"""Robust-covariance (MCD-style) anomaly detection, numpy only."""

import math
import statistics

import numpy as np


def _chi2_quantile(p, df):
    """Approximate chi-square quantile via the Wilson-Hilferty transform.

    Uses the standard-normal quantile from the stdlib
    (``statistics.NormalDist``) — no scipy needed.
    """
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    z = statistics.NormalDist().inv_cdf(p)
    a = 2.0 / (9.0 * df)
    return df * (1.0 - a + z * math.sqrt(a)) ** 3


def _gammainc_reg(a, x):
    """Regularised lower incomplete gamma P(a, x) (Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        # Series expansion.
        term = 1.0 / a
        total = term
        n = 1
        while True:
            term *= x / (a + n)
            total += term
            n += 1
            if abs(term) < abs(total) * 1e-14 or n > 1000:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for Q(a, x); P = 1 - Q.
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def _chi2_cdf(x, df):
    """Chi-square CDF via the regularised incomplete gamma function."""
    return _gammainc_reg(df / 2.0, x / 2.0)


def _consistency_factor(alpha, p):
    """MCD consistency factor: the covariance of an alpha-fraction
    truncated subset is ``factor`` times the true covariance, so the
    corrected covariance is ``raw_cov / factor``."""
    q = _chi2_quantile(alpha, p)
    return _chi2_cdf(q, p + 2) / alpha


class EllipticEnvelope:
    """Anomaly detection with a robust (MCD-style) covariance estimate.

    Fits an ellipse to the bulk of the data using the Minimum Covariance
    Determinant idea: many random subsets are drawn, each is refined by
    "concentration steps" (keep the ``h`` points closest to the current
    centre, refit), and the subset with the smallest covariance
    determinant wins. Because outliers cannot drag the centre or the
    covariance, the Mahalanobis distance to the robust centre is a
    reliable outlier score. Points whose squared distance exceeds the
    chi-square threshold for ``1 - contamination`` are flagged -1.

    Parameters
    ----------
    contamination : float, default 0.1
        Expected fraction of outliers, in (0, 0.5). Sets the decision
        threshold; it does not force exactly that many points to be
        flagged.
    n_trials : int, default 200
        Number of random starting subsets for the MCD search.
    n_concentration_steps : int, default 10
        Concentration steps per trial.
    random_state : int or None, default None
        Seed for the random subsets.
    """

    def __init__(self, contamination=0.1, n_trials=200,
                 n_concentration_steps=10, random_state=None):
        if not 0 < contamination < 0.5:
            raise ValueError(
                f"contamination must be in (0, 0.5), got {contamination!r}")
        if n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got {n_trials!r}")
        if n_concentration_steps <= 0:
            raise ValueError(
                f"n_concentration_steps must be positive, "
                f"got {n_concentration_steps!r}")
        self.contamination = contamination
        self.n_trials = n_trials
        self.n_concentration_steps = n_concentration_steps
        self.random_state = random_state

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values")
        return X

    @staticmethod
    def _mean_cov(X):
        mean = X.mean(axis=0)
        centred = X - mean
        cov = (centred.T @ centred) / max(len(X) - 1, 1)
        # Ridge so a degenerate subset never yields a singular matrix.
        cov = cov + np.eye(X.shape[1]) * 1e-8
        return mean, cov

    @staticmethod
    def _mahalanobis_sq(X, mean, cov):
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov)
        diff = X - mean
        return np.einsum("ij,jk,ik->i", diff, precision, diff)

    def _mcd(self, X, rng):
        n, p = X.shape
        h = (n + p + 1) // 2
        best_det, best_mean, best_cov = np.inf, None, None
        start_size = min(p + 1, n)
        for _ in range(self.n_trials):
            idx = rng.choice(n, size=start_size, replace=False)
            subset = X[idx]
            mean, cov = self._mean_cov(subset)
            for _ in range(self.n_concentration_steps):
                d2 = self._mahalanobis_sq(X, mean, cov)
                keep = np.argsort(d2)[:h]
                new_mean, new_cov = self._mean_cov(X[keep])
                if np.allclose(new_mean, mean) and np.allclose(new_cov, cov):
                    mean, cov = new_mean, new_cov
                    break
                mean, cov = new_mean, new_cov
            det = np.linalg.det(cov)
            if det < best_det:
                best_det, best_mean, best_cov = det, mean, cov
        return best_mean, best_cov, h

    # -- API -------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit the robust location and covariance to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        X = self._check_X(X)
        n, p = X.shape
        if n <= p:
            raise ValueError(
                f"Need more samples ({n}) than features ({p}) to estimate "
                "a covariance")
        rng = np.random.default_rng(self.random_state)
        self.location_, self.covariance_, h = self._mcd(X, rng)
        # Consistency correction: the covariance of the h-subset
        # underestimates the true covariance, so rescale it.
        self.covariance_ = self.covariance_ / _consistency_factor(h / n, p)
        try:
            self.precision_ = np.linalg.inv(self.covariance_)
        except np.linalg.LinAlgError:
            self.precision_ = np.linalg.pinv(self.covariance_)
        # Reweighting: refit on the points inside the chi-square cut-off,
        # with the same consistency correction for the kept fraction.
        d2 = self._mahalanobis_sq(X, self.location_, self.covariance_)
        threshold = _chi2_quantile(1.0 - self.contamination, p)
        inliers = d2 <= threshold
        if inliers.sum() > p:
            self.location_, self.covariance_ = self._mean_cov(X[inliers])
            kept = float(inliers.mean())
            self.covariance_ = self.covariance_ / _consistency_factor(kept, p)
            try:
                self.precision_ = np.linalg.inv(self.covariance_)
            except np.linalg.LinAlgError:
                self.precision_ = np.linalg.pinv(self.covariance_)
        self.threshold_ = float(
            _chi2_quantile(1.0 - self.contamination, p))
        self.support_ = self._mahalanobis_sq(
            X, self.location_, self.covariance_) <= self.threshold_
        return self

    def _dist_sq(self, X):
        if not hasattr(self, "location_"):
            raise RuntimeError(
                "EllipticEnvelope is not fitted yet; call fit() first")
        X = self._check_X(X)
        if X.shape[1] != self.location_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.location_.shape[0]}")
        return self._mahalanobis_sq(X, self.location_, self.covariance_)

    def decision_function(self, X):
        """Signed distance to the threshold; positive means inlier."""
        d2 = self._dist_sq(X)
        return self.threshold_ - d2

    def score_samples(self, X):
        """Outlier score: negative squared Mahalanobis distance.

        Higher means more normal (closer to the robust centre).
        """
        return -self._dist_sq(X)

    def predict(self, X):
        """Return 1 for inliers and -1 for outliers."""
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def mahalanobis(self, X):
        """Mahalanobis distance of each sample to the robust centre."""
        return np.sqrt(np.clip(self._dist_sq(X), 0, None))
