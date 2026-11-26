"""Gaussian kernel density estimator, from scratch with numpy only.

Fits on samples, evaluates (log-)density at new points, picks a bandwidth
with Silverman's rule of thumb (or a fixed value), and can draw new samples
from the fitted density.
"""
from __future__ import annotations

import numpy as np


class GaussianKDE:
    """Gaussian kernel density estimator.

    Parameters
    ----------
    bandwidth : float or "silverman"
        Kernel bandwidth. ``"silverman"`` (default) uses Silverman's rule
        of thumb on the fitted data; a positive float fixes the bandwidth.
    """

    def __init__(self, bandwidth: float | str = "silverman"):
        if isinstance(bandwidth, str):
            if bandwidth != "silverman":
                raise ValueError("bandwidth must be a positive float or 'silverman'")
        elif isinstance(bandwidth, (int, float)) and not isinstance(bandwidth, bool):
            if float(bandwidth) <= 0:
                raise ValueError("bandwidth must be positive, got %r" % (bandwidth,))
        else:
            raise TypeError("bandwidth must be a positive float or 'silverman'")
        self.bandwidth = bandwidth
        self._samples: np.ndarray | None = None
        self.bandwidth_: float | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray) -> "GaussianKDE":
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if x.shape[0] == 1:
            raise ValueError("need at least 2 samples to estimate a density")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        self._samples = x
        if isinstance(self.bandwidth, str):
            self.bandwidth_ = float(self._silverman_bandwidth(x))
        else:
            self.bandwidth_ = float(self.bandwidth)
        if self.bandwidth_ <= 0:
            raise ValueError("computed bandwidth must be positive")
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        """Log-density of each point in ``x`` under the fitted density."""
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self._samples.shape[1]:
            raise ValueError(
                "x has %d features, expected %d" % (x.shape[1], self._samples.shape[1])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return np.log(self._density(x))

    def score(self, x: np.ndarray) -> np.ndarray:
        """Density of each point in ``x`` under the fitted density."""
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self._samples.shape[1]:
            raise ValueError(
                "x has %d features, expected %d" % (x.shape[1], self._samples.shape[1])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return self._density(x)

    def sample(self, n_samples: int = 1, random_state: int | None = None) -> np.ndarray:
        """Draw ``n_samples`` new points from the fitted density."""
        self._check_fitted()
        if not isinstance(n_samples, int) or isinstance(n_samples, bool):
            raise ValueError("n_samples must be an integer")
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, self._samples.shape[0], size=n_samples)
        chosen = self._samples[idx]
        noise = rng.normal(0.0, self.bandwidth_, size=chosen.shape)
        return chosen + noise

    def mode(self, grid_points: int = 200) -> np.ndarray:
        """Approximate the mode: the grid point with the highest density.

        The grid spans the range of the fitted samples (extended by one
        bandwidth on each side) in every dimension.
        """
        self._check_fitted()
        if not isinstance(grid_points, int) or isinstance(grid_points, bool):
            raise ValueError("grid_points must be an integer")
        if grid_points < 2:
            raise ValueError("grid_points must be at least 2")
        d = self._samples.shape[1]
        axes = []
        for j in range(d):
            lo = self._samples[:, j].min() - self.bandwidth_
            hi = self._samples[:, j].max() + self.bandwidth_
            axes.append(np.linspace(lo, hi, grid_points))
        mesh = np.meshgrid(*axes, indexing="ij")
        grid = np.stack([m.ravel() for m in mesh], axis=1)
        densities = self._density(grid)
        return grid[int(np.argmax(densities))]

    # -- internals ------------------------------------------------------
    def _check_fitted(self):
        if self._samples is None or self.bandwidth_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    @staticmethod
    def _silverman_bandwidth(x: np.ndarray) -> float:
        """Silverman's rule of thumb, per-dimension then averaged."""
        n, d = x.shape
        bandwidths = []
        for j in range(d):
            col = x[:, j]
            std = float(np.std(col, ddof=1))
            iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
            scale = min(std, iqr / 1.34) if iqr > 0 else std
            if scale <= 0:
                scale = std if std > 0 else 1.0
            bandwidths.append(0.9 * scale * n ** (-1.0 / (d + 4)))
        return float(np.mean(bandwidths))

    def _density(self, x: np.ndarray) -> np.ndarray:
        # Gaussian kernel: (2*pi*h^2)^(-d/2) * mean_i exp(-||x - x_i||^2 / 2h^2)
        diff = x[:, None, :] - self._samples[None, :, :]
        sq = np.sum(diff ** 2, axis=2)
        kernel = np.exp(-sq / (2.0 * self.bandwidth_ ** 2))
        d = self._samples.shape[1]
        norm = (2.0 * np.pi * self.bandwidth_ ** 2) ** (-d / 2.0)
        return norm * kernel.mean(axis=1)
