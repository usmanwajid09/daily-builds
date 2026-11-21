"""SMOTE synthetic minority oversampling, from scratch with numpy only.

SMOTE (Synthetic Minority Over-sampling Technique, Chawla et al. 2002)
fixes class imbalance by creating new, synthetic minority samples instead
of just duplicating existing ones. For each minority sample it picks one
of its ``k_neighbors`` nearest neighbours *of the same class* at random
and places a new point somewhere on the line segment between the two:

    synthetic = x_i + gap * (x_neighbour - x_i),  gap ~ Uniform[0, 1]

Because every synthetic point is a convex combination of two real
minority points, it stays inside the minority region instead of
inventing points in empty space. Every class below the majority count
is oversampled (multi-class is supported), and the random number
generator is seeded, so results are reproducible.
"""
from __future__ import annotations

import numpy as np


def _as_python_scalar(label):
    """Convert a numpy scalar label to a plain Python scalar for clean dicts."""
    if isinstance(label, np.generic):
        return label.item()
    return label


class SMOTE:
    """Synthetic Minority Over-sampling Technique.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest same-class neighbours to choose from when
        generating a synthetic point. Must be >= 1 and smaller than the
        number of samples in the smallest class being oversampled.
    random_state : int or None
        Seed for the random number generator (neighbour choice and
        interpolation factor). Same seed -> identical output.
    sampling_strategy : "auto", float, or dict
        Which classes to oversample and to what size:

        - ``"auto"`` (default): every class is raised to the majority
          class count.
        - float ``ratio`` in (0, 1]: every minority class is raised to
          ``round(ratio * majority_count)`` samples.
        - dict ``{class_label: target_count}``: only the listed classes
          are oversampled, each to its given count (which must be at
          least the class's current count).
    """

    def __init__(self, k_neighbors: int = 5, random_state: int | None = None,
                 sampling_strategy="auto"):
        if isinstance(k_neighbors, bool) or not isinstance(k_neighbors, int):
            raise ValueError("k_neighbors must be an integer")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise ValueError("random_state must be an integer or None")
        if isinstance(sampling_strategy, str):
            if sampling_strategy != "auto":
                raise ValueError('sampling_strategy must be "auto", a float, or a dict')
        elif isinstance(sampling_strategy, bool) or not isinstance(
            sampling_strategy, (int, float, dict)
        ):
            raise ValueError('sampling_strategy must be "auto", a float, or a dict')
        elif isinstance(sampling_strategy, (int, float)):
            if not 0 < float(sampling_strategy) <= 1:
                raise ValueError("sampling_strategy as a float must be in (0, 1]")
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.classes_ = None
        self.class_counts_ = None
        self.n_synthetic_ = 0
        self.n_features_in_ = None
        self._X = None
        self._y = None
        self._targets = None
        self._fitted = False

    # -- validation -----------------------------------------------------

    @staticmethod
    def _validate_inputs(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if X.shape[0] == 0 or y.size == 0:
            raise ValueError("X and y must not be empty")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y have mismatched lengths: %d samples vs %d labels"
                % (X.shape[0], y.shape[0])
            )
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinite values")
        return X, y

    def _compute_targets(self, classes, counts):
        """Target sample count per class, from sampling_strategy."""
        strategy = self.sampling_strategy
        majority_count = int(max(counts.values()))
        if isinstance(strategy, str):  # "auto"
            return {c: majority_count for c in classes}
        if isinstance(strategy, dict):
            targets = dict(counts)  # classes not listed stay as they are
            for label, target in strategy.items():
                if label not in counts:
                    raise ValueError(
                        "sampling_strategy dict contains unknown class label: %r"
                        % (label,)
                    )
                if isinstance(target, bool) or not isinstance(target, (int, np.integer)):
                    raise ValueError(
                        "sampling_strategy dict values must be integers, got %r"
                        % (target,)
                    )
                if int(target) < counts[label]:
                    raise ValueError(
                        "sampling_strategy target for class %r (%d) is below its "
                        "current count (%d)" % (label, int(target), counts[label])
                    )
                targets[label] = int(target)
            return targets
        # float ratio
        ratio = float(strategy)
        target = int(round(ratio * majority_count))
        return {c: max(counts[c], target) for c in classes}

    # -- fitting --------------------------------------------------------

    def fit(self, X, y) -> "SMOTE":
        """Learn class counts and per-class target counts from X, y."""
        X, y = self._validate_inputs(X, y)
        classes, class_counts = np.unique(y, return_counts=True)
        classes = [_as_python_scalar(c) for c in classes]
        counts = {c: int(n) for c, n in zip(classes, class_counts)}
        targets = self._compute_targets(classes, counts)
        # Classes that actually need new samples.
        oversampled = {c: counts[c] for c in classes if targets[c] > counts[c]}
        for label, n in oversampled.items():
            if n < 2:
                raise ValueError(
                    "cannot SMOTE class %r: it has only %d sample(s); a class "
                    "needs at least 2 samples" % (label, n)
                )
            if self.k_neighbors >= n:
                raise ValueError(
                    "k_neighbors=%d must be smaller than the number of samples "
                    "(%d) in the smallest class being oversampled (class %r)"
                    % (self.k_neighbors, n, label)
                )
        self.classes_ = list(classes)
        self.class_counts_ = dict(counts)
        self._targets = targets
        self.n_synthetic_ = int(sum(targets[c] - counts[c] for c in classes))
        self.n_features_in_ = X.shape[1]
        self._X = np.array(X, dtype=float, copy=True)
        self._y = np.array(y, copy=True)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("SMOTE is not fitted yet - call fit() first")

    # -- generation -----------------------------------------------------

    def _generate_for_class(self, X_class: np.ndarray, n_needed: int,
                            rng: np.random.Generator) -> np.ndarray:
        """Generate n_needed synthetic points from one class's samples."""
        n = X_class.shape[0]
        # Pairwise squared Euclidean distances within the class.
        diff = X_class[:, None, :] - X_class[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)  # a sample is never its own neighbour
        # Indices of the k nearest same-class neighbours of each sample.
        neighbours = np.argsort(dists, axis=1)[:, : self.k_neighbors]
        synthetic = np.empty((n_needed, X_class.shape[1]), dtype=float)
        for i in range(n_needed):
            src = int(rng.integers(n))
            nn = int(neighbours[src, rng.integers(self.k_neighbors)])
            gap = float(rng.random())  # interpolation factor in [0, 1]
            synthetic[i] = X_class[src] + gap * (X_class[nn] - X_class[src])
        return synthetic

    def resample(self):
        """Generate the resampled (X, y) from the data passed to fit().

        Requires fit() to have been called first.
        """
        self._check_fitted()
        if self.n_synthetic_ == 0:
            return np.array(self._X, copy=True), np.array(self._y, copy=True)
        rng = np.random.default_rng(self.random_state)
        synth_X_parts = []
        synth_y_parts = []
        for label in self.classes_:
            n_needed = self._targets[label] - self.class_counts_[label]
            if n_needed <= 0:
                continue
            mask = self._y == label
            synth_X_parts.append(
                self._generate_for_class(self._X[mask], n_needed, rng)
            )
            synth_y_parts.append(
                np.full(n_needed, label, dtype=self._y.dtype)
            )
        X_res = np.vstack([self._X] + synth_X_parts)
        y_res = np.concatenate([self._y] + synth_y_parts)
        return X_res, y_res

    def fit_resample(self, X, y):
        """Fit on (X, y) and return the resampled (X_res, y_res)."""
        self.fit(X, y)
        return self.resample()


def fit_resample(X, y, k_neighbors: int = 5, random_state: int | None = None,
                 sampling_strategy="auto"):
    """Convenience wrapper: ``SMOTE(...).fit_resample(X, y)``.

    Returns ``(X_res, y_res)`` with minority classes oversampled with
    synthetic points until they reach their target counts.
    """
    return SMOTE(
        k_neighbors=k_neighbors,
        random_state=random_state,
        sampling_strategy=sampling_strategy,
    ).fit_resample(X, y)
