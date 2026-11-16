"""Multinomial logistic regression (softmax regression) from scratch.

Softmax regression generalises logistic regression to more than two
classes: a linear score is computed for every class, the softmax turns
those scores into a probability distribution, and the model is trained
by minimising cross-entropy loss with batch gradient descent.
Features are standardised inside ``fit`` so that gradient descent
converges on a sensible timescale regardless of the input scale.

Only numpy is used — no scikit-learn.
"""

import numpy as np


def _softmax(z):
    """Numerically stable softmax (subtracts the row max first)."""
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


class SoftmaxRegression:
    """Multinomial logistic regression trained by batch gradient descent.

    Parameters
    ----------
    learning_rate : float, default 0.1
        Step size for gradient descent.
    n_epochs : int, default 1000
        Maximum number of full-batch gradient steps.
    l2 : float, default 0.0
        L2 penalty strength on the weights only (the bias is not
        penalised).
    tol : float, default 1e-6
        Training stops early when the absolute change in loss between
        two consecutive epochs falls below this value.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels seen in ``fit``, in sorted order.
    coef_ : ndarray of shape (n_features, n_classes)
        Learned weights, one column per class.
    intercept_ : ndarray of shape (n_classes,)
        Learned bias, one value per class.
    loss_history_ : list of float
        Cross-entropy loss after each epoch.
    n_epochs_run_ : int
        Number of epochs actually run (less than ``n_epochs`` when
        early stopping triggers).
    """

    def __init__(self, learning_rate=0.1, n_epochs=1000, l2=0.0, tol=1e-6):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive, "
                             f"got {learning_rate}.")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be at least 1, got {n_epochs}.")
        if l2 < 0:
            raise ValueError(f"l2 must be non-negative, got {l2}.")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}.")
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2 = l2
        self.tol = tol

    # -- validation -----------------------------------------------------
    @staticmethod
    def _validate_xy(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must be non-empty.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D.")
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got {y.ndim}D.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y have mismatched lengths: {len(X)} vs {len(y)}.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values (NaN or inf).")
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError(
                f"Need at least 2 classes to train, got {len(classes)}.")
        return X, y, classes

    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError(
                "SoftmaxRegression is not fitted yet. Call fit(X, y) first.")

    def _validate_predict_X(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D.")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fitted "
                f"with {self.coef_.shape[0]}.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values (NaN or inf).")
        return X

    # -- core -----------------------------------------------------------
    def fit(self, X, y):
        """Fit the model to training data ``X`` (n_samples, n_features)
        and integer-ish labels ``y`` (n_samples,). Returns self."""
        X, y, classes = self._validate_xy(X, y)
        n_samples, n_features = X.shape
        n_classes = len(classes)
        self.classes_ = classes

        # Standardise features; a constant feature gets scale 1.
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Xs = (X - self.mean_) / self.scale_

        # One-hot encode the labels.
        class_index = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_index[v] for v in y])
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y_idx] = 1.0

        rng = np.random.default_rng(0)
        W = rng.normal(scale=0.01, size=(n_features, n_classes))
        b = np.zeros(n_classes)

        self.loss_history_ = []
        self.n_epochs_run_ = 0
        prev_loss = None
        for epoch in range(self.n_epochs):
            logits = Xs @ W + b
            probs = _softmax(logits)
            # Cross-entropy, clipped so log(0) never appears.
            loss = -np.mean(np.log(np.clip(
                probs[np.arange(n_samples), y_idx], 1e-15, 1.0)))
            if self.l2 > 0:
                loss += 0.5 * self.l2 * np.sum(W ** 2)
            self.loss_history_.append(float(loss))
            self.n_epochs_run_ = epoch + 1

            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            # Gradient of cross-entropy wrt logits is (probs - Y) / n.
            grad_logits = (probs - Y) / n_samples
            grad_W = Xs.T @ grad_logits
            if self.l2 > 0:
                grad_W = grad_W + self.l2 * W  # weights only, not the bias
            grad_b = grad_logits.sum(axis=0)
            W -= self.learning_rate * grad_W
            b -= self.learning_rate * grad_b

        self.coef_ = W
        self.intercept_ = b
        return self

    def predict_proba(self, X):
        """Predicted class probabilities, shape (n_samples, n_classes)."""
        X = self._validate_predict_X(X)
        Xs = (X - self.mean_) / self.scale_
        return _softmax(Xs @ self.coef_ + self.intercept_)

    def predict(self, X):
        """Predicted class label for each row of ``X``."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
