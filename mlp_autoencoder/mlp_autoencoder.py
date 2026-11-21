"""MLP autoencoder from scratch (numpy only).

A symmetric multilayer-perceptron autoencoder
``input -> tanh hidden -> tanh latent -> tanh hidden -> linear output``
trained by minibatch gradient descent on mean squared reconstruction
error. It compresses inputs through a low-dimensional bottleneck and
reconstructs them, so the per-sample reconstruction error doubles as
an anomaly score: samples the network has never seen reconstruct
badly and score high.
"""

import numpy as np

__all__ = ["MLPAutoencoder"]


def _check_2d_float(X, name="X"):
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


class MLPAutoencoder:
    """MLP autoencoder with one hidden layer on each side of the bottleneck.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Width of the hidden layers (encoder and decoder share it).
    latent_dim : int
        Bottleneck width; must be smaller than ``input_dim``.
    seed : int
        Random seed for weight initialisation (deterministic by default).
    """

    def __init__(self, input_dim, hidden_dim=32, latent_dim=2, seed=0):
        for name, value in (("input_dim", input_dim),
                            ("hidden_dim", hidden_dim),
                            ("latent_dim", latent_dim)):
            if not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        input_dim, hidden_dim, latent_dim = int(input_dim), int(hidden_dim), int(latent_dim)
        if latent_dim >= input_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be smaller than input_dim "
                f"({input_dim}) so the network has a real bottleneck")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seed = seed
        rng = np.random.default_rng(seed)

        def _init(fan_in, fan_out):
            return rng.normal(0.0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))

        self.W1 = _init(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = _init(hidden_dim, latent_dim)
        self.b2 = np.zeros(latent_dim)
        self.W3 = _init(latent_dim, hidden_dim)
        self.b3 = np.zeros(hidden_dim)
        self.W4 = _init(hidden_dim, input_dim)
        self.b4 = np.zeros(input_dim)
        self.history_ = []
        self.train_error_median_ = None
        self.is_fitted_ = False

    # -- forward pass -------------------------------------------------
    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.tanh(z2)  # latent code
        z3 = a2 @ self.W3 + self.b3
        a3 = np.tanh(z3)
        out = a3 @ self.W4 + self.b4
        cache = (X, a1, a2, a3, out)
        return out, cache

    @staticmethod
    def _loss(X, out):
        return float(np.mean((out - X) ** 2))

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("model is not fitted yet; call fit() first")

    # -- training ------------------------------------------------------
    def fit(self, X, epochs=200, lr=0.01, batch_size=64, verbose=False):
        """Train on ``X`` (shape ``(n_samples, input_dim)``).

        Optimised with Adam (implemented here from scratch). Returns
        ``self``. ``history_`` holds the mean squared error after each
        epoch.
        """
        X = _check_2d_float(X)
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} features but input_dim is {self.input_dim}")
        if not isinstance(epochs, (int, np.integer)) or int(epochs) < 1:
            raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr!r}")
        if not isinstance(batch_size, (int, np.integer)) or int(batch_size) < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        epochs, batch_size = int(epochs), int(batch_size)
        rng = np.random.default_rng(self.seed + 1)
        n = X.shape[0]
        params = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]
        m_adam = {p: np.zeros_like(getattr(self, p)) for p in params}
        v_adam = {p: np.zeros_like(getattr(self, p)) for p in params}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        self.history_ = []
        for epoch in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = X[order[start:start + batch_size]]
                out, (xb, a1, a2, a3, _) = self._forward(batch)
                m = batch.shape[0]
                dout = 2.0 * (out - batch) / (m * batch.shape[1])
                dW4 = a3.T @ dout
                db4 = dout.sum(axis=0)
                da3 = dout @ self.W4.T
                dz3 = da3 * (1.0 - a3 ** 2)
                dW3 = a2.T @ dz3
                db3 = dz3.sum(axis=0)
                da2 = dz3 @ self.W3.T
                dz2 = da2 * (1.0 - a2 ** 2)
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (1.0 - a1 ** 2)
                dW1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)
                grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                         "W3": dW3, "b3": db3, "W4": dW4, "b4": db4}
                step += 1
                for p in params:
                    g = grads[p]
                    m_adam[p] = beta1 * m_adam[p] + (1 - beta1) * g
                    v_adam[p] = beta2 * v_adam[p] + (1 - beta2) * g * g
                    m_hat = m_adam[p] / (1 - beta1 ** step)
                    v_hat = v_adam[p] / (1 - beta2 ** step)
                    setattr(self, p, getattr(self, p)
                            - lr * m_hat / (np.sqrt(v_hat) + eps))
            out, _ = self._forward(X)
            loss = self._loss(X, out)
            self.history_.append(loss)
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"epoch {epoch + 1}/{epochs}: mse={loss:.6f}")
        self.is_fitted_ = True
        self.train_error_median_ = float(np.median(self.reconstruction_error(X)))
        return self

    # -- inference -----------------------------------------------------
    def encode(self, X):
        """Map ``X`` to latent codes of shape ``(n_samples, latent_dim)``."""
        self._check_fitted()
        X = _check_2d_float(X)
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} features but input_dim is {self.input_dim}")
        _, (_, _, a2, _, _) = self._forward(X)
        return a2

    def decode(self, Z):
        """Map latent codes ``Z`` back to input space."""
        self._check_fitted()
        Z = _check_2d_float(Z, name="Z")
        if Z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Z has {Z.shape[1]} features but latent_dim is {self.latent_dim}")
        a3 = np.tanh(Z @ self.W3 + self.b3)
        return a3 @ self.W4 + self.b4

    def reconstruct(self, X):
        """Reconstruct ``X`` through the bottleneck."""
        self._check_fitted()
        X = _check_2d_float(X)
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} features but input_dim is {self.input_dim}")
        out, _ = self._forward(X)
        return out

    def reconstruction_error(self, X):
        """Per-sample mean squared reconstruction error, shape ``(n_samples,)``."""
        self._check_fitted()
        X = _check_2d_float(X)
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {X.shape[1]} features but input_dim is {self.input_dim}")
        out, _ = self._forward(X)
        return np.mean((out - X) ** 2, axis=1)

    def anomaly_score(self, X):
        """Reconstruction error divided by the median training error.

        Normal samples score near 1; anomalies score well above 1.
        """
        errors = self.reconstruction_error(X)
        scale = self.train_error_median_ if self.train_error_median_ else 1.0
        return errors / max(scale, 1e-12)
