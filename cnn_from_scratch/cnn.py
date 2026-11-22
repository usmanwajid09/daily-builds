"""Tiny convolutional neural network from scratch (numpy only).

Architecture (single-channel grayscale input)::

    input (H, W) -> Conv2D (F filters, kxk) -> ReLU -> MaxPool 2x2
                 -> flatten -> Dense -> softmax

Trained with plain SGD on softmax cross-entropy loss. Everything is
implemented with explicit forward/backward passes so each layer can be
tested on its own. Images are small (e.g. 12x12) so direct convolution
loops are fast enough.
"""

import numpy as np

__all__ = [
    "Conv2D",
    "ReLU",
    "MaxPool2D",
    "Dense",
    "softmax",
    "cross_entropy_loss",
    "TinyCNN",
    "make_circle",
    "make_square",
    "make_shape_dataset",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def softmax(logits):
    """Numerically stable softmax over the last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 0:
        raise ValueError("softmax expects at least a 1-D array, got a scalar.")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    total = np.sum(exp, axis=-1, keepdims=True)
    if np.any(total == 0):
        raise ValueError("softmax failed: all logits are -inf.")
    return exp / total


def cross_entropy_loss(probs, labels):
    """Mean cross-entropy loss. ``probs`` is (N, C), ``labels`` is (N,)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (N, C), got shape {probs.shape}.")
    if labels.shape != (probs.shape[0],):
        raise ValueError(
            f"labels must have shape ({probs.shape[0]},), got {labels.shape}."
        )
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("labels contain a class index outside [0, n_classes).")
    eps = 1e-12
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(picked, eps, 1.0))))


def _check_2d_image(img, name="image"):
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (H, W), got shape {img.shape}.")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(f"{name} must not be empty, got shape {img.shape}.")
    if not np.all(np.isfinite(img)):
        raise ValueError(f"{name} must contain only finite values.")
    return img


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
class Conv2D:
    """Single-channel 2-D convolution, stride 1, no padding."""

    def __init__(self, n_filters, kernel_size=3, rng=None):
        if not isinstance(n_filters, int) or n_filters <= 0:
            raise ValueError(f"n_filters must be a positive int, got {n_filters!r}.")
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive int, got {kernel_size!r}.")
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        rng = np.random.default_rng(rng)
        limit = np.sqrt(1.0 / (kernel_size * kernel_size))
        self.filters = rng.uniform(-limit, limit, size=(n_filters, kernel_size, kernel_size))
        self.bias = np.zeros(n_filters)
        self.d_filters = np.zeros_like(self.filters)
        self.d_bias = np.zeros_like(self.bias)
        self._cache = None

    @property
    def output_shape(self):
        return None  # depends on the input size; see forward()

    def forward(self, x):
        x = _check_2d_image(x, "input")
        k = self.kernel_size
        h, w = x.shape
        if h < k or w < k:
            raise ValueError(
                f"input {x.shape} is smaller than kernel size {k}x{k}."
            )
        oh, ow = h - k + 1, w - k + 1
        out = np.zeros((oh, ow, self.n_filters))
        for i in range(oh):
            for j in range(ow):
                patch = x[i:i + k, j:j + k]
                out[i, j, :] = np.sum(patch[None, :, :] * self.filters, axis=(1, 2)) + self.bias
        self._cache = x
        return out

    def backward(self, dout):
        if self._cache is None:
            raise ValueError("backward() called before forward().")
        x = self._cache
        k = self.kernel_size
        oh, ow, _ = dout.shape
        self.d_filters = np.zeros_like(self.filters)
        self.d_bias = np.sum(dout, axis=(0, 1))
        dx = np.zeros_like(x)
        for i in range(oh):
            for j in range(ow):
                patch = x[i:i + k, j:j + k]
                for f in range(self.n_filters):
                    self.d_filters[f] += dout[i, j, f] * patch
                    dx[i:i + k, j:j + k] += dout[i, j, f] * self.filters[f]
        return dx

    def step(self, lr):
        if lr <= 0:
            raise ValueError(f"learning rate must be positive, got {lr}.")
        self.filters -= lr * self.d_filters
        self.bias -= lr * self.d_bias


class ReLU:
    """Element-wise ReLU activation."""

    def __init__(self):
        self._mask = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            raise ValueError("ReLU input must not be empty.")
        self._mask = x > 0
        return np.where(self._mask, x, 0.0)

    def backward(self, dout):
        if self._mask is None:
            raise ValueError("backward() called before forward().")
        dout = np.asarray(dout, dtype=np.float64)
        if dout.shape != self._mask.shape:
            raise ValueError(
                f"gradient shape {dout.shape} does not match cached input shape {self._mask.shape}."
            )
        return dout * self._mask


class MaxPool2D:
    """2x2 max-pooling with stride 2 (drops a trailing odd row/column)."""

    def __init__(self, pool_size=2):
        if not isinstance(pool_size, int) or pool_size <= 0:
            raise ValueError(f"pool_size must be a positive int, got {pool_size!r}.")
        self.pool_size = pool_size
        self._cache = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"MaxPool2D expects a 3-D array (H, W, C), got shape {x.shape}.")
        h, w, c = x.shape
        p = self.pool_size
        if h < p or w < p:
            raise ValueError(f"input {x.shape} is smaller than pool size {p}.")
        oh, ow = h // p, w // p
        trimmed = x[:oh * p, :ow * p, :]
        windows = trimmed.reshape(oh, p, ow, p, c)
        out = windows.max(axis=(1, 3))
        # Remember where the maxima were for the backward pass.
        flat = windows.reshape(oh, ow, p * p, c)
        argmax = flat.argmax(axis=2)  # (oh, ow, c)
        self._cache = (x.shape, argmax)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise ValueError("backward() called before forward().")
        shape, argmax = self._cache
        h, w, c = shape
        p = self.pool_size
        oh, ow = h // p, w // p
        if dout.shape != (oh, ow, c):
            raise ValueError(
                f"gradient shape {dout.shape} does not match pooled shape {(oh, ow, c)}."
            )
        dx = np.zeros((oh * p, ow * p, c))
        for i in range(oh):
            for j in range(ow):
                for ch in range(c):
                    k = argmax[i, j, ch]
                    di, dj = divmod(int(k), p)
                    dx[i * p + di, j * p + dj, ch] += dout[i, j, ch]
        full = np.zeros(shape)
        full[:oh * p, :ow * p, :] = dx
        return full


class Dense:
    """Fully connected layer."""

    def __init__(self, in_features, out_features, rng=None):
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(f"in_features must be a positive int, got {in_features!r}.")
        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(f"out_features must be a positive int, got {out_features!r}.")
        rng = np.random.default_rng(rng)
        limit = np.sqrt(1.0 / in_features)
        self.weights = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.bias = np.zeros(out_features)
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)
        self._cache = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"Dense expects a 1-D input, got shape {x.shape}.")
        if x.shape[0] != self.weights.shape[0]:
            raise ValueError(
                f"input length {x.shape[0]} does not match in_features {self.weights.shape[0]}."
            )
        self._cache = x
        return x @ self.weights + self.bias

    def backward(self, dout):
        if self._cache is None:
            raise ValueError("backward() called before forward().")
        x = self._cache
        dout = np.asarray(dout, dtype=np.float64)
        if dout.shape != (self.weights.shape[1],):
            raise ValueError(
                f"gradient shape {dout.shape} does not match out_features {self.weights.shape[1]}."
            )
        self.d_weights = np.outer(x, dout)
        self.d_bias = dout.copy()
        return dout @ self.weights.T

    def step(self, lr):
        if lr <= 0:
            raise ValueError(f"learning rate must be positive, got {lr}.")
        self.weights -= lr * self.d_weights
        self.bias -= lr * self.d_bias


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class TinyCNN:
    """Conv -> ReLU -> MaxPool -> Dense -> softmax classifier.

    Parameters
    ----------
    input_shape : (H, W) of the grayscale input image.
    n_filters, kernel_size : convolution settings.
    n_classes : number of output classes.
    seed : random seed for reproducible weight initialisation.
    """

    def __init__(self, input_shape=(12, 12), n_filters=4, kernel_size=3,
                 n_classes=2, seed=0):
        if len(input_shape) != 2 or any(s <= 0 for s in input_shape):
            raise ValueError(f"input_shape must be (H, W) with positive sizes, got {input_shape!r}.")
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError(f"n_classes must be an int >= 2, got {n_classes!r}.")
        h, w = input_shape
        if h < kernel_size or w < kernel_size:
            raise ValueError(
                f"input {input_shape} is smaller than kernel size {kernel_size}."
            )
        pooled_h = (h - kernel_size + 1) // 2
        pooled_w = (w - kernel_size + 1) // 2
        if pooled_h <= 0 or pooled_w <= 0:
            raise ValueError("input is too small to survive conv + 2x2 pooling.")
        self.input_shape = tuple(input_shape)
        self.n_classes = n_classes
        self.conv = Conv2D(n_filters, kernel_size, rng=seed)
        self.relu = ReLU()
        self.pool = MaxPool2D(2)
        self.dense = Dense(pooled_h * pooled_w * n_filters, n_classes,
                           rng=seed + 1 if isinstance(seed, int) else None)
        self._cache = None

    def forward(self, x):
        x = _check_2d_image(x, "input")
        if x.shape != self.input_shape:
            raise ValueError(
                f"expected input shape {self.input_shape}, got {x.shape}."
            )
        conv_out = self.conv.forward(x)
        relu_out = self.relu.forward(conv_out)
        pooled = self.pool.forward(relu_out)
        flat = pooled.reshape(-1)
        logits = self.dense.forward(flat)
        probs = softmax(logits)
        self._cache = probs
        return probs

    def _backward(self, probs, label):
        dlogits = probs.copy()
        dlogits[label] -= 1.0
        dflat = self.dense.backward(dlogits)
        pooled_shape = self.pool._cache[0] if self.pool._cache else None
        dpooled = dflat.reshape(self._pooled_shape())
        drelu = self.pool.backward(dpooled)
        dconv = self.relu.backward(drelu)
        self.conv.backward(dconv)

    def _pooled_shape(self):
        h, w = self.input_shape
        k = self.conv.kernel_size
        return ((h - k + 1) // 2, (w - k + 1) // 2, self.conv.n_filters)

    def fit(self, X, y, epochs=30, lr=0.05, verbose=False):
        """Train with SGD, one sample at a time.

        Returns a list of per-epoch mean losses.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (N, H, W), got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if y.shape != (X.shape[0],):
            raise ValueError(
                f"y must have shape ({X.shape[0]},), got {y.shape}."
            )
        if np.any(y < 0) or np.any(y >= self.n_classes):
            raise ValueError("y contains a class index outside [0, n_classes).")
        if X.shape[1:] != self.input_shape:
            raise ValueError(
                f"each sample must have shape {self.input_shape}, got {X.shape[1:]}."
            )
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"epochs must be a positive int, got {epochs!r}.")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}.")
        losses = []
        n = len(y)
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(n):
                probs = self.forward(X[i])
                total_loss += cross_entropy_loss(probs[None, :], y[i:i + 1])
                self._backward(probs, int(y[i]))
                self.conv.step(lr)
                self.dense.step(lr)
            losses.append(total_loss / n)
            if verbose:
                print(f"epoch {epoch + 1}/{epochs} loss={losses[-1]:.4f}")
        return losses

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[None, :, :]
        if X.ndim != 3:
            raise ValueError(f"X must be 2-D or 3-D, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        return np.array([int(np.argmax(self.forward(x))) for x in X])

    def score(self, X, y):
        """Classification accuracy on (X, y)."""
        y = np.asarray(y)
        preds = self.predict(X)
        if preds.shape != y.shape:
            raise ValueError(f"y shape {y.shape} does not match predictions {preds.shape}.")
        return float(np.mean(preds == y))


# ---------------------------------------------------------------------------
# Synthetic shape data
# ---------------------------------------------------------------------------
def make_circle(size=12, centre=None, radius=None, rng=None):
    """A filled circle on a black background, values in [0, 1]."""
    rng = np.random.default_rng(rng)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    centre = centre or (size // 2 + int(rng.integers(-2, 3)),
                        size // 2 + int(rng.integers(-2, 3)))
    radius = radius if radius is not None else size // 4 + int(rng.integers(0, 2))
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")
    yy, xx = np.mgrid[0:size, 0:size]
    img = ((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius ** 2).astype(np.float64)
    return img


def make_square(size=12, centre=None, half_side=None, rng=None):
    """A filled square on a black background, values in [0, 1]."""
    rng = np.random.default_rng(rng)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    centre = centre or (size // 2 + int(rng.integers(-2, 3)),
                        size // 2 + int(rng.integers(-2, 3)))
    half_side = half_side if half_side is not None else size // 4
    if half_side <= 0:
        raise ValueError(f"half_side must be positive, got {half_side}.")
    yy, xx = np.mgrid[0:size, 0:size]
    img = ((np.abs(yy - centre[0]) <= half_side)
           & (np.abs(xx - centre[1]) <= half_side)).astype(np.float64)
    return img


def make_shape_dataset(n_per_class=40, size=12, noise=0.08, seed=0):
    """Synthetic circle (label 0) vs square (label 1) dataset.

    Returns (X, y): X is (2*n_per_class, size, size), y is (2*n_per_class,).
    """
    if n_per_class <= 0:
        raise ValueError(f"n_per_class must be positive, got {n_per_class}.")
    if noise < 0:
        raise ValueError(f"noise must be >= 0, got {noise}.")
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for _ in range(n_per_class):
        images.append(make_circle(size=size, rng=rng))
        labels.append(0)
        images.append(make_square(size=size, rng=rng))
        labels.append(1)
    X = np.stack(images)
    if noise > 0:
        X = np.clip(X + rng.normal(0, noise, size=X.shape), 0, 1)
    y = np.array(labels)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]
