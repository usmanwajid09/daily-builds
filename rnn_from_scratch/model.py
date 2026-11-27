"""Vanilla RNN sequence classifier from scratch (numpy only).

A single-layer recurrent network with a tanh hidden state:

    h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)

The class label is predicted from the *final* hidden state only:

    logits = Why @ h_T + by
    probs  = softmax(logits)

Training uses full backpropagation through time (BPTT) with a
cross-entropy loss, plain mini-batch stochastic gradient descent
(batches of 32, one update per batch), and gradient clipping to
[-5, 5] to keep the recurrent gradients from exploding. Mini-batching
matters here: per-sample updates are too noisy for the parity task,
where the network must learn to flip its hidden state once per 1-bit,
and plain per-sample SGD stalls at chance accuracy.

The classic demo task is *parity*: given a sequence of 0/1 bits, predict
whether the number of 1s is odd or even. Parity is a genuinely sequential
problem -- no single time step carries the answer, so the network has to
carry information forward through the hidden state.
"""

import numpy as np


def make_parity_dataset(n_samples, seq_len=6, seed=42):
    """Generate a parity dataset.

    Returns ``(X, y)`` where ``X`` has shape ``(n_samples, seq_len, 2)``
    (one-hot encoded bits) and ``y`` is 0 for even parity, 1 for odd.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_samples, seq_len))
    y = (bits.sum(axis=1) % 2).astype(int)
    X = np.zeros((n_samples, seq_len, 2), dtype=float)
    X[np.arange(n_samples)[:, None], np.arange(seq_len), bits] = 1.0
    return X, y


class RNNClassifier:
    """Vanilla RNN sequence classifier trained with BPTT + SGD.

    Parameters
    ----------
    hidden_size : int, default 32
        Number of units in the recurrent hidden state.
    learning_rate : float, default 0.1
        SGD step size.
    epochs : int, default 100
        Number of passes over the training set.
    seed : int or None, default 42
        Seed for weight initialisation and per-epoch shuffling, so runs
        are reproducible. ``None`` gives non-deterministic initialisation.
    """

    def __init__(self, hidden_size=32, learning_rate=0.1, epochs=100, seed=42):
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, (int, np.integer)):
            raise ValueError("hidden_size must be an integer >= 1, "
                             f"got {hidden_size!r}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        if learning_rate is None or learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)):
            raise ValueError(f"epochs must be an integer >= 1, got {epochs!r}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")

        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.seed = seed
        self.loss_history_ = []
        self.classes_ = None

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_X_y(X, y):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (n_samples, seq_len, input_size), "
                             f"got shape {X.shape}")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must be numeric")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)} labels")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer labels")
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError(f"y must contain at least 2 classes, got {classes}")
        return X.astype(float), y, classes

    @staticmethod
    def _validate_X(X, expected_input_size=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (n_samples, seq_len, input_size), "
                             f"got shape {X.shape}")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must be numeric")
        if expected_input_size is not None and X.shape[2] != expected_input_size:
            raise ValueError(f"X has input_size {X.shape[2]}, expected "
                             f"{expected_input_size}")
        return X.astype(float)

    def _check_fitted(self):
        if self.classes_ is None:
            raise ValueError("model is not fitted yet -- call fit() first")

    # ------------------------------------------------------------------
    # forward / backward
    # ------------------------------------------------------------------
    def _init_params(self, input_size, n_classes, rng):
        h = self.hidden_size
        # Xavier-ish scales: variance ~ 1 / fan_in keeps tanh in its
        # well-behaved range at the start of training.
        self.Wxh_ = rng.normal(0.0, 1.0 / np.sqrt(input_size), (h, input_size))
        self.Whh_ = rng.normal(0.0, 1.0 / np.sqrt(h), (h, h))
        self.bh_ = np.zeros(h)
        self.Why_ = rng.normal(0.0, 1.0 / np.sqrt(h), (n_classes, h))
        self.by_ = np.zeros(n_classes)

    def _forward(self, x):
        """Run one sequence; return (probs, hidden states).

        ``hidden`` holds h_{-1} (= zeros) followed by h_0 ... h_{T-1}.
        """
        seq_len = x.shape[0]
        hidden = [np.zeros(self.hidden_size)]
        for t in range(seq_len):
            h = np.tanh(self.Wxh_ @ x[t] + self.Whh_ @ hidden[-1] + self.bh_)
            hidden.append(h)
        logits = self.Why_ @ hidden[-1] + self.by_
        logits = logits - logits.max()  # numerical stability
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return probs, hidden

    def _backward(self, x, probs, hidden, y_idx):
        """BPTT for one sequence; returns clipped gradients."""
        seq_len = x.shape[0]
        h = self.hidden_size
        n_classes = len(self.classes_)

        dlogits = probs.copy()
        dlogits[y_idx] -= 1.0  # softmax + cross-entropy gradient

        dWhy = np.outer(dlogits, hidden[-1])
        dby = dlogits.copy()
        dWxh = np.zeros_like(self.Wxh_)
        dWhh = np.zeros_like(self.Whh_)
        dbh = np.zeros_like(self.bh_)

        dh = self.Why_.T @ dlogits  # gradient flowing into h_T
        for t in reversed(range(seq_len)):
            # tanh'(z) = 1 - h_t^2
            dtanh = dh * (1.0 - hidden[t + 1] ** 2)
            dWxh += np.outer(dtanh, x[t])
            dWhh += np.outer(dtanh, hidden[t])
            dbh += dtanh
            dh = self.Whh_.T @ dtanh  # carry the gradient one step back

        grads = [dWxh, dWhh, dbh, dWhy, dby]
        return [np.clip(g, -5.0, 5.0) for g in grads]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Train the RNN with BPTT + mini-batch SGD. Returns self.

        Gradients are averaged over mini-batches of 32 sequences; one
        SGD update is applied per batch.
        """
        X, y, classes = self._validate_X_y(X, y)
        self.classes_ = classes
        n_classes = len(classes)
        y_idx = np.searchsorted(classes, y)  # labels -> 0..n_classes-1

        rng = np.random.default_rng(self.seed)
        self._init_params(X.shape[2], n_classes, rng)

        batch_size = 32
        params = [self.Wxh_, self.Whh_, self.bh_, self.Why_, self.by_]
        self.loss_history_ = []
        lr = self.learning_rate
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            total_loss = 0.0
            for start in range(0, len(X), batch_size):
                idxs = order[start:start + batch_size]
                grad_acc = [np.zeros_like(p) for p in params]
                for i in idxs:
                    probs, hidden = self._forward(X[i])
                    total_loss += -np.log(max(probs[y_idx[i]], 1e-15))
                    grads = self._backward(X[i], probs, hidden, y_idx[i])
                    for k in range(len(params)):
                        grad_acc[k] += grads[k] / len(idxs)
                self.Wxh_ -= lr * grad_acc[0]
                self.Whh_ -= lr * grad_acc[1]
                self.bh_ -= lr * grad_acc[2]
                self.Why_ -= lr * grad_acc[3]
                self.by_ -= lr * grad_acc[4]
            self.loss_history_.append(total_loss / len(X))
        return self

    def predict_proba(self, X):
        """Class probabilities, shape (n_samples, n_classes)."""
        self._check_fitted()
        X = self._validate_X(X, expected_input_size=self.Wxh_.shape[1])
        probs = np.array([self._forward(x)[0] for x in X])
        return probs

    def predict(self, X):
        """Predicted class labels for each sequence."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """Classification accuracy in [0, 1]."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
