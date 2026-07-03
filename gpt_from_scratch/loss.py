"""Cross-entropy loss with a combined softmax + backward pass.

Combining softmax and cross-entropy into one function keeps the backward
pass numerically simple: dL/dlogits = (softmax(logits) - one_hot(target)) / N.
"""
from __future__ import annotations

import numpy as np


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray):
    """logits: (batch, seq, vocab); targets: (batch, seq) int ids.

    Returns (loss: float, dlogits: same shape as logits).
    """
    if logits.ndim != 3:
        raise ValueError("logits must be (batch, seq, vocab)")
    batch, seq, vocab = logits.shape
    if targets.shape != (batch, seq):
        raise ValueError(f"targets shape {targets.shape} must match (batch, seq)=({batch},{seq})")

    logits2 = logits.reshape(-1, vocab)
    targets2 = np.asarray(targets).reshape(-1)
    if targets2.min() < 0 or targets2.max() >= vocab:
        raise ValueError("targets contains an id outside [0, vocab_size)")

    # numerically stable log-softmax
    shifted = logits2 - logits2.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - logsumexp

    n = logits2.shape[0]
    rows = np.arange(n)
    loss = -log_probs[rows, targets2].mean()

    probs = np.exp(log_probs)
    dlogits2 = probs.copy()
    dlogits2[rows, targets2] -= 1.0
    dlogits2 /= n

    return float(loss), dlogits2.reshape(batch, seq, vocab)
