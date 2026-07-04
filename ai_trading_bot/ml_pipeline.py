"""Walk-forward train/test pipeline: wires `features.py` + `ml_model.py`
together and turns the model's predictions into a signal array the
existing `backtest.py` engine can run directly.

Walk-forward, not shuffled: this is a time series, so training on rows
that come chronologically after some test rows would leak future
information into what's supposed to be an out-of-sample evaluation. The
split is always "first N rows train, the rest test" (see `split_dataset`)
-- never a random shuffle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import build_dataset
from .ml_model import LogisticRegressionGD, StandardScaler


@dataclass
class WalkForwardResult:
    model: LogisticRegressionGD
    scaler: StandardScaler
    train_idx: np.ndarray  # original bar indices used for training
    test_idx: np.ndarray  # original bar indices held out for testing
    y_test: np.ndarray
    test_pred: np.ndarray
    test_proba: np.ndarray


def split_dataset(
    X: np.ndarray, y: np.ndarray, valid_idx: np.ndarray, train_frac: float = 0.7
):
    """Chronological (walk-forward) train/test split -- NOT a random
    shuffle. Returns (X_train, y_train, train_idx, X_test, y_test, test_idx).
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be strictly between 0 and 1")
    n = len(X)
    split = int(n * train_frac)
    if split < 1 or split >= n:
        raise ValueError(
            f"train_frac={train_frac} leaves too few rows for a train or "
            f"test set with only {n} total rows"
        )
    return (
        X[:split],
        y[:split],
        valid_idx[:split],
        X[split:],
        y[split:],
        valid_idx[split:],
    )


def train_direction_model(
    close: np.ndarray,
    volume: np.ndarray,
    train_frac: float = 0.7,
    learning_rate: float = 0.1,
    n_iters: int = 2000,
    l2: float = 0.01,
) -> WalkForwardResult:
    """Build features/labels, split chronologically, fit a scaler on the
    training rows ONLY (never on test rows -- fitting on the full dataset
    would leak test-set statistics into an evaluation meant to measure
    generalization), then train logistic regression and evaluate on the
    held-out test rows.
    """
    X, y, valid_idx = build_dataset(close, volume)
    X_train, y_train, train_idx, X_test, y_test, test_idx = split_dataset(
        X, y, valid_idx, train_frac=train_frac
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegressionGD(
        learning_rate=learning_rate, n_iters=n_iters, l2=l2
    ).fit(X_train_scaled, y_train)

    test_proba = model.predict_proba(X_test_scaled)
    test_pred = (test_proba >= 0.5).astype(int)

    return WalkForwardResult(
        model=model,
        scaler=scaler,
        train_idx=train_idx,
        test_idx=test_idx,
        y_test=y_test,
        test_pred=test_pred,
        test_proba=test_proba,
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, precision, recall, and a 2x2 confusion matrix for the
    binary up/down prediction task.

    Precision/recall are reported for the "predicted up" (1) class, since
    that's the class that actually triggers a trade in `signal_from_predictions`
    -- a false positive here (predicted up, actually down) is the costly
    mistake for this strategy, so precision on class 1 is the more
    decision-relevant number, not just an arbitrary pick.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics on an empty array")

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def signal_from_predictions(
    n_bars: int, test_idx: np.ndarray, test_pred: np.ndarray
) -> np.ndarray:
    """Build a full-length (n_bars,) position signal from test-set
    predictions: 1 on bars the model predicts "up" AND that are in the
    held-out test set, 0 everywhere else (training-period bars, indicator
    warm-up bars, and predicted-down test bars).

    IMPORTANT: bars before `test_idx[0]` are zero-signal *padding*, not a
    real "stay flat" decision -- when reporting performance, slice the
    resulting equity curve/returns to `test_idx[0]:` (or later) so the
    training period doesn't get counted as part of the strategy's (flat,
    zero-return) track record.
    """
    if len(test_idx) != len(test_pred):
        raise ValueError("test_idx and test_pred must be the same length")
    signal = np.zeros(n_bars, dtype=int)
    signal[test_idx] = test_pred
    return signal
