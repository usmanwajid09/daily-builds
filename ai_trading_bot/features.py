"""Feature engineering for the next-day-direction ML model.

Every feature at row `t` is computed using only data available through bar
`t` (no future information), so a feature matrix built here can be fed
straight into a walk-forward train/test split without introducing
look-ahead bias. The label, by contrast, is deliberately about the future
(bar `t+1`'s direction relative to bar `t`) -- that's the whole point of a
supervised *prediction* target, and it lines up with how `backtest.py`
already treats a signal decided at bar `t` as acting on the `t -> t+1`
return (see backtest.py's no-look-ahead note).
"""
from __future__ import annotations

import numpy as np

from .indicators import ema, macd, rsi, sma


FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "sma20_ratio",
    "rsi14_scaled",
    "macd_hist_norm",
    "volume_change",
]


def build_features(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Build a (n, len(FEATURE_NAMES)) feature matrix from close/volume.

    Rows with any NaN feature (due to indicator warm-up periods) are left
    as NaN -- callers should drop them (see `build_dataset`) rather than
    have this function silently decide a cutoff, so the same matrix can be
    reused for both training and later live/paper-trading feature lookups.
    """
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    n = len(close)
    if len(volume) != n:
        raise ValueError("close and volume must be the same length")
    if n < 2:
        raise ValueError("need at least 2 bars to compute any feature")

    return_1d = np.full(n, np.nan)
    return_1d[1:] = (close[1:] - close[:-1]) / close[:-1]

    return_5d = np.full(n, np.nan)
    if n > 5:
        return_5d[5:] = (close[5:] - close[:-5]) / close[:-5]

    sma20 = sma(close, window=20)
    sma20_ratio = close / sma20 - 1.0

    rsi14 = rsi(close, window=14)
    rsi14_scaled = rsi14 / 100.0

    macd_line, signal_line, hist = macd(close, fast=12, slow=26, signal=9)
    # Normalize by price so the feature is comparable across different
    # price levels/assets rather than being in raw price units.
    macd_hist_norm = hist / close

    volume_change = np.full(n, np.nan)
    nonzero_prev_vol = volume[:-1] != 0
    volume_change[1:][nonzero_prev_vol] = (
        volume[1:][nonzero_prev_vol] - volume[:-1][nonzero_prev_vol]
    ) / volume[:-1][nonzero_prev_vol]

    return np.column_stack(
        [return_1d, return_5d, sma20_ratio, rsi14_scaled, macd_hist_norm, volume_change]
    )


def build_labels(close: np.ndarray) -> np.ndarray:
    """Next-day direction label: 1 if close[t+1] > close[t] else 0.

    The last bar has no "next day" yet, so its label is NaN -- the caller
    must drop it (it can never be used for training or evaluation).
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    labels = np.full(n, np.nan)
    labels[:-1] = (close[1:] > close[:-1]).astype(float)
    return labels


def build_dataset(
    close: np.ndarray, volume: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a clean (no-NaN) feature matrix + label vector for training.

    Returns (X, y, valid_idx) where `valid_idx` are the original bar
    indices each row of X/y corresponds to -- callers need this to map
    predictions back onto the original close/volume series (e.g. to run
    the existing backtest engine, or to know which bar a paper-trading
    decision applies to).
    """
    features = build_features(close, volume)
    labels = build_labels(close)
    valid = ~np.any(np.isnan(features), axis=1) & ~np.isnan(labels)
    valid_idx = np.nonzero(valid)[0]
    return features[valid], labels[valid], valid_idx
