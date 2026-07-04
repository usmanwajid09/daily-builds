"""Technical indicators implemented from scratch (no ta-lib / pandas-ta).

Every function takes and returns plain numpy arrays so they can be used
independently of the rest of the package. Where an indicator needs a
"warm-up" period (e.g. SMA needs `window` prior points), the returned array
is the same length as the input with `np.nan` in the positions that don't
yet have enough history.
"""
from __future__ import annotations

import numpy as np


def sma(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average over `window` periods."""
    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(values)
    out = np.full(n, np.nan)
    if window > n:
        return out
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def ema(values: np.ndarray, window: int) -> np.ndarray:
    """Exponential moving average over `window` periods.

    Seeded with the SMA of the first `window` values (a common convention),
    then recursively smoothed with alpha = 2 / (window + 1). Positions
    before the seed index are `nan`, matching `sma`'s warm-up behavior.
    """
    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(values)
    out = np.full(n, np.nan)
    if window > n:
        return out
    alpha = 2.0 / (window + 1)
    seed = values[:window].mean()
    out[window - 1] = seed
    prev = seed
    for i in range(window, n):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi(values: np.ndarray, window: int = 14) -> np.ndarray:
    """Relative Strength Index (Wilder's smoothing), 0-100 scale.

    Standard definition: average gain / average loss over `window` periods,
    smoothed with Wilder's method (equivalent to an EMA with alpha=1/window)
    after an initial simple average. First `window` values are `nan`.
    """
    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(values)
    out = np.full(n, np.nan)
    if n <= window:
        return out

    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:window].mean()
    avg_loss = losses[:window].mean()

    def _rsi_from_avgs(ag: float, al: float) -> float:
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    out[window] = _rsi_from_avgs(avg_gain, avg_loss)
    for i in range(window + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (window - 1) + g) / window
        avg_loss = (avg_loss * (window - 1) + l) / window
        out[i] = _rsi_from_avgs(avg_gain, avg_loss)
    return out


def macd(
    values: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence Divergence.

    Returns (macd_line, signal_line, histogram), each the same length as
    `values`. macd_line = EMA(fast) - EMA(slow); signal_line = EMA(macd_line,
    signal); histogram = macd_line - signal_line.

    Note: unlike `ema()` in isolation, the signal-line EMA here is seeded
    from the first *valid* (non-nan) macd_line value rather than re-running
    `ema()` on an array that still contains leading nans, since `ema()`
    seeds from a plain window-mean and would choke on nans.
    """
    if fast >= slow:
        raise ValueError("fast period must be < slow period")
    values = np.asarray(values, dtype=float)
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = ema_fast - ema_slow  # nan until ema_slow warms up (index slow-1)

    n = len(values)
    signal_line = np.full(n, np.nan)
    first_valid = slow - 1
    if n - first_valid >= signal:
        alpha = 2.0 / (signal + 1)
        seed_start = first_valid
        seed_end = first_valid + signal
        seed = macd_line[seed_start:seed_end].mean()
        signal_line[seed_end - 1] = seed
        prev = seed
        for i in range(seed_end, n):
            prev = alpha * macd_line[i] + (1 - alpha) * prev
            signal_line[i] = prev

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
