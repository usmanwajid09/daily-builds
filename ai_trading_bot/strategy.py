"""Strategy signal generation built on top of the from-scratch indicators.

A "strategy" here is just a function that takes a `close` price array and
returns an integer position array of the same length, one of:

    1  -> be long (fully invested)
    0  -> be flat (in cash)
   -1  -> be short (only meaningful if the backtester allows shorting)

Positions are decided using only information available up to and including
day `i` (no look-ahead), and take effect starting the *next* bar in the
backtester (see backtest.py) — you can't trade on a signal before it exists.
"""
from __future__ import annotations

import numpy as np

from .indicators import ema, macd, rsi, sma


def sma_crossover_signal(
    close: np.ndarray, fast: int = 20, slow: int = 50
) -> np.ndarray:
    """Classic trend-following signal: long while fast SMA > slow SMA.

    Flat (0) wherever either SMA hasn't warmed up yet.
    """
    if fast >= slow:
        raise ValueError("fast window must be < slow window")
    close = np.asarray(close, dtype=float)
    fast_sma = sma(close, fast)
    slow_sma = sma(close, slow)
    position = np.zeros(len(close), dtype=int)
    valid = ~np.isnan(fast_sma) & ~np.isnan(slow_sma)
    position[valid] = np.where(fast_sma[valid] > slow_sma[valid], 1, 0)
    return position


def macd_signal(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> np.ndarray:
    """Long while the MACD line is above its signal line, else flat."""
    close = np.asarray(close, dtype=float)
    macd_line, signal_line, _hist = macd(close, fast=fast, slow=slow, signal=signal)
    position = np.zeros(len(close), dtype=int)
    valid = ~np.isnan(macd_line) & ~np.isnan(signal_line)
    position[valid] = np.where(macd_line[valid] > signal_line[valid], 1, 0)
    return position


def rsi_mean_reversion_signal(
    close: np.ndarray,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> np.ndarray:
    """Mean-reversion signal: go long when RSI dips below `oversold`, exit
    (go flat) once RSI climbs back above `overbought`. Holds position
    between those two triggers (a simple stateful threshold strategy).
    """
    if not 0 <= oversold < overbought <= 100:
        raise ValueError("require 0 <= oversold < overbought <= 100")
    close = np.asarray(close, dtype=float)
    r = rsi(close, window=window)
    n = len(close)
    position = np.zeros(n, dtype=int)
    holding = False
    for i in range(n):
        if np.isnan(r[i]):
            continue
        if not holding and r[i] < oversold:
            holding = True
        elif holding and r[i] > overbought:
            holding = False
        position[i] = 1 if holding else 0
    return position


def combined_trend_and_momentum_signal(
    close: np.ndarray,
    sma_fast: int = 20,
    sma_slow: int = 50,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_window: int = 9,
) -> np.ndarray:
    """Long only when BOTH the SMA-crossover trend filter and the MACD
    momentum filter agree it's a long day. This is stricter (fewer, more
    confirmed trades) than either signal alone -- a common way to combine a
    trend indicator with a momentum indicator to cut down on whipsaws.
    """
    trend = sma_crossover_signal(close, fast=sma_fast, slow=sma_slow)
    momentum = macd_signal(
        close, fast=macd_fast, slow=macd_slow, signal=macd_signal_window
    )
    return np.where((trend == 1) & (momentum == 1), 1, 0)
