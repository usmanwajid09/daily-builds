"""A minimal, from-scratch backtesting engine.

BACKTEST/PAPER-TRADING ONLY. This module never connects to a broker and
never places a real order -- it only simulates what a strategy's equity
curve *would have* looked like against historical or synthetic price data
already loaded in memory (see data.py). Nothing here is financial advice.

Design:
- Positions are decided using `strategy.py` signal functions, which look
  only at data up to and including day `i`.
- To avoid look-ahead bias, a signal computed at the close of day `i` is
  only acted on starting day `i + 1`'s return (you cannot trade on a
  close-of-day signal *during* that same day).
- Optional per-trade transaction cost (in basis points) is deducted from
  the equity curve whenever the position changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestResult:
    equity_curve: np.ndarray  # starts at `initial_capital`, one value per bar
    returns: np.ndarray  # daily strategy returns (same length as equity_curve)
    positions: np.ndarray  # the *effective* position held during each bar's return
    trades: int  # number of times the position changed
    initial_capital: float
    final_capital: float


def run_backtest(
    close: np.ndarray,
    signal: np.ndarray,
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """Simulate trading `signal` (a position array from strategy.py) against
    `close` prices.

    Parameters
    ----------
    close: daily close prices, oldest -> newest.
    signal: integer position (-1/0/1) decided at the close of each bar
        using only data available up to that bar. Must be same length as
        `close`.
    initial_capital: starting portfolio value.
    transaction_cost_bps: cost, in basis points of traded notional, charged
        whenever the position changes from one bar to the next (e.g. 5 bps
        = 0.05% round-trip-style friction per position change). Set to 0
        for a frictionless backtest.

    No look-ahead: the position used for bar `i`'s return is `signal[i-1]`
    (yesterday's decision), since you cannot act on information from a bar
    during that same bar's return. The very first bar therefore always has
    zero effective position (nothing to react to yet).
    """
    close = np.asarray(close, dtype=float)
    signal = np.asarray(signal, dtype=int)
    n = len(close)
    if len(signal) != n:
        raise ValueError(
            f"signal length ({len(signal)}) must match close length ({n})"
        )
    if n < 2:
        raise ValueError("need at least 2 bars to compute any return")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    daily_returns = np.diff(close) / close[:-1]  # length n-1, return for bar i -> i+1

    # effective_position[i] = position held while earning daily_returns[i-1]
    # (i.e. the position decided at the close of bar i-1, using signal[i-1]).
    effective_position = np.zeros(n, dtype=int)
    effective_position[1:] = signal[:-1]

    cost_rate = transaction_cost_bps / 10_000.0
    position_changes = np.diff(effective_position, prepend=0)
    trades = int(np.count_nonzero(position_changes))

    equity = np.empty(n, dtype=float)
    equity[0] = initial_capital
    for i in range(1, n):
        gross_return = effective_position[i] * daily_returns[i - 1]
        equity[i] = equity[i - 1] * (1.0 + gross_return)
        if position_changes[i] != 0:
            equity[i] *= 1.0 - cost_rate * abs(position_changes[i])

    strategy_returns = np.empty(n, dtype=float)
    strategy_returns[0] = 0.0
    strategy_returns[1:] = equity[1:] / equity[:-1] - 1.0

    return BacktestResult(
        equity_curve=equity,
        returns=strategy_returns,
        positions=effective_position,
        trades=trades,
        initial_capital=initial_capital,
        final_capital=float(equity[-1]),
    )


def buy_and_hold_baseline(
    close: np.ndarray, initial_capital: float = 10_000.0
) -> BacktestResult:
    """Convenience baseline: long every bar, no transaction costs.

    Useful to compare a strategy's equity curve against "just holding the
    asset the whole time." Uses the same `run_backtest` engine (and
    therefore the same no-look-ahead convention: effective position at
    bar 0 is 0, matching every strategy backtest), so it stays directly
    comparable to strategy results rather than getting a one-bar head
    start.
    """
    close = np.asarray(close, dtype=float)
    signal = np.ones(len(close), dtype=int)
    return run_backtest(
        close, signal, initial_capital=initial_capital, transaction_cost_bps=0.0
    )
