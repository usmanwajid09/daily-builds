"""Mock paper-trading simulation loop.

BACKTEST/PAPER-TRADING ONLY. This module never connects to a real broker,
never places a real order, and never touches real money -- everything
here is an in-memory `PaperAccount` walked forward one bar at a time
against historical or synthetic close prices already in memory. Nothing
in this file (or this package) is financial advice.

This is deliberately a *different shape* than `backtest.run_backtest`:
that engine is vectorized (whole arrays at once) for fast strategy
comparison. This module is an explicit step-by-step loop with a
persistent account object and a running trade log -- structurally closer
to what a real live-trading main loop looks like (poll for the latest
bar, decide, act, log, repeat), just aimed at historical/synthetic data
and a mock account instead of a real feed and a real broker connection.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PaperAccount:
    """An in-memory, mock trading account. No real money, ever."""

    cash: float
    shares: float = 0.0
    trade_log: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cash < 0:
            raise ValueError("cash must be non-negative")

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def is_long(self) -> bool:
        return self.shares > 0

    def go_long(self, date, price: float, transaction_cost_bps: float) -> None:
        """Deploy all cash into the position, if not already long."""
        if self.is_long():
            return
        if price <= 0:
            raise ValueError(f"cannot trade at non-positive price {price}")
        cost_rate = transaction_cost_bps / 10_000.0
        self.shares = (self.cash * (1.0 - cost_rate)) / price
        self.trade_log.append(
            {
                "date": date,
                "action": "BUY",
                "price": price,
                "shares": self.shares,
                "cash_after": 0.0,
            }
        )
        self.cash = 0.0

    def go_flat(self, date, price: float, transaction_cost_bps: float) -> None:
        """Liquidate the position back to cash, if currently long."""
        if not self.is_long():
            return
        if price <= 0:
            raise ValueError(f"cannot trade at non-positive price {price}")
        cost_rate = transaction_cost_bps / 10_000.0
        proceeds = self.shares * price * (1.0 - cost_rate)
        self.trade_log.append(
            {
                "date": date,
                "action": "SELL",
                "price": price,
                "shares": self.shares,
                "cash_after": proceeds,
            }
        )
        self.cash = proceeds
        self.shares = 0.0


def run_mock_paper_trading(
    dates: list,
    close: np.ndarray,
    desired_position: np.ndarray,
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
) -> tuple[PaperAccount, np.ndarray]:
    """Step through bars one at a time, simulating a live trading loop.

    Same no-look-ahead DECISION TIMING as `backtest.run_backtest`: the
    desired position decided using data through bar `i-1`
    (`desired_position[i-1]`) is the earliest that can be acted on, at bar
    `i` -- that decision couldn't have been made before bar `i-1` closed,
    so bar `i` is the first opportunity to act on it. Bar 0 is always
    observation-only (nothing to react to yet), matching `run_backtest`.

    IMPORTANT DIFFERENCE from `run_backtest`'s FILL PRICE: `run_backtest`
    is an idealized, vectorized model that applies the bar `i-1 -> i` %
    return directly to existing equity, as if the position were already
    held continuously through that move. This loop instead executes an
    actual BUY/SELL at bar `i`'s price when the position changes -- which
    is what a live order placed after seeing bar `i-1`'s close would
    realistically fill at. Consequence: the bar on which you *first* go
    long does NOT capture that bar's price move (you're buying in at the
    new price, not retroactively at yesterday's close); only bars where
    you're *already* long benefit from further price appreciation. This
    makes `run_mock_paper_trading`'s numbers a bit more conservative/
    realistic than `run_backtest`'s for the entry bar specifically -- see
    `test_run_mock_paper_trading_no_lookahead_bar_zero` for a worked
    example, and REVIEW.md for why this wasn't "fixed" to match
    `run_backtest` exactly.

    Returns (account, equity_curve) where `equity_curve[i]` is the mark-
    to-market account value at bar `i`'s close.
    """
    close = np.asarray(close, dtype=float)
    desired_position = np.asarray(desired_position, dtype=int)
    n = len(close)
    if not (len(dates) == len(desired_position) == n):
        raise ValueError(
            f"dates ({len(dates)}), close ({n}), and desired_position "
            f"({len(desired_position)}) must all be the same length"
        )
    if n < 1:
        raise ValueError("need at least 1 bar to run the simulation")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    account = PaperAccount(cash=initial_capital)
    equity_curve = np.empty(n)
    equity_curve[0] = account.equity(close[0])

    for i in range(1, n):
        desired = desired_position[i - 1]
        if desired == 1:
            account.go_long(dates[i], close[i], transaction_cost_bps)
        else:
            account.go_flat(dates[i], close[i], transaction_cost_bps)
        equity_curve[i] = account.equity(close[i])

    return account, equity_curve
