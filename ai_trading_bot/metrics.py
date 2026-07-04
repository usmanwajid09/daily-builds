"""Backtest performance metrics, implemented from scratch.

All functions take a `returns` array (per-bar simple returns, as produced by
`backtest.run_backtest(...).returns`) and/or an `equity_curve` array.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def total_return(equity_curve: np.ndarray) -> float:
    """Overall return over the full backtest, e.g. 0.25 == +25%."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) < 1 or equity_curve[0] == 0:
        raise ValueError("equity_curve must be non-empty with a non-zero start")
    return float(equity_curve[-1] / equity_curve[0] - 1.0)


def cagr(equity_curve: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate implied by the equity curve's length."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    n = len(equity_curve)
    if n < 2:
        raise ValueError("need at least 2 points to annualize a growth rate")
    if equity_curve[0] <= 0 or equity_curve[-1] <= 0:
        raise ValueError("equity_curve values must be positive")
    years = (n - 1) / periods_per_year
    if years <= 0:
        raise ValueError("periods_per_year too small for series length")
    growth = equity_curve[-1] / equity_curve[0]
    return float(growth ** (1.0 / years) - 1.0)


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio of a per-period returns series.

    `risk_free_rate` is annual; it's converted to a per-period rate before
    subtracting. Returns 0.0 (rather than raising or returning inf/nan) if
    the excess-return series has zero variance, since a strategy with
    literally zero variance is undefined risk-adjusted return by this
    formula, and 0.0 is a safer default than crashing a report.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        raise ValueError("need at least 2 returns to compute a Sharpe ratio")
    per_period_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - per_period_rf
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough decline, as a negative fraction (e.g. -0.32
    means the worst drawdown was -32% from a prior peak).
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) < 1:
        raise ValueError("equity_curve must be non-empty")
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1.0
    return float(drawdowns.min())


def win_rate(returns: np.ndarray) -> float:
    """Fraction of *non-zero* per-period returns that were positive.

    Zero-return bars (e.g. flat/no-position days) are excluded from the
    denominator since they're neither a win nor a loss -- including them
    would understate win rate for a strategy that's frequently in cash.
    """
    returns = np.asarray(returns, dtype=float)
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float(np.count_nonzero(nonzero > 0) / len(nonzero))


def summarize(
    equity_curve: np.ndarray,
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> dict:
    """Bundle the common metrics into a single dict, e.g. for a report."""
    return {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(
            returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        ),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(returns),
    }
