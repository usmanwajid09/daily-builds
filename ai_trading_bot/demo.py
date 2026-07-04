"""End-to-end demo: generate synthetic data, run a few strategies through the
backtester, and print a comparison report against a buy-and-hold baseline.

BACKTEST ONLY -- no live broker connection, no real orders. Run with:

    python -m ai_trading_bot.demo

Nothing in this file (or this package) is financial advice.
"""
from __future__ import annotations

import numpy as np

from .backtest import buy_and_hold_baseline, run_backtest
from .data import generate_synthetic_ohlcv
from .metrics import summarize
from .strategy import (
    combined_trend_and_momentum_signal,
    macd_signal,
    rsi_mean_reversion_signal,
    sma_crossover_signal,
)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def main() -> None:
    series = generate_synthetic_ohlcv(n_days=756, seed=42)  # ~3 trading years
    close = series.close

    strategies = {
        "buy_and_hold": None,  # handled specially below
        "sma_crossover(20,50)": lambda c: sma_crossover_signal(c, fast=20, slow=50),
        "macd(12,26,9)": lambda c: macd_signal(c, fast=12, slow=26, signal=9),
        "rsi_mean_reversion(14,30,70)": lambda c: rsi_mean_reversion_signal(
            c, window=14, oversold=30, overbought=70
        ),
        "combined_trend_momentum": lambda c: combined_trend_and_momentum_signal(c),
    }

    print(f"Synthetic backtest over {len(close)} trading days "
          f"({series.dates[0]} to {series.dates[-1]})")
    print(f"Start price: {close[0]:.2f}   End price: {close[-1]:.2f}   "
          f"Raw price change: {_fmt_pct(close[-1] / close[0] - 1)}")
    print()
    header = f"{'strategy':<32}{'total_return':>14}{'cagr':>10}{'sharpe':>10}{'max_dd':>10}{'win_rate':>10}{'trades':>8}"
    print(header)
    print("-" * len(header))

    for name, sig_fn in strategies.items():
        if sig_fn is None:
            result = buy_and_hold_baseline(close, initial_capital=10_000.0)
        else:
            signal = sig_fn(close)
            result = run_backtest(
                close, signal, initial_capital=10_000.0, transaction_cost_bps=5.0
            )
        stats = summarize(result.equity_curve, result.returns)
        print(
            f"{name:<32}"
            f"{_fmt_pct(stats['total_return']):>14}"
            f"{_fmt_pct(stats['cagr']):>10}"
            f"{stats['sharpe_ratio']:>10.2f}"
            f"{_fmt_pct(stats['max_drawdown']):>10}"
            f"{_fmt_pct(stats['win_rate']):>10}"
            f"{result.trades:>8d}"
        )

    print()
    print("Note: synthetic data + backtest-only. Past (or simulated) performance")
    print("is not indicative of future results. This is not financial advice.")


if __name__ == "__main__":
    main()
