"""End-to-end demo for Milestone 2: train a next-day-direction model on a
walk-forward split, backtest its out-of-sample predictions, and run the
same predictions through the mock paper-trading loop.

BACKTEST/PAPER-TRADING ONLY -- no live broker, no real orders, no real
money, anywhere in this file. Run with:

    python -m ai_trading_bot.demo_ml

NOT FINANCIAL ADVICE. This demo exists to show the pipeline working
end-to-end on synthetic data; the printed accuracy/return numbers are
about this specific synthetic series and this specific simple model, not
a claim that this strategy would work on real markets.
"""
from __future__ import annotations

import numpy as np

from .backtest import run_backtest
from .data import generate_synthetic_ohlcv
from .metrics import summarize
from .ml_pipeline import classification_metrics, signal_from_predictions, train_direction_model
from .paper_trading import run_mock_paper_trading


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def main() -> None:
    series = generate_synthetic_ohlcv(n_days=1000, seed=42)
    close = series.close
    dates = series.dates

    result = train_direction_model(
        close, series.volume, train_frac=0.7, learning_rate=0.1, n_iters=2000, l2=0.01
    )

    print(f"Synthetic series: {len(close)} bars ({dates[0]} to {dates[-1]})")
    print(
        f"Walk-forward split: {len(result.train_idx)} train rows "
        f"(bars {result.train_idx[0]}-{result.train_idx[-1]}), "
        f"{len(result.test_idx)} test rows "
        f"(bars {result.test_idx[0]}-{result.test_idx[-1]})"
    )
    print()

    metrics = classification_metrics(result.y_test, result.test_pred)
    cm = metrics["confusion_matrix"]
    print("Out-of-sample classification metrics (predicting next-day up/down):")
    print(f"  accuracy:  {metrics['accuracy']:.3f}")
    print(f"  precision: {metrics['precision']:.3f}  (of predicted 'up' days, how many were actually up)")
    print(f"  recall:    {metrics['recall']:.3f}  (of actual 'up' days, how many were predicted)")
    print(f"  confusion matrix: tp={cm['tp']} fp={cm['fp']} tn={cm['tn']} fn={cm['fn']}")
    print(
        f"  (for reference, always predicting the majority class would score "
        f"{max(result.y_test.mean(), 1 - result.y_test.mean()):.3f} accuracy)"
    )
    print()

    # --- Vectorized backtest (backtest.py) on the out-of-sample signal ---
    signal = signal_from_predictions(len(close), result.test_idx, result.test_pred)
    bt = run_backtest(close, signal, initial_capital=10_000.0, transaction_cost_bps=5.0)
    test_start = result.test_idx[0]
    test_equity = bt.equity_curve[test_start:]
    test_returns = bt.returns[test_start:]
    stats = summarize(test_equity, test_returns)
    print("Vectorized backtest (backtest.run_backtest), out-of-sample period only:")
    print(
        f"  total_return={_fmt_pct(stats['total_return'])}  "
        f"cagr={_fmt_pct(stats['cagr'])}  sharpe={stats['sharpe_ratio']:.2f}  "
        f"max_dd={_fmt_pct(stats['max_drawdown'])}  trades={sum(1 for i in range(test_start, len(bt.positions)) if bt.positions[i] != bt.positions[i-1])}"
    )
    print()

    # --- Mock paper-trading loop (paper_trading.py) over the same period ---
    test_dates = dates[test_start:]
    test_close = close[test_start:]
    test_signal = signal[test_start:]
    account, equity_curve = run_mock_paper_trading(
        test_dates, test_close, test_signal, initial_capital=10_000.0, transaction_cost_bps=5.0
    )
    paper_return = equity_curve[-1] / equity_curve[0] - 1.0
    print("Mock paper-trading loop (paper_trading.run_mock_paper_trading), same period:")
    print(f"  final equity: ${equity_curve[-1]:,.2f}  (total_return={_fmt_pct(paper_return)})")
    print(f"  trades executed: {len(account.trade_log)}")
    if account.trade_log:
        first, last = account.trade_log[0], account.trade_log[-1]
        print(f"  first trade: {first['action']} on {first['date']} @ {first['price']:.2f}")
        print(f"  last trade:  {last['action']} on {last['date']} @ {last['price']:.2f}")

    print()
    print("Note: synthetic data, backtest/paper-trading only, one simple linear model.")
    print("Not financial advice. Past (or simulated) performance is not indicative of")
    print("future results, and a model that beats buy-and-hold on one synthetic run")
    print("proves nothing about real markets -- see README.md 'Limitations'.")


if __name__ == "__main__":
    main()
