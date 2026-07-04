# AI Trading Bot (backtest/paper only)

**Safety note, read first:** this project never connects to a live broker
and never places a real order. Everything here runs against historical CSV
data you supply or a deterministic synthetic price generator. Nothing in
this repo is financial advice. See "Safety rules" below.

## What's in Milestone 1

- `data.py` -- historical OHLCV ingestion: `load_csv(path)` for real
  daily-bar CSV data (date/open/high/low/close/volume, with common aliases
  like "Adj Close" handled), and `generate_synthetic_ohlcv(...)` for a
  seeded, reproducible random-walk price series used for backtesting/demo
  purposes when you don't have a data file handy.
- `indicators.py` -- SMA, EMA, RSI (Wilder-smoothed), and MACD, all
  implemented from scratch in numpy (no ta-lib/pandas-ta dependency).
- `strategy.py` -- four signal generators built on those indicators: SMA
  crossover (trend-following), MACD crossover (momentum), RSI mean-reversion
  (threshold-based), and a combined trend+momentum filter.
- `backtest.py` -- a small from-scratch backtest engine (`run_backtest`) with
  an explicit no-look-ahead rule (a signal decided at bar `i`'s close only
  affects the bar `i -> i+1` return) and optional transaction costs in basis
  points, plus a `buy_and_hold_baseline` for comparison.
- `metrics.py` -- total return, CAGR, annualized Sharpe ratio, max drawdown,
  and win rate, plus a `summarize()` convenience bundle.
- `demo.py` -- runs all four strategies plus buy-and-hold against a 3-year
  synthetic series and prints a comparison table.

## Usage

```bash
pip install -r requirements.txt
python -m ai_trading_bot.demo
```

Example output (synthetic data, so exact numbers vary only with the `seed`
argument, not between runs with the same seed):

```
strategy                          total_return      cagr    sharpe    max_dd  win_rate  trades
----------------------------------------------------------------------------------------------
buy_and_hold                           -32.72%   -12.39%     -0.41   -44.52%   +49.01%       1
sma_crossover(20,50)                   -28.40%   -10.55%     -0.73   -29.42%   +47.86%      16
macd(12,26,9)                          -32.74%   -12.40%     -0.69   -42.99%   +43.78%      63
rsi_mean_reversion(14,30,70)            +1.92%    +0.64%      0.12   -23.59%   +48.43%       5
combined_trend_momentum                -11.36%    -3.94%     -0.43   -13.99%   +42.20%      28
```

To backtest your own historical data instead of the synthetic generator:

```python
from ai_trading_bot.data import load_csv
from ai_trading_bot.strategy import sma_crossover_signal
from ai_trading_bot.backtest import run_backtest
from ai_trading_bot.metrics import summarize

series = load_csv("my_daily_bars.csv")  # date,open,high,low,close,volume
signal = sma_crossover_signal(series.close, fast=20, slow=50)
result = run_backtest(series.close, signal, initial_capital=10_000, transaction_cost_bps=5)
print(summarize(result.equity_curve, result.returns))
```

## Run tests

```bash
python -m pytest ai_trading_bot/tests/
```

51 tests covering data ingestion (synthetic generation determinism, OHLC
consistency, CSV round-trip and error handling), indicators (SMA/EMA/RSI/MACD
against hand-computed values and edge cases), strategies (signal shape,
trend/downtrend behavior, threshold validation), the backtest engine
(no-look-ahead correctness, transaction costs, buy-and-hold baseline), and
metrics (Sharpe, CAGR, drawdown, win rate against hand-computed values).

## Why synthetic data instead of a live market data API

Real free market-data endpoints tested from this sandbox were unusable:
`stooq.com`'s CSV endpoint returns a JavaScript proof-of-work challenge page
instead of data, and Yahoo Finance's chart API returned HTTP 429 (rate
limited) on every request. Rather than depend on a flaky/blocked external
service, Milestone 1 ships a seeded synthetic OHLCV generator (geometric
random walk with configurable drift/volatility) so the rest of the pipeline
is fully testable and reproducible offline. `load_csv()` is there so real
historical data (e.g. exported from a broker's history page, or a public
dataset downloaded separately) can be dropped in and used the same way.

## Safety rules (do not remove)

- **Never** connect this code to a live broker API or place a real order.
- **Never** add real account credentials, API keys, or order-execution code.
- All "trading" here is a backtest/paper simulation against historical or
  synthetic price data already in memory -- nothing touches real money.
- This is a learning/research project. Nothing produced by `demo.py` or any
  strategy in `strategy.py` is investment advice.

## What Milestone 2 will add

An ML model (logistic regression or random forest) predicting next-day
price direction, backtested the same way, plus a mock paper-trading
simulation loop and a "not financial advice" disclaimer baked into that
module's own README section.
