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

## What's in Milestone 2

- `features.py` -- turns close/volume into a small feature matrix (1-day
  and 5-day returns, SMA-20 ratio, RSI-14, normalized MACD histogram,
  volume change), all computed using only data through bar `t` (no
  future leakage), plus a next-day-direction label. `build_dataset()`
  drops NaN warm-up rows and returns the original bar indices alongside
  the clean matrix so predictions can be mapped back onto the price
  series.
- `ml_model.py` -- logistic regression trained with batch gradient
  descent, implemented from scratch in numpy (no scikit-learn -- see
  "Why no scikit-learn" below), plus a `StandardScaler` fit on training
  data only.
- `ml_pipeline.py` -- a **chronological** (walk-forward, never shuffled)
  train/test split, `train_direction_model()` which fits the scaler+model
  on the training rows and evaluates on held-out future rows, classification
  metrics (accuracy/precision/recall/confusion matrix), and
  `signal_from_predictions()` which turns out-of-sample predictions into a
  signal array `backtest.run_backtest()` can consume directly.
- `paper_trading.py` -- a mock, in-memory `PaperAccount` and a
  step-by-step `run_mock_paper_trading()` loop that walks bar-by-bar
  through the test period, executing simulated BUY/SELL orders and
  logging them, structurally closer to what a real live-trading main loop
  looks like than the vectorized backtest engine (see "Two backtest
  engines" below for an important difference between them).
- `demo_ml.py` -- runs the whole pipeline end to end on synthetic data:
  trains the model, prints out-of-sample classification metrics, backtests
  the predictions with `backtest.py`, and separately runs them through
  `paper_trading.py`.

### Usage

```bash
python -m ai_trading_bot.demo_ml
```

### Why no scikit-learn

Installing `scikit-learn` in the build sandbox didn't reliably complete
(timed out / not present after install attempts) -- similar to the
`torch` situation noted for the `gpt-from-scratch` arc. Rather than
depend on a flaky install, Milestone 2 implements logistic regression
from scratch in numpy, consistent with the rest of this repo
(`linreg_gd`, `gpt_from_scratch`, and Milestone 1's own indicators). If
you have scikit-learn available in your own environment, `ml_model.py`'s
`LogisticRegressionGD`/`StandardScaler` interface (`fit`/`predict`/
`predict_proba`/`transform`) is intentionally close to scikit-learn's, so
swapping in `sklearn.linear_model.LogisticRegression` and
`sklearn.preprocessing.StandardScaler` for the pipeline in `ml_pipeline.py`
is a small, mostly mechanical change if you'd rather use a battle-tested
library for real use.

### Two backtest engines -- and why they can disagree

This repo now has two ways to evaluate a signal against price history,
and **they are not directly comparable for a strategy's entry bars**:

- `backtest.run_backtest()` (Milestone 1) is vectorized and idealized: it
  applies each bar's % return directly to existing equity, as if the
  position had already been held continuously through that move.
- `paper_trading.run_mock_paper_trading()` (Milestone 2) actually executes
  a BUY/SELL at the bar's price when the position changes -- closer to
  how a real order would fill. This means the bar you *first* go long
  does not capture that bar's price move (you're buying in at the new
  price, not retroactively at the prior close); only bars where you're
  already long benefit from further moves.

Running the same model-derived signal through both engines in
`demo_ml.py` can produce noticeably different total returns for a
high-turnover strategy -- this was observed during self-review (see
`REVIEW.md`) and is a genuine modeling difference, not a bug in either
engine. Treat `run_backtest`'s numbers as an optimistic upper bound and
`run_mock_paper_trading`'s numbers as the more conservative, fill-aware
estimate.

## Limitations (read before drawing any conclusion from demo_ml.py's numbers)

- The demo runs against **synthetic, seeded random-walk data**. A
  geometric random walk has no genuine autocorrelation structure for a
  model to learn from bar-to-bar price/volume features alone -- so
  near-50% out-of-sample accuracy (at or below a majority-class baseline)
  is the *expected*, honest result, not a bug. `demo_ml.py` prints the
  majority-class baseline alongside the model's accuracy specifically so
  this is visible rather than glossed over.
- A single logistic regression over six simple features is a toy model.
  It is not tuned, not cross-validated beyond one train/test split, and
  makes no attempt at feature selection, ensembling, or hyperparameter
  search.
- One synthetic backtest run "beating" buy-and-hold (or not) proves
  nothing about real markets. Real price series have regime changes,
  transaction costs beyond a flat bps estimate, liquidity constraints, and
  survivorship effects that none of this code models.
- **Nothing in this package is financial advice, and nothing in this
  package should be connected to a live broker or used to place a real
  order.** See "Safety rules" below.

## Safety rules (do not remove)

- **Never** connect this code to a live broker API or place a real order.
- **Never** add real account credentials, API keys, or order-execution code.
- All "trading" here is a backtest/paper simulation against historical or
  synthetic price data already in memory -- nothing touches real money.
- This is a learning/research project. Nothing produced by `demo.py`,
  `demo_ml.py`, any strategy in `strategy.py`, nor any prediction from
  `ml_model.py`/`ml_pipeline.py` is investment advice.
