# Self-review — ai-trading-bot (Milestone 1: data + indicators + strategy + backtest + metrics)

What I checked, and what I found:

1. **Real market data sources were unusable from this sandbox (documented,
   not "fixed").** Tried `stooq.com`'s free CSV endpoint (returns a
   JavaScript proof-of-work challenge page, not data) and Yahoo Finance's
   chart API (HTTP 429 on every request). Rather than build on a flaky or
   blocked dependency, `data.py` ships a seeded, deterministic synthetic
   OHLCV generator for backtesting/demo, plus a `load_csv()` path for real
   historical data supplied as a file. This is called out explicitly in
   `ai_trading_bot/README.md` so it isn't mistaken for a design preference.

2. **US-format dates would have silently broken `load_csv` (fixed).**
   The first version only parsed ISO `YYYY-MM-DD` dates via
   `date.fromisoformat`. Many real broker/export CSVs use `MM/DD/YYYY`.
   Rather than let that raise a confusing `ValueError` from
   `fromisoformat` (or worse, misparse silently if I'd used a naive
   split), added `_parse_date()` with an explicit ordered list of accepted
   formats (ISO first since it's unambiguous, then `MM/DD/YYYY` /
   `MM/DD/YY`) and a clear error naming the formats tried if none match.
   Added tests for the slash-format path and for a genuinely unparseable
   date.

3. **No-look-ahead correctness in the backtester -- tested explicitly, not
   just assumed.** `run_backtest`'s core invariant is that a signal decided
   using data through bar `i` can only affect the `i -> i+1` return, never
   bar `i`'s own return. Wrote `test_run_backtest_no_lookahead_bar_zero_is_always_flat`
   to pin this down with hand-computed numbers (a contrived case where bar
   0->1 is a +100% move) rather than trusting the docstring. Caught my own
   mistake here: my first draft of that test asserted the +100% move
   should NOT be captured at all, which was wrong -- reacting to bar 0's
   close by holding a position during the bar 0->1 return is exactly
   correct no-look-ahead behavior (the decision was made before that
   return happened). Fixed the test's expectation, not the code, after
   working through the semantics by hand for three separate cases
   (all-flat, no-lookahead, and buy-and-hold).

4. **Transaction costs are charged on every position change, sized by the
   magnitude of the change.** A flip from long to short (magnitude 2) costs
   twice a flip from flat to long (magnitude 1). No strategy in this
   milestone actually goes short (`strategy.py` only emits 0/1), so this
   path is only exercised at magnitude 1 by the current test suite --
   noting this as a known gap for when Milestone 2's ML-based strategy
   might introduce short signals.

5. **RSI and MACD validated against hand-computed values, not just "runs
   without crashing."** `test_rsi_all_gains_is_100` / `test_rsi_all_losses_is_0`
   check the two extremes where the answer is knowable by construction.
   `test_ema_matches_hand_computed_values` manually recomputes 3 EMA steps
   with pen-and-paper arithmetic and compares against the function output.
   `test_macd_zero_on_flat_series` checks that a perfectly flat price
   series produces exactly zero MACD (both EMAs converge to the same
   constant), which would catch a sign error or a fast/slow swap.

6. **Division-by-zero / degenerate-input guards.** `sharpe_ratio` returns
   `0.0` (not `nan`/`inf`) when the excess-return series has zero variance
   (e.g. a strategy that's flat the whole backtest) -- covered by
   `test_sharpe_ratio_zero_variance_returns_zero`. `rsi`'s internal
   `_rsi_from_avgs` returns 100.0 when average loss is 0 rather than
   dividing by zero. `cagr` and `total_return` raise clear `ValueError`s on
   non-positive/empty equity curves instead of producing `nan` silently.

7. **`win_rate` excludes zero-return bars from the denominator on purpose**
   (documented in its docstring) -- a strategy that's flat most of the time
   would otherwise have its win rate diluted by "non-losses" that are
   really just non-trades. Tested explicitly with a mixed array of
   positive/negative/zero returns.

8. **Not built yet, on purpose:** no ML model, no walk-forward
   train/test split, no mock paper-trading loop with a persistent
   position/PnL tracker across "sessions" -- all deferred to Milestone 2
   per the arc plan.

All 53 tests pass (`python -m pytest ai_trading_bot/tests/`).

## Safety rule compliance

Per `ARC_QUEUE.md`'s standing rule for this arc: no live broker connection,
no real order placement, anywhere in this milestone. Verified by inspection
-- the only I/O in the package is `open()` calls in `data.py` for reading/
writing plain local CSV files. No network calls, no API keys, no broker
SDK imports.
