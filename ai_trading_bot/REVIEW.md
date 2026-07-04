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

---

# Self-review — ai-trading-bot (Milestone 2: ML direction model + paper trading, arc complete)

What I checked, and what I found:

1. **scikit-learn wasn't reliably installable in the build sandbox
   (documented, not "fixed").** `pip install scikit-learn` timed out /
   left the package missing across two attempts. Rather than block the
   milestone on a flaky install, implemented logistic regression +
   feature standardization from scratch in numpy, consistent with the
   rest of this repo's from-scratch ethos (`linreg_gd`, `gpt_from_scratch`,
   Milestone 1's own indicators). Documented in README with a note on how
   to swap in real scikit-learn if it's available in your own environment
   -- the `fit`/`predict`/`predict_proba`/`transform` interface is
   intentionally close to scikit-learn's shape.

2. **No look-ahead in the ML pipeline -- verified structurally, not just
   asserted.** `features.py` only ever uses data through bar `t` to build
   bar `t`'s features (the label is deliberately about the future, which
   is the whole point of a *prediction* target). `ml_pipeline.split_dataset`
   is chronological only -- `test_train_direction_model_end_to_end_shapes`
   explicitly asserts `train_idx[-1] < test_idx[0]`, i.e. every training
   row comes strictly before every test row in time. The `StandardScaler`
   is fit on training rows only (`train_direction_model` calls
   `StandardScaler().fit(X_train)`, never on the combined or test set) --
   fitting on the full dataset would leak test-period mean/std into an
   evaluation meant to measure out-of-sample generalization.

3. **The demo is honest about a near-chance accuracy, on purpose.**
   `demo_ml.py` prints the majority-class baseline right next to the
   model's accuracy. On the synthetic random-walk series, accuracy comes
   out to ~50% (at or slightly below the majority-class baseline of
   ~53.8%), which is the *correct*, expected result for a model trying to
   predict direction from simple price/volume features on a series with
   no real autocorrelation structure -- not a bug, and not something to
   quietly improve by tuning until the demo "looks good." Called out
   explicitly in README's new "Limitations" section so it isn't mistaken
   for a broken pipeline.

4. **Found and worked through a real, non-obvious discrepancy between the
   two backtest engines -- documented rather than "fixed" away.**
   Running the same out-of-sample signal through `backtest.run_backtest`
   (Milestone 1, vectorized, assumes continuous exposure through each
   bar's return) and `paper_trading.run_mock_paper_trading` (Milestone 2,
   executes an actual buy/sell at the bar's price) produced very different
   total returns in `demo_ml.py` (+11.92% vs -25.40% in one run) for a
   high-turnover (76-trade) signal. Traced this to a genuine modeling
   difference, not an indexing bug: `run_backtest` credits the bar you
   first go long with that bar's full price move (idealized "already
   holding" assumption), while `run_mock_paper_trading` buys in at that
   bar's actual price and therefore misses it (a more realistic
   "can't fill at yesterday's price" assumption). Verified the timing
   itself lines up correctly between the two engines (both start reacting
   to the same first model prediction at the same original bar) before
   concluding the difference was about fill price, not an off-by-one.
   Documented this prominently in both `paper_trading.py`'s docstring and
   README's new "Two backtest engines" section, with a worked-through test
   (`test_run_mock_paper_trading_no_lookahead_bar_zero`) pinning down the
   exact mechanics. Did not try to make the two engines agree -- both are
   legitimate, and the disagreement itself is useful information for
   anyone using this repo (treat `run_backtest` as an optimistic upper
   bound, `run_mock_paper_trading` as the more conservative estimate).

5. **My own first-draft test for the paper-trading no-lookahead behavior
   was wrong, caught by actually tracing the mechanics by hand** (same
   category of mistake as Milestone 1's backtest test). Initially expected
   `run_mock_paper_trading` to behave exactly like `run_backtest` on a
   contrived 100->200 price jump; the actual (correct) behavior is
   different for the reason in point 4. Fixed the test's expectation and
   added an explanatory comment rather than changing the code to match a
   wrong assumption.

6. **Divergence guard in `LogisticRegressionGD.fit` needed a genuinely
   extreme test case to actually trigger.** Unlike `linreg_gd`'s linear
   regression (where the gradient scales with an unbounded residual and
   can blow up exponentially from a merely-too-large learning rate),
   logistic regression's gradient is bounded by the sigmoid saturating the
   error term to [-1, 1] -- so a "normal" bad learning rate just gives bad
   (but finite) weights, not `nan`/`inf`. Had to use a combined extreme
   feature scale (1e200) and learning rate (1e200) to actually force a
   float64 overflow within a few iterations and confirm the guard clause
   fires. Documented why in the test itself so this doesn't look like
   arbitrary magic numbers.

7. **Precision/recall edge case: no predicted-positive days.** If the
   model never predicts "up" on the test set, precision's denominator
   (`tp + fp`) is 0 -- returns `0.0` rather than `nan`, matching the same
   "safe default over a crashing/NaN report" pattern used in Milestone 1's
   `sharpe_ratio`. Covered by
   `test_classification_metrics_no_predicted_positives_precision_zero`.

8. **`signal_from_predictions`' zero-padding before the test period is
   documented as padding, not a real decision** -- its docstring explicitly
   warns callers to slice the resulting equity curve/returns from
   `test_idx[0]` onward when reporting performance, so the flat training
   period doesn't get counted as part of the strategy's track record.
   `demo_ml.py` does this slicing (`bt.equity_curve[test_start:]`).

9. **Not built, on purpose (arc scope ends here):** no ensembling, no
   feature selection/importance analysis, no cross-validation beyond one
   chronological split, no walk-forward *retraining* (the model is trained
   once on the first 70% and evaluated once on the last 30%, rather than
   periodically retrained as more data arrives). Documented as limitations
   rather than silently left out.

All 90 tests pass (`python -m pytest ai_trading_bot/tests/`), covering both
milestones.

## Safety rule compliance (arc-wide, both milestones)

Per `ARC_QUEUE.md`'s standing rule: no live broker connection, no real
order placement, anywhere in this arc. Verified by inspection across both
milestones -- the only I/O anywhere in `ai_trading_bot/` is `open()` calls
in `data.py` for local CSV files. No network calls, no broker SDK imports,
no API keys, no real account credentials, in either milestone. `PaperAccount`
in `paper_trading.py` is purely an in-memory dataclass with no external
side effects.

This is the final milestone for the `ai-trading-bot` arc (2/2 per
`ARC_QUEUE.md`) -- next up per the queue is `saas-starter` (Stripe
TEST-mode only, no real payments).
