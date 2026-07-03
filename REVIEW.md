# Self-review — linreg_gd

What I checked, and what I found:

1. **Off-by-one in loss tracking (fixed).** In `LinearRegressionGD.fit`,
   `loss_history_` was appended *before* each parameter update, so
   `loss_history_[-1]` reflected the loss of the second-to-last parameter
   values, not the `w_`/`b_` actually returned by `fit()`. Harmless once
   converged (the last update is tiny), but wrong in principle and
   misleading if someone fits with few iterations or a high learning rate
   and reads `loss_history_[-1]` as "the final model's loss." Fixed by
   computing one more loss value from the final `w_, b_` after the loop.

2. **Input validation.** Checked `fit()` against mismatched-length arrays,
   empty arrays, and non-positive `learning_rate`/`n_iters` — all raise
   `ValueError` with a clear message rather than failing deep inside numpy
   with a cryptic broadcast error. Covered by tests.

3. **Divergence guard.** A learning rate that's too high causes gradient
   descent to blow up to `inf`/`nan`. Rather than let that fail silently
   (predict() would just return `nan` forever), `fit()` raises
   `FloatingPointError` as soon as `w`/`b` stop being finite, with a message
   telling the user what to change.

4. **`predict()` before `fit()`.** Raises `RuntimeError` instead of
   returning garbage from `None * x`. Covered by a test.

5. **Not changed, considered and rejected:** adding an early-stopping
   tolerance (stop when loss stops improving). Would be a reasonable
   feature but is scope creep for what's meant to be a small, readable
   from-scratch implementation — noted here rather than silently added.

All 6 tests pass after the fix; demo still runs and produces a sane fit
(learned w=2.57 vs true w=2.5, on noisy synthetic data with seed=42).
