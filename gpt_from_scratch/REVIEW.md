# Self-review — gpt-from-scratch (Milestone 1: architecture)

What I checked, and what I found:

1. **Empty-sequence crash (fixed).** `TinyGPT.__call__` validated token id
   *range* with `token_ids.min()`/`.max()`, but on an empty array (batch=0
   or seq_len=0) numpy's `min`/`max` themselves raise
   `ValueError: zero-size array to reduction operation minimum which has
   no identity` - a confusing, implementation-leaking error instead of our
   own clear message. Added an explicit empty check before the range
   check, plus a test for both empty-batch and empty-seq_len.

2. **Causal masking correctness.** Didn't just check output shape - wrote
   a test that perturbs only the *last* token's input and asserts every
   earlier position's output is unchanged. That's the actual property a
   causal mask is supposed to guarantee, and shape-only tests wouldn't
   catch a mask that's transposed or off-by-one.

3. **Numerical stability of masked softmax.** The causal mask sets future
   positions to `-inf` before softmax. Checked this doesn't produce NaNs:
   `np.max` over a row that has at least one finite entry (the diagonal,
   which is never masked) is always finite, so `exp(-inf - finite) = 0`
   safely for the masked entries. No row can be all `-inf` since a token
   can always attend to itself.

4. **Shape/validation coverage.** `d_model % n_heads != 0`,
   `vocab_size <= 0`, `n_layers <= 0`, sequence longer than `max_seq_len`,
   and out-of-vocab token ids all raise clear `ValueError`s and are
   covered by tests, rather than failing deep inside a matmul.

5. **Determinism.** Same seed -> identical output, verified by a test.
   Matters for milestone 2, where we'll want to compare trained vs
   untrained weights starting from the same init.

6. **Not built yet, on purpose:** no backward pass, no training, no
   weight updates. This milestone is architecture + shape/behavior
   correctness only; training is milestone 2.

All 15 tests pass (`python -m pytest gpt_from_scratch/tests/`).

---

# Self-review — gpt-from-scratch (Milestone 2: training loop)

What I checked, and what I found:

1. **Off-by-one bug in `get_batch`'s sampling range (real bug, fixed).**
   The valid start positions for a `seq_len`-length chunk are
   `s in [0, n - seq_len - 1]` inclusive (so `y = data[s+1 : s+seq_len+1]`
   never runs past the end of the corpus). The code used
   `rng.integers(0, n - seq_len - 1, ...)`, but `integers()`'s `high` is
   *exclusive* - so it should have been `n - seq_len`, not `n - seq_len - 1`.
   Two consequences: (a) at the smallest legal corpus length
   (`n == seq_len + 1`, exactly one valid start position, `s=0`), the
   call became `rng.integers(0, 0, ...)`, which numpy rejects with
   `ValueError: high <= 0` - a hard crash on a legitimate boundary input;
   (b) for any larger corpus, the very last valid start position was
   silently never sampled, quietly wasting a slice of training data every
   run. Fixed the bound to `n - seq_len`, verified by hand (traced through
   `n=10, seq_len=3` -> valid range is exactly `0..6`, confirmed the fixed
   code produces all 7 of those and the old code only ever produced
   `0..5`), and added two regression tests: one hitting the exact
   boundary corpus length that used to crash, one confirming the last
   valid start position now actually gets sampled over enough draws.

2. **Gradient correctness verified numerically, not just "runs without
   error."** Manual backprop is exactly the kind of code where a sign
   flip or transposed matrix still produces a plausible-shaped output but
   a silently wrong gradient. `test_backprop.py` checks every
   `backward()` (Linear, LayerNorm, GELU, causal attention, feedforward,
   full transformer block, cross-entropy, and a full end-to-end
   embeddings->blocks->head->loss integration check) against
   central-difference numerical gradients. All pass within `rtol=1e-2`.
   This is the single most valuable check in this milestone - without it
   I'd only have "the loss went down," which doesn't rule out a gradient
   that's correct in aggregate direction but wrong in some components.

3. **Masked-attention gradient at `-inf` positions.** The causal mask
   sets future scores to `-inf` before softmax, so those weights are
   exactly `0.0`, not just very small. Confirmed the backward pass
   doesn't propagate NaN/Inf through those positions: since
   `weights == 0` exactly there, `dscores = weights * (...)` is `0 *
   finite = 0` (no `-inf` ever enters the backward computation directly -
   only the already-computed, already-finite `weights` array does). Also
   added an explicit `np.where(causal_mask, 0.0, dscores)` as a defensive
   belt-and-suspenders in case `dweights` were ever `inf` for some other
   reason (`0 * inf = nan` otherwise) - cheap insurance, and the gradient
   check test would catch it if this ever mattered in practice.

4. **Adam optimizer state is keyed by list position, not object
   identity.** `parameters_and_grads()` returns a fresh list (with fresh
   gradient arrays, since `backward()` reassigns `self.dW` etc. each
   call) every step, but the *parameter* arrays (`self.W`, `self.gamma`,
   ...) are stable objects across the whole training run and the list
   order is deterministic (fixed by the model's static structure). Adam
   relies on this: it indexes its per-parameter momentum/variance state
   by position in that list. Added an explicit guard that raises if the
   parameter count ever changes between `step()` calls, since silently
   reusing mismatched state would corrupt training instead of failing
   loudly.

5. **Ran the actual training demo end-to-end**, not just unit tests:
   `python -m gpt_from_scratch.train` trains 500 steps on the bundled
   corpus in ~30s on CPU, loss goes 4.08 -> 0.27, and greedy/temperature/
   top-k sampling all produce recognizable fragments of the training
   text (expected, given the corpus is only ~1.5KB - flagged as a known
   limitation in the README rather than a bug, since the milestone's goal
   was a working, correct training pipeline, not generalization from a
   toy amount of data).

6. **`generate()` recomputes the full forward pass from scratch on every
   new token** (no KV cache) - correct, but O(n^2) in sequence length.
   Fine for a ~200-token demo; noted as a known limitation rather than
   fixed, since adding KV caching would mean threading cache state through
   every layer's forward pass, which is a bigger change than this
   milestone's scope (training loop + generation, not performance).

7. **Nothing generated got tracked.** `__pycache__/` and `.pytest_cache/`
   are created locally by running the test suite and training demo but
   stay out of `git status --short`/`git ls-files` - the existing
   `.gitignore` from milestone 1 already covers them.

All 36 tests pass (`python -m pytest gpt_from_scratch/tests/`), including
2 new regression tests for the `get_batch` bug above.
