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
