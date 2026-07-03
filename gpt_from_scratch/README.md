# GPT from scratch (numpy only)

A tiny GPT-style transformer implemented with plain numpy - no PyTorch, no
`transformers` library. Building it to actually see how attention, residual
streams, backprop, and the training loop fit together mechanically.

**Why numpy only:** the sandbox this runs in can't install PyTorch (the
package is too large to fetch through the network proxy in the time
available, and `download.pytorch.org` is blocked outright). So this arc
implements both the forward pass (milestone 1) and a hand-written backward
pass / training loop (milestone 2) in numpy - more work, but it means
every gradient is something we actually derived rather than autograd
magic.

## What's here

**Milestone 1 - architecture (forward pass only):**
- `tokenizer.py` - minimal character-level tokenizer (encode/decode over
  the unique characters in a training corpus).
- `layers.py` - `Linear`, `LayerNorm`, and `gelu` building blocks.
- `attention.py` - causal (masked) multi-head self-attention.
- `block.py` - a pre-norm transformer block (`x = x + attn(ln(x))`,
  `x = x + ffn(ln(x))`), same structure GPT-2 uses.
- `model.py` - `TinyGPT`: token + positional embeddings, N stacked
  blocks, final layer norm, linear head to vocab logits.

**Milestone 2 - training (backward pass + optimizer + generation):**
- Every layer above (`Linear`, `LayerNorm`, `gelu`, `CausalSelfAttention`,
  `FeedForward`, `TransformerBlock`, `TinyGPT`) now has a `backward()`
  method alongside `forward()`, hand-derived with the chain rule - no
  autograd. `parameters_and_grads()` on each exposes `(param, grad)`
  pairs for the optimizer.
- `loss.py` - numerically-stable softmax cross-entropy with a combined
  forward+backward (`dL/dlogits = (softmax(logits) - one_hot(target)) / N`).
- `optim.py` - `Adam` optimizer operating on the `parameters_and_grads()`
  protocol.
- `train.py` - training loop: samples random `(input, target)` chunks
  from a corpus (next-character prediction), runs forward -> loss ->
  backward -> optimizer step, repeat.
- `generate.py` - autoregressive sampling with temperature scaling and
  top-k filtering (temperature=0 is greedy/deterministic decoding).
- `data/tiny_corpus.txt` - a short original story used as the default
  training corpus for the `train.py` demo.
- `tests/test_backprop.py` - **gradient checking**: every `backward()` is
  verified against a central-difference numerical gradient of a scalar
  loss (including a full embeddings -> blocks -> head -> loss integration
  check). This is what actually gives confidence the hand-written
  backprop is correct, as opposed to just "it runs without crashing."

## Usage

Forward pass only (milestone 1 style):

```python
from gpt_from_scratch.tokenizer import CharTokenizer
from gpt_from_scratch.model import TinyGPT

tok = CharTokenizer("hello world")
model = TinyGPT(vocab_size=tok.vocab_size, d_model=16, n_heads=4, n_layers=2)

ids = tok.encode("hello")
logits = model([ids])  # (1, 5, vocab_size)
```

Training + generation (milestone 2):

```python
from gpt_from_scratch.train import train
from gpt_from_scratch.generate import generate

model, tok, losses = train(steps=500, seq_len=32, d_model=64, n_layers=3)
print("final loss:", losses[-1])

text = generate(model, tok, prompt="O", max_new_tokens=200, temperature=0.7, top_k=10, seed=0)
print(text)
```

Or run the bundled end-to-end demo directly (trains ~500 steps on
`data/tiny_corpus.txt`, prints the loss curve, then samples text at three
different temperature/top-k settings, ~30s on a laptop CPU):

```bash
python -m gpt_from_scratch.train
```

Sample output after 500 steps (loss 4.08 -> 0.27 on this small corpus -
mostly memorization given how little text it has to learn from, but it
demonstrates the full pipeline working: attention patterns, punctuation,
paragraph breaks, and word boundaries are all learned, not hardcoded):

```
[greedy (temperature=0)]
One w."


Tin did not speak, but the little brass heart ticked faster, and the
girl smiled the way the tinkerer used the workshop was quiet and cold.
```

## Run tests

```bash
python -m pytest gpt_from_scratch/tests/
```

## v2 upgrade: BPE, bigger model, dropout/tying/clipping/LR schedule, KV-cache, top-p

Everything below was added on top of milestone 2 without touching any of
the milestone-1/2 files' *behavior* - every new capability is either a
new file or an opt-in constructor/function argument that defaults to the
old behavior, so all of the original tests still pass unmodified.

- `bpe_tokenizer.py` - `BPETokenizer`: byte-level BPE (the same family of
  algorithm GPT-2 uses), trained from scratch with the classic
  "repeatedly merge the most frequent adjacent pair" algorithm. Unlike
  `CharTokenizer`, it operates on raw UTF-8 bytes, so `encode()` can
  never hit an unknown-character error - any string in any language
  round-trips losslessly, even text the tokenizer never trained on.
- `data/larger_corpus.txt` - a ~12KB original short story (multiple
  chapters, ~8x bigger than `tiny_corpus.txt`), used as `train_v2.py`'s
  default corpus.
- `layers.py` additions:
  - `Dropout` - inverted dropout (`forward(x, training, rng)` /
    `backward(dout)`), used as residual dropout inside each block.
  - `TiedHead` - output projection sharing the token embedding matrix
    (transposed) instead of a separate weight matrix (Press & Wolf,
    2017). `TinyGPT.backward` sums its gradient contribution into
    `d_token_emb` alongside the embedding-lookup contribution.
- `block.py` / `model.py` - `TransformerBlock` and `TinyGPT` both take
  optional `dropout_p` (residual dropout, default `0.0`) and `TinyGPT`
  takes `tie_weights` (default `False`) constructor args. Both default
  to their old milestone-2 behavior exactly.
- `optim.py` additions:
  - `clip_grad_norm(params_and_grads, max_norm)` - global-norm gradient
    clipping.
  - `cosine_lr_with_warmup(step, max_lr, warmup_steps, max_steps,
    min_lr_ratio)` - linear warmup then cosine decay, the standard
    GPT-family LR schedule.
- **KV-cache generation** - `CausalSelfAttention.forward_incremental` and
  `TinyGPT.forward_incremental` process only the *new* tokens each step,
  reusing cached K/V for everything already seen, instead of recomputing
  attention over the whole sequence every time. Verified in
  `tests/test_kv_cache.py` to produce logits identical (to float
  precision) to the full-recompute path - this is the property that
  actually matters, since a subtly-wrong cache would still produce
  plausible-looking but silently-incorrect text. `generate(...,
  use_cache=True)` uses this path; in a quick check, 50 cached tokens
  took ~0.06s vs. ~0.35s for the same 50 via full recompute at this
  model size - the gap grows with sequence length since full recompute
  is O(n^2) and the cached path is O(n).
- **Nucleus (top-p) sampling** - `generate()` gained a `top_p` parameter,
  composable with `top_k` (top-k filters first, then top-p filters
  what's left of the distribution).
- `train_v2.py` - wires all of the above together: BPE tokenizer, a
  bigger `TinyGPT` (128-dim, 8 heads, 4 layers, 64-token context -
  ~868K parameters vs. milestone 2's ~50K), weight tying, dropout,
  gradient-clipped Adam with the cosine+warmup schedule, and a held-out
  validation split with periodic val-loss logging. Run with:

  ```bash
  python -m gpt_from_scratch.train_v2
  ```

  Defaults to 2000 steps; expect several minutes on a laptop CPU (numpy,
  no GPU). Reduce `steps`/`d_model`/`n_layers` for a quicker run.

### A real result, and an honest one

Training this config on the ~12KB corpus for 275 steps (a partial run,
sandboxed for time) showed exactly the failure mode you'd expect from an
~868K-parameter model against ~4.5K training tokens: train loss kept
falling (6.28 -> 1.33) while val loss, after an initial drop, started
climbing back up (6.24 -> 4.85 around step 150, then up to 5.76 by step
275) - textbook overfitting. This is precisely why `train_v2.py` tracks
val loss separately rather than only reporting train loss like `train.py`
does: without it, the rising val loss would be invisible and the falling
train loss alone would look like unambiguous progress.

**Takeaway, and what "fixing" this would actually require:** the
architecture, dropout, weight tying, and gradient clipping are all
correct and doing their job - the bottleneck here is data, not code. A
much bigger corpus (hundreds of KB to MB, not 12KB) is the real fix;
turning dropout up further or cutting model size would only mask the
symptom. This is left as-is deliberately, as an honest demonstration of
val-loss tracking catching a real problem, rather than cherry-picking a
step count that looks good.

## Limitations

- `tiny_corpus.txt` (milestone 2, ~1.5KB) and `larger_corpus.txt` (v2,
  ~12KB) are both far smaller than what a from-scratch language model
  needs to generalize rather than memorize - see the overfitting result
  above. Orders of magnitude more text would be the real fix.
- KV-cache generation (`use_cache=True`) doesn't support generating past
  `max_seq_len` total tokens (prompt + generated) - it raises a clear
  error instead. The non-cached path (`use_cache=False`) instead slides
  its context window forward, trading unbounded length for losing the
  earliest context and O(n^2) recompute cost.
- Batch gradient checking in tests samples a handful of random parameter
  entries per layer rather than every entry (full numerical gradient
  checking of every parameter would be slow); this is standard practice
  and still catches the vast majority of formula errors.
- No multi-query/grouped-query attention, no rotary/ALiBi position
  encodings, no mixed precision - this is still a small, from-scratch
  teaching implementation, not a production training stack.
