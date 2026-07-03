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

## Limitations

- Trained on ~1.5KB of text, so it mostly memorizes rather than
  generalizes - a real corpus would need to be orders of magnitude
  larger to show genuine generalization.
- No learning-rate schedule, gradient clipping, or dropout - kept simple
  since the goal was implementing and verifying backprop by hand, not
  chasing state-of-the-art loss curves.
- Batch gradient checking in tests samples a handful of random parameter
  entries per layer rather than every entry (full numerical gradient
  checking of every parameter would be slow); this is standard practice
  and still catches the vast majority of formula errors.
