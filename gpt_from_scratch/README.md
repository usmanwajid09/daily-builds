# GPT from scratch (numpy only)

A tiny GPT-style transformer implemented with plain numpy - no PyTorch, no
`transformers` library. Building it to actually see how attention, residual
streams, and the stacked-block architecture fit together mechanically.

**Why numpy only:** the sandbox this runs in can't install PyTorch (the
package is too large to fetch through the network proxy in the time
available, and `download.pytorch.org` is blocked outright). So this arc
implements both the forward pass (this milestone) and, next milestone, a
hand-written backward pass / training loop in numpy - more work, but it
means every gradient is something we actually derived rather than
autograd magic.

## What's here (Milestone 1 of 2)

- `tokenizer.py` - minimal character-level tokenizer (encode/decode over
  the unique characters in a training corpus).
- `layers.py` - `Linear`, `LayerNorm`, and `gelu` building blocks.
- `attention.py` - causal (masked) multi-head self-attention.
- `block.py` - a pre-norm transformer block (`x = x + attn(ln(x))`,
  `x = x + ffn(ln(x))`), same structure GPT-2 uses.
- `model.py` - `TinyGPT`: token + positional embeddings, N stacked
  blocks, final layer norm, linear head to vocab logits.

This milestone is **forward pass only** - the weights are randomly
initialized and never updated. Milestone 2 adds the training loop.

## Usage

```python
from gpt_from_scratch.tokenizer import CharTokenizer
from gpt_from_scratch.model import TinyGPT

tok = CharTokenizer("hello world")
model = TinyGPT(vocab_size=tok.vocab_size, d_model=16, n_heads=4, n_layers=2)

ids = tok.encode("hello")
logits = model([ids])  # (1, 5, vocab_size) - untrained, so logits are noise
```

## Run tests

```bash
python -m pytest gpt_from_scratch/tests/
```
