"""v2 training script: wires together every upgrade on top of the
milestone-2 pipeline (train.py, kept untouched and still fully working)
into one bigger, more capable model:

  - byte-level BPE tokenizer (bpe_tokenizer.py) instead of char-level
  - a bigger TinyGPT: more layers/heads/d_model, longer context
  - weight tying (tie_weights=True) + residual dropout (dropout_p>0)
  - Adam + gradient clipping + cosine LR schedule with warmup
  - a held-out validation split, with periodic val-loss logging (to
    actually see over/underfitting, not just the train loss)
  - generation at the end using KV-cache decoding (use_cache=True) and
    nucleus (top-p) sampling, composed with top-k and temperature

Run directly:

    python -m gpt_from_scratch.train_v2
"""
from __future__ import annotations

import os
import time

import numpy as np

from gpt_from_scratch.bpe_tokenizer import BPETokenizer
from gpt_from_scratch.generate import generate
from gpt_from_scratch.loss import cross_entropy_loss
from gpt_from_scratch.model import TinyGPT
from gpt_from_scratch.optim import Adam, clip_grad_norm, cosine_lr_with_warmup
from gpt_from_scratch.train import get_batch

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "larger_corpus.txt")


def load_corpus(path: str = CORPUS_PATH) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def train_v2(
    steps: int = 2000,
    seq_len: int = 64,
    batch_size: int = 24,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    vocab_size: int = 512,
    dropout_p: float = 0.1,
    max_lr: float = 4e-3,
    warmup_steps: int = 100,
    grad_clip_norm: float = 1.0,
    val_frac: float = 0.1,
    seed: int = 0,
    log_every: int = 100,
    corpus: str | None = None,
    verbose: bool = True,
):
    """Returns (model, tokenizer, history) where history has 'train_loss'
    (one entry per step) and 'val_loss' (one entry every log_every steps,
    paired with the step number in 'val_steps')."""
    text = corpus if corpus is not None else load_corpus()
    tok = BPETokenizer(text, vocab_size=vocab_size)
    data_ids = np.array(tok.encode(text), dtype=np.int64)

    n_val = max(seq_len + 1, int(len(data_ids) * val_frac))
    train_ids, val_ids = data_ids[:-n_val], data_ids[-n_val:]
    if verbose:
        print(f"corpus: {len(text)} chars -> {len(data_ids)} tokens "
              f"({len(data_ids) / len(text):.2f} tokens/char), "
              f"vocab_size={tok.vocab_size}, {len(train_ids)} train / {len(val_ids)} val tokens")

    rng = np.random.default_rng(seed)
    model = TinyGPT(
        vocab_size=tok.vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_seq_len=seq_len, seed=seed, dropout_p=dropout_p, tie_weights=True,
    )
    n_params = sum(p.size for p, _ in model.parameters_and_grads())
    if verbose:
        print(f"model: d_model={d_model} n_heads={n_heads} n_layers={n_layers} "
              f"seq_len={seq_len} dropout_p={dropout_p} tie_weights=True -> {n_params:,} parameters")

    opt = Adam()
    history = {"train_loss": [], "val_loss": [], "val_steps": []}
    t0 = time.time()

    for step in range(1, steps + 1):
        lr = cosine_lr_with_warmup(step, max_lr=max_lr, warmup_steps=warmup_steps, max_steps=steps)
        opt.lr = lr

        x, y = get_batch(train_ids, seq_len, batch_size, rng)
        logits = model.forward(x, training=True, rng=rng)
        loss, dlogits = cross_entropy_loss(logits, y)
        model.backward(dlogits)
        clip_grad_norm(model.parameters_and_grads(), grad_clip_norm)
        opt.step(model.parameters_and_grads())
        history["train_loss"].append(loss)

        if step == 1 or step % log_every == 0:
            vx, vy = get_batch(val_ids, seq_len, min(batch_size, len(val_ids) - seq_len - 1), rng)
            val_logits = model.forward(vx, training=False)
            val_loss, _ = cross_entropy_loss(val_logits, vy)
            history["val_loss"].append(val_loss)
            history["val_steps"].append(step)
            if verbose:
                elapsed = time.time() - t0
                print(f"step {step:5d}  lr {lr:.5f}  train_loss {loss:.4f}  "
                      f"val_loss {val_loss:.4f}  ({elapsed:.0f}s)")

    return model, tok, history


if __name__ == "__main__":
    model, tok, history = train_v2()
    print(f"\nfinal train loss: {history['train_loss'][-1]:.4f} "
          f"(started at {history['train_loss'][0]:.4f})")
    print(f"final val loss:   {history['val_loss'][-1]:.4f} "
          f"(started at {history['val_loss'][0]:.4f})")

    prompt = load_corpus()[:20]
    print(f"\n--- sample generations (prompt={prompt!r}, KV-cache decoding) ---")
    for label, kwargs in [
        ("greedy", dict(temperature=0)),
        ("temperature=0.8, top_k=40", dict(temperature=0.8, top_k=40)),
        ("temperature=0.9, top_p=0.9", dict(temperature=0.9, top_p=0.9)),
    ]:
        sample = generate(
            model, tok, prompt=prompt, max_new_tokens=180, seed=0, use_cache=True, **kwargs
        )
        print(f"\n[{label}]\n{sample}")
