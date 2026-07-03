"""Training loop: char-level language model, trained with the
hand-written numpy backward pass (layers.py/attention.py/block.py/
model.py) and the Adam optimizer in optim.py. No autograd library -
every gradient here was derived and implemented by hand, and checked in
tests/test_backprop.py.

Run as a script for a small end-to-end demo (trains for a few hundred
steps on gpt_from_scratch/data/tiny_corpus.txt, then samples some text):

    python -m gpt_from_scratch.train
"""
from __future__ import annotations

import os

import numpy as np

from gpt_from_scratch.generate import generate
from gpt_from_scratch.loss import cross_entropy_loss
from gpt_from_scratch.model import TinyGPT
from gpt_from_scratch.optim import Adam
from gpt_from_scratch.tokenizer import CharTokenizer

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "tiny_corpus.txt")


def load_corpus(path: str = CORPUS_PATH) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(data_ids: np.ndarray, seq_len: int, batch_size: int, rng: np.random.Generator):
    """Sample `batch_size` random contiguous (input, target) chunks of
    length `seq_len` from `data_ids`, where target is input shifted by 1
    (the standard next-token-prediction setup)."""
    n = len(data_ids)
    if n <= seq_len:
        raise ValueError(
            f"corpus has {n} tokens, too short for seq_len={seq_len} "
            "(need at least seq_len + 1 tokens)"
        )
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    x = np.stack([data_ids[s:s + seq_len] for s in starts])
    y = np.stack([data_ids[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def train(
    steps: int = 500,
    seq_len: int = 32,
    batch_size: int = 16,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    lr: float = 3e-3,
    seed: int = 0,
    log_every: int = 50,
    corpus: str | None = None,
):
    """Train a TinyGPT on `corpus` (or the bundled tiny_corpus.txt if not
    given). Returns (model, tokenizer, losses)."""
    text = corpus if corpus is not None else load_corpus()
    tok = CharTokenizer(text)
    data_ids = np.array(tok.encode(text), dtype=np.int64)

    rng = np.random.default_rng(seed)
    model = TinyGPT(
        vocab_size=tok.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=seq_len,
        seed=seed,
    )
    opt = Adam(lr=lr)

    losses = []
    for step in range(1, steps + 1):
        x, y = get_batch(data_ids, seq_len, batch_size, rng)
        logits = model.forward(x)
        loss, dlogits = cross_entropy_loss(logits, y)
        model.backward(dlogits)
        opt.step(model.parameters_and_grads())
        losses.append(loss)
        if step == 1 or step % log_every == 0:
            print(f"step {step:5d}  loss {loss:.4f}")

    return model, tok, losses


if __name__ == "__main__":
    model, tok, losses = train()
    print(f"\nfinal loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")

    prompt = load_corpus()[0]
    print("\n--- sample generations (same prompt, different sampling settings) ---")
    for temperature, top_k in [(1.0, None), (0.7, 10), (0.0, None)]:
        sample = generate(
            model, tok, prompt=prompt, max_new_tokens=200,
            temperature=temperature, top_k=top_k, seed=0,
        )
        label = "greedy (temperature=0)" if temperature == 0 else f"temperature={temperature}, top_k={top_k}"
        print(f"\n[{label}]\n{sample}")
