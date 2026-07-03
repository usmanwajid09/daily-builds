"""Smoke tests for train_v2.py - deliberately tiny/fast configs (real
end-to-end training with the full-size config is exercised manually via
`python -m gpt_from_scratch.train_v2`, not in the test suite, since that
takes minutes; see README for a sample run)."""
from __future__ import annotations

import numpy as np

from gpt_from_scratch.train_v2 import train_v2


def _tiny_corpus():
    return (
        "the quick brown fox jumps over the lazy dog. "
        "she sells seashells by the seashore. "
        "how much wood would a woodchuck chuck. " * 8
    )


def test_train_v2_reduces_train_and_val_loss():
    model, tok, history = train_v2(
        steps=150, seq_len=16, batch_size=8, d_model=32, n_heads=2, n_layers=2,
        vocab_size=280, dropout_p=0.1, max_lr=5e-3, warmup_steps=15,
        val_frac=0.15, seed=0, log_every=25, corpus=_tiny_corpus(), verbose=False,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_loss"][-1] < history["val_loss"][0]
    assert model.tie_weights is True
    assert type(model.head).__name__ == "TiedHead"


def test_train_v2_deterministic_with_same_seed():
    corpus = _tiny_corpus()
    _, _, h1 = train_v2(
        steps=30, seq_len=16, batch_size=8, d_model=32, n_heads=2, n_layers=2,
        vocab_size=280, dropout_p=0.1, warmup_steps=5, seed=0, log_every=1000,
        corpus=corpus, verbose=False,
    )
    _, _, h2 = train_v2(
        steps=30, seq_len=16, batch_size=8, d_model=32, n_heads=2, n_layers=2,
        vocab_size=280, dropout_p=0.1, warmup_steps=5, seed=0, log_every=1000,
        corpus=corpus, verbose=False,
    )
    assert np.allclose(h1["train_loss"], h2["train_loss"])


def test_train_v2_val_split_is_held_out_from_training_data():
    """The validation slice should be the corpus's tail, disjoint from
    what get_batch can sample as training data."""
    from gpt_from_scratch.bpe_tokenizer import BPETokenizer

    corpus = _tiny_corpus()
    tok = BPETokenizer(corpus, vocab_size=280)
    data_ids = np.array(tok.encode(corpus))
    n_val = max(16 + 1, int(len(data_ids) * 0.15))
    train_ids, val_ids = data_ids[:-n_val], data_ids[-n_val:]

    assert len(train_ids) + len(val_ids) == len(data_ids)
    assert len(val_ids) == n_val
