import numpy as np
import pytest

from gpt_from_scratch.train import get_batch, train


def test_get_batch_shapes_and_shift():
    rng = np.random.default_rng(0)
    data = np.arange(50)
    x, y = get_batch(data, seq_len=5, batch_size=4, rng=rng)
    assert x.shape == (4, 5)
    assert y.shape == (4, 5)
    # y should be x shifted by exactly one position (since both are slices
    # of the same underlying arange, y[i] == x[i] + 1 elementwise)
    assert np.array_equal(y, x + 1)


def test_get_batch_rejects_corpus_shorter_than_seq_len():
    rng = np.random.default_rng(0)
    data = np.arange(4)
    with pytest.raises(ValueError):
        get_batch(data, seq_len=10, batch_size=2, rng=rng)


def test_training_reduces_loss_on_repetitive_corpus():
    """A short, highly repetitive corpus should be easy to mostly-memorize
    within a couple hundred steps - a good, fast smoke test that the full
    forward/backward/optimizer loop actually learns something."""
    text = "the quick fox runs. the quick fox jumps. the quick fox sleeps. " * 6
    model, tok, losses = train(
        steps=200, seq_len=16, batch_size=8, d_model=32, n_heads=2,
        n_layers=2, lr=5e-3, seed=0, log_every=1000, corpus=text,
    )
    assert len(losses) == 200
    assert losses[-1] < losses[0]
    # loss should have come down substantially, not just wobbled
    assert losses[-1] < 1.0
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_training_is_deterministic_with_same_seed():
    text = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc" * 3
    _, _, losses1 = train(steps=20, seq_len=8, batch_size=4, d_model=16,
                           n_heads=2, n_layers=1, lr=1e-2, seed=0,
                           log_every=1000, corpus=text)
    _, _, losses2 = train(steps=20, seq_len=8, batch_size=4, d_model=16,
                           n_heads=2, n_layers=1, lr=1e-2, seed=0,
                           log_every=1000, corpus=text)
    assert np.allclose(losses1, losses2)
