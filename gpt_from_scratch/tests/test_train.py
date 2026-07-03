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


def test_get_batch_works_at_minimum_valid_corpus_length():
    """Regression test: corpus length == seq_len + 1 is the smallest legal
    input (exactly one valid start position, s=0). This used to crash with
    a numpy "high <= 0" ValueError due to an off-by-one in the sampling
    bound - see REVIEW.md."""
    rng = np.random.default_rng(0)
    data = np.arange(4)  # seq_len=3 -> only s=0 is valid
    x, y = get_batch(data, seq_len=3, batch_size=5, rng=rng)
    assert x.shape == (5, 3)
    assert np.array_equal(x, np.tile([0, 1, 2], (5, 1)))
    assert np.array_equal(y, np.tile([1, 2, 3], (5, 1)))


def test_get_batch_samples_the_last_valid_start_position():
    """The off-by-one also meant the very last valid start position was
    never sampled. With enough draws it must show up now."""
    rng = np.random.default_rng(0)
    n, seq_len = 10, 3
    max_valid_start = n - seq_len - 1  # = 6
    starts_seen = set()
    for _ in range(3000):
        x, _ = get_batch(np.arange(n), seq_len=seq_len, batch_size=1, rng=rng)
        starts_seen.add(int(x[0, 0]))
    assert max_valid_start in starts_seen


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
