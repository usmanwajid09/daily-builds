import numpy as np
import pytest

from gpt_from_scratch.model import TinyGPT


def test_output_shape_is_batch_seq_vocab():
    model = TinyGPT(vocab_size=13, d_model=16, n_heads=4, n_layers=2, max_seq_len=10)
    tokens = np.array([[0, 1, 2, 3]])
    logits = model(tokens)
    assert logits.shape == (1, 4, 13)


def test_rejects_sequence_longer_than_max_len():
    model = TinyGPT(vocab_size=5, d_model=8, n_heads=2, n_layers=1, max_seq_len=3)
    with pytest.raises(ValueError):
        model(np.array([[0, 1, 2, 3]]))


def test_rejects_token_id_out_of_vocab_range():
    model = TinyGPT(vocab_size=5, d_model=8, n_heads=2, n_layers=1, max_seq_len=10)
    with pytest.raises(ValueError):
        model(np.array([[0, 1, 99]]))


def test_rejects_bad_constructor_args():
    with pytest.raises(ValueError):
        TinyGPT(vocab_size=0)
    with pytest.raises(ValueError):
        TinyGPT(vocab_size=5, d_model=10, n_heads=3)  # not divisible
    with pytest.raises(ValueError):
        TinyGPT(vocab_size=5, n_layers=0)


def test_deterministic_with_same_seed():
    m1 = TinyGPT(vocab_size=10, d_model=16, n_heads=2, n_layers=1, seed=42)
    m2 = TinyGPT(vocab_size=10, d_model=16, n_heads=2, n_layers=1, seed=42)
    tokens = np.array([[1, 2, 3]])
    assert np.allclose(m1(tokens), m2(tokens))


def test_rejects_empty_token_ids():
    model = TinyGPT(vocab_size=5, d_model=8, n_heads=2, n_layers=1)
    with pytest.raises(ValueError):
        model(np.zeros((1, 0), dtype=int))
    with pytest.raises(ValueError):
        model(np.zeros((0, 3), dtype=int))
