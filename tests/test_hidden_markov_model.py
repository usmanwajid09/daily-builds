"""Tests for hidden_markov_model (deterministic, fixed seeds)."""
import numpy as np
import pytest

from hidden_markov_model import HiddenMarkovModel
from hidden_markov_model.model import HiddenMarkovModel as HMM

# Well-separated 2-state, 2-symbol model used by several tests.
TRUE_START = np.array([0.5, 0.5])
TRUE_TRANS = np.array([[0.9, 0.1],
                       [0.1, 0.9]])
TRUE_EMIT = np.array([[0.95, 0.05],
                      [0.05, 0.95]])


def true_model():
    m = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=0)
    return m.set_params(start_prob=TRUE_START, trans_mat=TRUE_TRANS,
                        emission_mat=TRUE_EMIT)


def make_sequences(n_seq=10, length=100, seed=7):
    rng = np.random.default_rng(seed)
    truth = true_model()
    seqs, states = [], []
    for _ in range(n_seq):
        obs, st = truth.sample(length, random_state=int(rng.integers(1 << 30)))
        seqs.append(obs)
        states.append(st)
    return seqs, states


def test_import_exports_class():
    assert HMM is HiddenMarkovModel


def test_log_likelihood_non_decreasing_across_fit():
    seqs, _ = make_sequences()
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=1)
    model.fit(seqs, n_iter=20)
    hist = model.log_likelihood_history_
    assert len(hist) >= 2
    diffs = np.diff(hist)
    assert np.all(diffs >= -1e-6), f"log-likelihood decreased: {hist}"
    assert hist[-1] > hist[0]


def test_fit_returns_self_and_tracks_history():
    seqs, _ = make_sequences(n_seq=3, length=50)
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=3)
    assert model.fit(seqs, n_iter=5) is model
    assert len(model.log_likelihood_history_) <= 5
    assert all(np.isfinite(v) for v in model.log_likelihood_history_)


def test_viterbi_length_and_valid_states():
    seqs, _ = make_sequences(n_seq=2, length=80)
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=2)
    model.fit(seqs, n_iter=10)
    for seq in seqs:
        decoded = model.predict(seq)
        assert decoded.shape == (len(seq),)
        assert decoded.dtype.kind == "i"
        assert set(np.unique(decoded)) <= {0, 1}


def test_viterbi_recovers_true_states_on_easy_problem():
    seqs, true_states = make_sequences(n_seq=5, length=200, seed=11)
    model = true_model()  # decode with the true parameters
    accuracies = []
    for seq, true in zip(seqs, true_states):
        decoded = model.predict(seq)
        accuracies.append(max(float(np.mean(decoded == true)),
                              float(np.mean((1 - decoded) == true))))
    assert np.mean(accuracies) >= 0.8, accuracies


def test_forward_backward_gamma_rows_sum_to_one():
    seqs, _ = make_sequences(n_seq=2, length=60)
    model = true_model()
    for seq in seqs:
        alpha, beta, gamma, xi = model.forward_backward(seq)
        assert alpha.shape == (len(seq), 2)
        assert beta.shape == (len(seq), 2)
        assert gamma.shape == (len(seq), 2)
        assert xi.shape == (len(seq) - 1, 2, 2)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-8)
        np.testing.assert_allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-8)


def test_sample_lengths_and_valid_values():
    model = true_model()
    obs, states = model.sample(50, random_state=5)
    assert len(obs) == 50 and len(states) == 50
    assert set(np.unique(obs)) <= {0, 1}
    assert set(np.unique(states)) <= {0, 1}
    obs2, _ = model.sample(50, random_state=5)
    np.testing.assert_array_equal(obs, obs2)  # same seed -> same output


def test_learned_matrices_rows_sum_to_one():
    seqs, _ = make_sequences(n_seq=8, length=100)
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=4)
    model.fit(seqs, n_iter=15)
    np.testing.assert_allclose(model.start_prob_.sum(), 1.0, atol=1e-8)
    np.testing.assert_allclose(model.trans_mat_.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(model.emission_mat_.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(model.trans_mat_ >= 0) and np.all(model.emission_mat_ >= 0)


def test_log_likelihood_matches_forward_backward():
    seqs, _ = make_sequences(n_seq=1, length=40)
    model = true_model()
    ll = model.log_likelihood(seqs[0])
    assert np.isfinite(ll) and ll < 0
    # score() of one sequence equals its log-likelihood.
    assert model.score(seqs[0]) == pytest.approx(ll)
    assert model.score(seqs) == pytest.approx(ll)


def test_fit_accepts_single_sequence():
    seq, _ = make_sequences(n_seq=1, length=60)
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=6)
    model.fit(seq[0], n_iter=3)
    assert np.isfinite(model.log_likelihood(seq[0]))


def test_error_out_of_range_symbol():
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=0)
    with pytest.raises(ValueError, match="symbols must be in"):
        model.fit([np.array([0, 1, 2])])
    with pytest.raises(ValueError, match="symbols must be in"):
        model.fit([np.array([0, -1])])


def test_error_empty_sequence():
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=0)
    with pytest.raises(ValueError, match="empty"):
        model.fit([np.array([], dtype=int)])
    with pytest.raises(ValueError, match="empty"):
        model.fit(np.array([], dtype=int))


def test_error_before_fit():
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=0)
    seq = np.array([0, 1, 0, 1])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.log_likelihood(seq)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(seq)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.forward_backward(seq)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.score(seq)


def test_error_bad_n_states_n_symbols():
    with pytest.raises(ValueError):
        HiddenMarkovModel(n_states=0)
    with pytest.raises(ValueError):
        HiddenMarkovModel(n_states=1.5)
    with pytest.raises(ValueError):
        HiddenMarkovModel(n_symbols=0)
    with pytest.raises(ValueError):
        HiddenMarkovModel(n_symbols=-2)


def test_error_bad_n_iter_and_tol():
    seqs, _ = make_sequences(n_seq=2, length=30)
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=0)
    with pytest.raises(ValueError):
        model.fit(seqs, n_iter=0)
    with pytest.raises(ValueError):
        model.fit(seqs, n_iter=2.5)
    with pytest.raises(ValueError):
        model.fit(seqs, tol=0)
    with pytest.raises(ValueError):
        model.fit(seqs, tol=-1e-4)


def test_error_bad_sample_and_set_params():
    model = HiddenMarkovModel(n_states=2, n_symbols=2, random_state=0)
    with pytest.raises(ValueError):
        model.sample(0)
    with pytest.raises(ValueError):
        model.set_params(trans_mat=[[0.5, 0.5]])  # wrong shape
    with pytest.raises(ValueError):
        model.set_params(emission_mat=[[0.5, 0.5], [-0.1, 1.1]])  # negative
