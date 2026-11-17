import numpy as np
import pytest

from nmf.model import NMF


def make_rank2(seed=0, m=30, n=16):
    rng = np.random.default_rng(seed)
    V = rng.uniform(0, 1, size=(m, 2)) @ rng.uniform(0, 1, size=(2, n))
    return V


def rel_err(V, W, H):
    return np.linalg.norm(V - W @ H) / np.linalg.norm(V)


def test_recovers_rank2_factorization():
    V = make_rank2()
    model = NMF(n_components=2, max_iter=1000, random_state=0).fit(V)
    assert rel_err(V, model.W_, model.components_) < 0.05


def test_W_and_H_non_negative():
    V = make_rank2()
    model = NMF(n_components=2, random_state=0).fit(V)
    assert np.all(model.W_ >= 0)
    assert np.all(model.components_ >= 0)


def test_reconstruction_error_decreases_with_iterations():
    V = make_rank2()
    few = NMF(n_components=2, max_iter=5, random_state=0).fit(V)
    many = NMF(n_components=2, max_iter=500, random_state=0).fit(V)
    assert many.reconstruction_err_ < few.reconstruction_err_


def test_final_error_below_initial_error():
    rng = np.random.default_rng(0)
    V = make_rank2()
    model = NMF(n_components=2, max_iter=500, random_state=0)
    # Error of the random starting point (same seed as fit uses).
    W0 = rng.uniform(0.1, 1.0, size=(V.shape[0], 2))
    H0 = rng.uniform(0.1, 1.0, size=(2, V.shape[1]))
    initial_err = np.linalg.norm(V - W0 @ H0)
    model.fit(V)
    assert model.reconstruction_err_ < initial_err


def test_fit_transform_shape():
    V = make_rank2()
    W = NMF(n_components=2, random_state=0).fit_transform(V)
    assert W.shape == (30, 2)


def test_transform_inverse_transform_round_trip_shape():
    V = make_rank2()
    model = NMF(n_components=2, random_state=0).fit(V)
    W_new = model.transform(V)
    assert W_new.shape == (30, 2)
    recon = model.inverse_transform(W_new)
    assert recon.shape == V.shape
    assert rel_err(V, W_new, model.components_) < 0.05


def test_score_explained_variance_near_one():
    V = make_rank2()
    model = NMF(n_components=2, max_iter=1000, random_state=0).fit(V)
    assert model.score(V) > 0.99


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        NMF(n_components=2).transform(make_rank2())


def test_1d_input_raises():
    with pytest.raises(ValueError):
        NMF(n_components=2).fit(np.ones(5))


def test_negative_entries_raise():
    V = make_rank2()
    V[0, 0] = -1.0
    with pytest.raises(ValueError):
        NMF(n_components=2).fit(V)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        NMF(n_components=2).fit(np.empty((0, 5)))


def test_nan_input_raises():
    V = make_rank2()
    V[0, 0] = np.nan
    with pytest.raises(ValueError):
        NMF(n_components=2).fit(V)


def test_n_components_zero_raises():
    with pytest.raises(ValueError):
        NMF(n_components=0).fit(make_rank2())


def test_n_components_too_large_raises():
    # min(shape) = 16 for the 30x16 matrix, so 17 is too many.
    with pytest.raises(ValueError):
        NMF(n_components=17).fit(make_rank2())
