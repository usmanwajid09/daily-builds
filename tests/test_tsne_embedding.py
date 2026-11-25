"""Tests for tsne_embedding (t-SNE from scratch, numpy only)."""
import numpy as np
import pytest

from tsne_embedding import TSNE


def make_blobs(n_per=20, seed=0, dim=5):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0] * dim, [8.0] + [0.0] * (dim - 1),
                        [0.0, 8.0] + [0.0] * (dim - 2)])
    X = np.vstack([rng.normal(c, 0.5, size=(n_per, dim)) for c in centers])
    labels = np.repeat(np.arange(3), n_per)
    return X, labels


def intra_inter_ratio(Y, labels):
    sq = np.sum(Y * Y, axis=1, keepdims=True)
    D = np.sqrt(np.maximum(sq + sq.T - 2.0 * (Y @ Y.T), 0.0))
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    return D[same].mean() / D[~same].mean()


def test_embedding_shape():
    X, _ = make_blobs()
    model = TSNE(perplexity=10, n_iter=250, random_state=0)
    Y = model.fit_transform(X)
    assert Y.shape == (60, 2)


def test_clusters_preserved():
    X, labels = make_blobs()
    Y = TSNE(perplexity=10, n_iter=500, random_state=0).fit_transform(X)
    assert intra_inter_ratio(Y, labels) < 1.0


def test_fit_transform_matches_embedding():
    X, _ = make_blobs()
    model = TSNE(perplexity=10, n_iter=250, random_state=0)
    Y = model.fit_transform(X)
    assert np.array_equal(Y, model.embedding_)
    # fit() alone also populates embedding_
    model2 = TSNE(perplexity=10, n_iter=250, random_state=0)
    model2.fit(X)
    assert model2.embedding_.shape == (60, 2)


def test_determinism_same_seed():
    X, _ = make_blobs()
    Y1 = TSNE(perplexity=10, n_iter=250, random_state=7).fit_transform(X)
    Y2 = TSNE(perplexity=10, n_iter=250, random_state=7).fit_transform(X)
    assert np.allclose(Y1, Y2)


def test_different_seed_different_embedding():
    X, _ = make_blobs()
    Y1 = TSNE(perplexity=10, n_iter=250, random_state=1).fit_transform(X)
    Y2 = TSNE(perplexity=10, n_iter=250, random_state=2).fit_transform(X)
    assert not np.allclose(Y1, Y2)


def test_perplexity_too_large_raises():
    X, _ = make_blobs()
    with pytest.raises(ValueError, match="perplexity"):
        TSNE(perplexity=60, n_iter=10).fit(X)
    with pytest.raises(ValueError, match="perplexity"):
        TSNE(perplexity=60, n_iter=10).fit(X)  # n_samples == 60


def test_perplexity_below_one_raises():
    X, _ = make_blobs()
    with pytest.raises(ValueError, match="perplexity"):
        TSNE(perplexity=0.5, n_iter=10).fit(X)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        TSNE(perplexity=5, n_iter=10).fit(np.empty((0, 5)))


def test_1d_input_raises():
    with pytest.raises(ValueError):
        TSNE(perplexity=5, n_iter=10).fit(np.zeros(10))


def test_nan_raises():
    X, _ = make_blobs()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        TSNE(perplexity=10, n_iter=10).fit(X)


def test_inf_raises():
    X, _ = make_blobs()
    X[0, 0] = np.inf
    with pytest.raises(ValueError):
        TSNE(perplexity=10, n_iter=10).fit(X)


def test_n_components_3_works():
    X, _ = make_blobs()
    Y = TSNE(n_components=3, perplexity=10, n_iter=250, random_state=0).fit_transform(X)
    assert Y.shape == (60, 3)


def test_small_n_works():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    Y = TSNE(perplexity=5, n_iter=250, random_state=0).fit_transform(X)
    assert Y.shape == (10, 2)
    assert np.isfinite(Y).all()


def test_embedding_is_finite():
    X, _ = make_blobs()
    Y = TSNE(perplexity=10, n_iter=250, random_state=0).fit_transform(X)
    assert np.isfinite(Y).all()
