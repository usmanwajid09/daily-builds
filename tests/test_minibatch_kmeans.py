"""Tests for MiniBatch K-Means."""

import numpy as np
import pytest

from minibatch_kmeans.model import MiniBatchKMeans


def make_blobs(seed=0, n=60):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.5]])
    x = np.vstack([rng.normal(loc=c, scale=0.8, size=(n, 2)) for c in centres])
    y = np.array([0] * n + [1] * n + [2] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def purity(labels, y_true, n_clusters):
    correct = 0
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            continue
        _, counts = np.unique(y_true[mask], return_counts=True)
        correct += counts.max()
    return correct / len(y_true)


def test_fit_finds_three_clusters():
    x, y = make_blobs()
    model = MiniBatchKMeans(n_clusters=3, batch_size=30, max_iter=100,
                            random_state=0).fit(x)
    assert model.cluster_centers_.shape == (3, 2)
    assert purity(model.labels_, y, 3) >= 0.90


def test_inertia_decreases_from_initialisation():
    x, _ = make_blobs()
    model = MiniBatchKMeans(n_clusters=3, batch_size=30, max_iter=100,
                            random_state=0)
    # Inertia of a freshly initialised model (before any updates) vs fitted.
    rng = np.random.default_rng(0)
    model.cluster_centers_ = model._init_centers(x, rng)
    before = model.inertia(x)
    model.fit(x)
    assert model.inertia_ < before


def test_predict_matches_labels():
    x, _ = make_blobs()
    model = MiniBatchKMeans(n_clusters=3, batch_size=30, max_iter=100,
                            random_state=0).fit(x)
    assert np.array_equal(model.predict(x), model.labels_)


def test_predict_new_points_near_centres():
    x, _ = make_blobs()
    model = MiniBatchKMeans(n_clusters=3, batch_size=30, max_iter=100,
                            random_state=0).fit(x)
    # Points near the true blob centres should land in distinct clusters.
    probes = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.5]])
    labels = model.predict(probes)
    assert len(np.unique(labels)) == 3


def test_transform_returns_distances():
    x, _ = make_blobs()
    model = MiniBatchKMeans(n_clusters=3, batch_size=30, max_iter=100,
                            random_state=0).fit(x)
    dists = model.transform(x)
    assert dists.shape == (len(x), 3)
    assert (dists >= 0).all()
    # Nearest centre distance must match the predicted label.
    assert np.array_equal(np.argmin(dists, axis=1), model.labels_)


def test_partial_fit_updates_centres():
    x, _ = make_blobs()
    model = MiniBatchKMeans(n_clusters=3, batch_size=30, max_iter=10,
                            random_state=0).fit(x)
    before = model.cluster_centers_.copy()
    model.partial_fit(x[:60])
    assert not np.allclose(before, model.cluster_centers_)


def test_partial_fit_initialises_from_first_batch():
    x, _ = make_blobs()
    model = MiniBatchKMeans(n_clusters=3, random_state=0)
    model.partial_fit(x)
    assert model.cluster_centers_.shape == (3, 2)
    assert model.predict(x).shape == (len(x),)


def test_random_init_works():
    x, y = make_blobs()
    model = MiniBatchKMeans(n_clusters=3, batch_size=30, max_iter=100,
                            init="random", random_state=0).fit(x)
    assert purity(model.labels_, y, 3) >= 0.85


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        MiniBatchKMeans(n_clusters=3).predict(np.zeros((2, 2)))


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        MiniBatchKMeans(n_clusters=0)
    with pytest.raises(ValueError):
        MiniBatchKMeans(batch_size=0)
    with pytest.raises(ValueError):
        MiniBatchKMeans(init="kmeans")


def test_more_clusters_than_samples_raises():
    with pytest.raises(ValueError):
        MiniBatchKMeans(n_clusters=10).fit(np.zeros((5, 2)))
