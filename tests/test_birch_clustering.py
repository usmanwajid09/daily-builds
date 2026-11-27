"""Tests for birch_clustering (BIRCH from scratch, numpy only)."""

import numpy as np
import pytest

from birch_clustering.model import BIRCH


def make_blobs(seed=42, n_per_blob=30, scale=0.5):
    """3 well-separated Gaussian blobs; returns (X, y_true)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    X = np.vstack([rng.normal(loc=c, scale=scale, size=(n_per_blob, 2))
                   for c in centers])
    y_true = np.repeat([0, 1, 2], n_per_blob)
    return X, y_true


def test_recovers_three_separated_blobs():
    X, y_true = make_blobs(seed=42)
    model = BIRCH(n_clusters=3)
    labels = model.fit_predict(X)
    assert sorted(set(labels.tolist())) == [0, 1, 2]
    for blob in range(3):
        blob_labels = set(labels[y_true == blob].tolist())
        assert len(blob_labels) == 1, (
            f"true blob {blob} split across labels {blob_labels}"
        )
    assert sorted((labels == k).sum() for k in range(3)) == [30, 30, 30]


def test_cluster_centers_shape():
    X, _ = make_blobs(seed=1)
    model = BIRCH(n_clusters=3).fit(X)
    assert model.cluster_centers_.shape == (3, 2)
    assert model.n_clusters_ == 3


def test_labels_length_and_range():
    X, _ = make_blobs(seed=2)
    model = BIRCH(n_clusters=3).fit(X)
    assert len(model.labels_) == len(X)
    assert set(model.labels_.tolist()) <= {0, 1, 2}
    assert model.labels_.min() >= 0


def test_n_clusters_none_gives_leaf_subclusters_as_clusters():
    X, _ = make_blobs(seed=3)
    model = BIRCH(n_clusters=None).fit(X)
    assert model.n_clusters_ == model.n_subclusters_
    assert model.cluster_centers_.shape == (model.n_subclusters_, 2)
    assert len(set(model.labels_.tolist())) == model.n_subclusters_


def test_threshold_zero_raises_value_error():
    X, _ = make_blobs(seed=1)
    with pytest.raises(ValueError, match="threshold"):
        BIRCH(threshold=0).fit(X)


def test_threshold_negative_raises_value_error():
    X, _ = make_blobs(seed=1)
    with pytest.raises(ValueError, match="threshold"):
        BIRCH(threshold=-0.5).fit(X)


def test_branching_factor_below_two_raises_value_error():
    X, _ = make_blobs(seed=1)
    with pytest.raises(ValueError, match="branching_factor"):
        BIRCH(branching_factor=1).fit(X)


def test_n_clusters_zero_raises_value_error():
    X, _ = make_blobs(seed=1)
    with pytest.raises(ValueError, match=">= 1"):
        BIRCH(n_clusters=0).fit(X)


def test_1d_input_raises_value_error():
    with pytest.raises(ValueError, match="2D"):
        BIRCH().fit(np.array([1.0, 2.0, 3.0]))


def test_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        BIRCH().fit(np.empty((0, 2)))


def test_fit_returns_self():
    X, _ = make_blobs(seed=4)
    model = BIRCH(n_clusters=3)
    assert model.fit(X) is model


def test_fit_predict_matches_labels():
    X, _ = make_blobs(seed=5)
    model = BIRCH(n_clusters=3)
    labels = model.fit_predict(X)
    assert np.array_equal(labels, model.labels_)


def test_small_threshold_creates_more_subclusters():
    X, _ = make_blobs(seed=6)
    small = BIRCH(threshold=0.1, n_clusters=None).fit(X)
    large = BIRCH(threshold=50.0, n_clusters=None).fit(X)
    assert small.n_subclusters_ > large.n_subclusters_
    assert large.n_subclusters_ == 1  # huge threshold: everything in one entry


def test_n_clusters_larger_than_subclusters_uses_all():
    X, _ = make_blobs(seed=7)
    model = BIRCH(threshold=5.0, n_clusters=10).fit(X)
    assert model.n_clusters_ == model.n_subclusters_
    assert model.cluster_centers_.shape[0] == model.n_subclusters_


def test_small_branching_factor_still_works():
    X, y_true = make_blobs(seed=8)
    labels = BIRCH(n_clusters=3, branching_factor=2).fit_predict(X)
    for blob in range(3):
        assert len(set(labels[y_true == blob].tolist())) == 1
