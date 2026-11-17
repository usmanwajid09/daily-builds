"""Tests for agglomerative_clustering (from scratch, numpy only)."""

import numpy as np
import pytest

from agglomerative_clustering.model import AgglomerativeClustering


def make_blobs(seed=0, n_per_blob=20, scale=0.5):
    """3 well-separated Gaussian blobs; returns (X, y_true)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    X = np.vstack([rng.normal(loc=c, scale=scale, size=(n_per_blob, 2))
                   for c in centers])
    y_true = np.repeat([0, 1, 2], n_per_blob)
    return X, y_true


def assert_blobs_recovered(X, y_true, labels):
    """Each true blob maps to exactly one predicted label."""
    assert sorted(set(labels.tolist())) == [0, 1, 2]
    for blob in range(3):
        blob_labels = set(labels[y_true == blob].tolist())
        assert len(blob_labels) == 1, (
            f"true blob {blob} split across labels {blob_labels}"
        )


def test_recovers_3_blobs_complete_linkage():
    X, y_true = make_blobs(seed=42)
    labels = AgglomerativeClustering(n_clusters=3, linkage="complete").fit_predict(X)
    assert_blobs_recovered(X, y_true, labels)
    assert sorted((labels == k).sum() for k in range(3)) == [20, 20, 20]


def test_works_with_single_linkage():
    X, y_true = make_blobs(seed=42)
    labels = AgglomerativeClustering(n_clusters=3, linkage="single").fit_predict(X)
    assert_blobs_recovered(X, y_true, labels)


def test_works_with_average_linkage():
    X, y_true = make_blobs(seed=42)
    labels = AgglomerativeClustering(n_clusters=3, linkage="average").fit_predict(X)
    assert_blobs_recovered(X, y_true, labels)


def test_n_clusters_1_gives_all_zeros():
    X, _ = make_blobs(seed=1)
    model = AgglomerativeClustering(n_clusters=1).fit(X)
    assert (model.labels_ == 0).all()
    assert model.n_clusters_ == 1
    assert len(model.merges_) == len(X) - 1


def test_n_clusters_equals_n_samples_gives_unique_labels():
    X, _ = make_blobs(seed=1)
    model = AgglomerativeClustering(n_clusters=len(X)).fit(X)
    assert len(set(model.labels_.tolist())) == len(X)
    assert model.merges_ == []
    assert model.children_ == []


def test_fit_predict_matches_fit_labels():
    X, _ = make_blobs(seed=7)
    model = AgglomerativeClustering(n_clusters=3)
    labels = model.fit_predict(X)
    assert np.array_equal(labels, model.labels_)
    assert len(model.merges_) == len(X) - 3
    assert len(model.children_) == len(model.merges_)


def test_1d_input_raises_value_error():
    with pytest.raises(ValueError, match="2D"):
        AgglomerativeClustering().fit(np.array([1.0, 2.0, 3.0]))


def test_n_clusters_zero_raises_value_error():
    X, _ = make_blobs(seed=1)
    with pytest.raises(ValueError, match=">= 1"):
        AgglomerativeClustering(n_clusters=0).fit(X)


def test_n_clusters_greater_than_n_samples_raises_value_error():
    X, _ = make_blobs(seed=1)
    with pytest.raises(ValueError, match="<= n_samples"):
        AgglomerativeClustering(n_clusters=len(X) + 1).fit(X)


def test_bad_linkage_raises_value_error():
    X, _ = make_blobs(seed=1)
    with pytest.raises(ValueError, match="linkage"):
        AgglomerativeClustering(linkage="ward").fit(X)


def test_nan_input_raises_value_error():
    X, _ = make_blobs(seed=1)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        AgglomerativeClustering().fit(X)
