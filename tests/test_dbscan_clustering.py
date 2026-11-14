import numpy as np
import pytest

from dbscan_clustering.model import DBSCAN


def _two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(50, 2))
    b = rng.normal(loc=[6.0, 0.0], scale=0.3, size=(50, 2))
    return np.vstack([a, b])


def test_recovers_two_well_separated_blobs():
    x = _two_blobs()
    model = DBSCAN(eps=0.5, min_samples=5).fit(x)
    assert model.n_clusters_ == 2
    # Each true blob (50 points) maps to a single cluster.
    assert len(np.unique(model.labels_[:50])) == 1
    assert len(np.unique(model.labels_[50:])) == 1
    assert model.labels_[0] != model.labels_[50]


def test_isolated_far_points_are_noise():
    x = _two_blobs()
    far = np.array([[100.0, 100.0], [-100.0, -100.0], [50.0, -50.0]])
    x = np.vstack([x, far])
    model = DBSCAN(eps=0.5, min_samples=5).fit(x)
    assert np.all(model.labels_[-3:] == -1)
    assert model.n_clusters_ == 2


def test_min_samples_one_two_far_points_give_two_clusters():
    x = np.array([[0.0, 0.0], [10.0, 10.0]])
    model = DBSCAN(eps=0.5, min_samples=1).fit(x)
    assert model.n_clusters_ == 2
    assert set(model.labels_.tolist()) == {0, 1}


def test_density_reachability_chain_is_one_cluster():
    # Points spaced 0.4 apart: endpoints are 3.6 apart (>> eps) but the
    # chain links them into a single cluster.
    x = np.array([[i * 0.4, 0.0] for i in range(10)])
    model = DBSCAN(eps=0.5, min_samples=2).fit(x)
    assert model.n_clusters_ == 1
    assert np.all(model.labels_ == 0)


def test_duplicate_points_cluster_together():
    x = np.array([[0.0, 0.0]] * 6 + [[5.0, 5.0]] * 6)
    model = DBSCAN(eps=0.5, min_samples=5).fit(x)
    assert model.n_clusters_ == 2
    assert len(np.unique(model.labels_[:6])) == 1
    assert len(np.unique(model.labels_[6:])) == 1


def test_single_point_min_samples_one_is_own_cluster():
    model = DBSCAN(eps=0.5, min_samples=1).fit(np.array([[1.0, 2.0]]))
    assert model.n_clusters_ == 1
    assert model.labels_.tolist() == [0]


def test_single_point_min_samples_above_one_is_noise():
    model = DBSCAN(eps=0.5, min_samples=5).fit(np.array([[1.0, 2.0]]))
    assert model.n_clusters_ == 0
    assert model.labels_.tolist() == [-1]


def test_labels_shape_and_dtype():
    x = _two_blobs()
    model = DBSCAN(eps=0.5, min_samples=5).fit(x)
    assert model.labels_.shape == (100,)
    assert model.labels_.dtype == np.int64 or model.labels_.dtype == int
    assert set(model.labels_.tolist()) <= {0, 1}


def test_n_clusters_excludes_noise():
    x = _two_blobs()
    x = np.vstack([x, [[100.0, 100.0]]])
    model = DBSCAN(eps=0.5, min_samples=5).fit(x)
    assert model.n_clusters_ == 2  # the far point is noise, not a cluster
    assert -1 in model.labels_


def test_core_sample_indices_only_core_points():
    # A tight group of 6 identical points: every one is core (6 >= 5
    # neighbours at distance 0). The far point is noise, not core.
    x = np.array([[0.0, 0.0]] * 6 + [[50.0, 50.0]])
    model = DBSCAN(eps=0.5, min_samples=5).fit(x)
    assert set(model.core_sample_indices_.tolist()) == {0, 1, 2, 3, 4, 5}
    assert 6 not in model.core_sample_indices_
    # And a known core point is present, a known noise point is absent.
    assert 0 in model.core_sample_indices_


def test_border_point_joins_cluster_but_is_not_core():
    # 5 core points spread along a 0.2-wide line; the point at 0.6 is
    # within eps of only 3 of them, so it has 4 neighbours (< 5) and is
    # a border point: in the cluster, but not core.
    x = np.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0],
                  [0.15, 0.0], [0.2, 0.0], [0.6, 0.0]])
    model = DBSCAN(eps=0.5, min_samples=5).fit(x)
    assert model.labels_[5] == model.labels_[0]
    assert 5 not in model.core_sample_indices_


def test_fit_predict_equals_labels():
    x = _two_blobs()
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(x)
    assert np.array_equal(labels, model.labels_)


def test_deterministic():
    x = _two_blobs()
    a = DBSCAN(eps=0.5, min_samples=5).fit(x).labels_
    b = DBSCAN(eps=0.5, min_samples=5).fit(x).labels_
    assert np.array_equal(a, b)


def test_rejects_eps_zero_and_negative():
    with pytest.raises(ValueError):
        DBSCAN(eps=0)
    with pytest.raises(ValueError):
        DBSCAN(eps=-0.5)


def test_rejects_min_samples_zero():
    with pytest.raises(ValueError):
        DBSCAN(min_samples=0)


def test_rejects_nan_input():
    x = np.array([[0.0, np.nan], [1.0, 1.0]])
    with pytest.raises(ValueError):
        DBSCAN().fit(x)


def test_rejects_1d_input():
    with pytest.raises(ValueError):
        DBSCAN().fit(np.array([0.0, 1.0, 2.0]))


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        DBSCAN().fit(np.empty((0, 2)))
