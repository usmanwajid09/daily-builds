import numpy as np
import pytest

from optics_clustering.model import OPTICS


def _two_blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(50, 2))
    b = rng.normal(loc=[6.0, 0.0], scale=0.3, size=(50, 2))
    return np.vstack([a, b])


def _three_blobs(seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(40, 2))
    b = rng.normal(loc=[6.0, 0.0], scale=0.3, size=(40, 2))
    c = rng.normal(loc=[3.0, 6.0], scale=0.3, size=(40, 2))
    return np.vstack([a, b, c])


def test_recovers_two_well_separated_blobs():
    x = _two_blobs()
    model = OPTICS(min_samples=5).fit(x)
    assert model.n_clusters_ == 2
    # Each true blob (50 points) maps to a single cluster.
    assert len(np.unique(model.labels_[:50])) == 1
    assert len(np.unique(model.labels_[50:])) == 1
    assert model.labels_[0] != model.labels_[50]


def test_three_blobs_give_three_clusters():
    x = _three_blobs()
    model = OPTICS(min_samples=5).fit(x)
    assert model.n_clusters_ == 3
    assert len(np.unique(model.labels_[:40])) == 1
    assert len(np.unique(model.labels_[40:80])) == 1
    assert len(np.unique(model.labels_[80:])) == 1


def test_far_noise_points_labelled_minus_one():
    x = _two_blobs()
    far = np.array([[100.0, 100.0], [-100.0, -100.0], [50.0, -50.0]])
    x = np.vstack([x, far])
    model = OPTICS(min_samples=5).fit(x)
    assert np.all(model.labels_[-3:] == -1)
    assert model.n_clusters_ == 2


def test_demo_data_two_clusters_and_eight_noise():
    # Same data as demo.py: two blobs of 60 plus 8 planted noise.
    rng = np.random.default_rng(42)
    a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(60, 2))
    b = rng.normal(loc=[5.0, 5.0], scale=0.3, size=(60, 2))
    noise = np.array([
        [-3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [8.0, 0.0],
        [0.0, 8.0], [8.0, 8.0], [-2.0, 6.0], [6.0, -2.0],
    ])
    x = np.vstack([a, b, noise])
    model = OPTICS(min_samples=5).fit(x)
    assert model.n_clusters_ == 2
    assert sorted(int(np.sum(model.labels_ == k)) for k in range(2)) == [60, 60]
    assert int(np.sum(model.labels_ == -1)) == 8
    assert np.all(model.labels_[120:] == -1)


def test_ordering_is_a_permutation_of_all_indices():
    x = _two_blobs()
    model = OPTICS(min_samples=5).fit(x)
    assert sorted(model.ordering_.tolist()) == list(range(len(x)))
    assert len(np.unique(model.ordering_)) == len(x)


def test_reachability_length_and_first_entry_inf():
    x = _two_blobs()
    model = OPTICS(min_samples=5).fit(x)
    assert model.reachability_.shape == (len(x),)
    assert np.isinf(model.reachability_[0])
    # Everything after the first point is reached from somewhere.
    assert np.all(np.isfinite(model.reachability_[1:]))


def test_core_distances_non_negative_and_known_value():
    # Points on a line 1 apart: with min_samples=2 the core distance
    # is the distance to the single nearest other point, 1.0.
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    model = OPTICS(min_samples=2).fit(x)
    assert np.all(model.core_distances_ >= 0)
    assert np.allclose(model.core_distances_, 1.0)
    # Dense blob points have small, finite core distances.
    blob = _two_blobs()
    model2 = OPTICS(min_samples=5).fit(blob)
    assert np.all(np.isfinite(model2.core_distances_))
    assert np.all(model2.core_distances_ >= 0)


def test_single_blob_gives_one_cluster():
    rng = np.random.default_rng(7)
    x = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(80, 2))
    model = OPTICS(min_samples=5).fit(x)
    assert model.n_clusters_ == 1
    assert np.all(model.labels_ == 0)


def test_min_samples_validation():
    for bad in (0, -1, -5):
        with pytest.raises(ValueError):
            OPTICS(min_samples=bad)
    for bad in (2.5, "5", None, True):
        with pytest.raises(ValueError):
            OPTICS(min_samples=bad)


def test_xi_validation():
    for bad in (0, 1, 1.5, -0.1):
        with pytest.raises(ValueError):
            OPTICS(xi=bad)
    with pytest.raises(ValueError):
        OPTICS(xi="0.05")


def test_max_eps_validation():
    for bad in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            OPTICS(max_eps=bad)
    with pytest.raises(ValueError):
        OPTICS(max_eps="1.0")


def test_x_validation():
    model = OPTICS(min_samples=5)
    with pytest.raises(ValueError):
        model.fit(np.array([1.0, 2.0, 3.0]))  # 1D
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)))  # empty
    with pytest.raises(ValueError):
        model.fit(np.array([["a", "b"], ["c", "d"]]))  # non-numeric
    with pytest.raises(ValueError):
        model.fit(np.array([[1.0, 2.0]]))  # a single sample
    with pytest.raises(ValueError):
        model.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))  # NaN


def test_fit_returns_self():
    x = _two_blobs()
    model = OPTICS(min_samples=5)
    assert model.fit(x) is model


def test_labels_consistent_with_n_clusters():
    x = _two_blobs()
    model = OPTICS(min_samples=5).fit(x)
    kept = sorted(set(model.labels_.tolist()) - {-1})
    assert kept == list(range(model.n_clusters_))
    assert model.labels_.shape == (len(x),)
    # Every cluster holds at least min_samples points.
    for k in kept:
        assert int(np.sum(model.labels_ == k)) >= 5


def test_reachability_plot_data_matches_attributes():
    x = _two_blobs()
    model = OPTICS(min_samples=5).fit(x)
    ordering, reachability = model.reachability_plot_data()
    assert np.array_equal(ordering, model.ordering_)
    assert np.array_equal(reachability, model.reachability_,
                          equal_nan=True)
    assert ordering.shape == (len(x),)
    # The hierarchy covers the same clusters as labels_.
    assert len(model.cluster_hierarchy_) == model.n_clusters_
    assert sum(h["size"] for h in model.cluster_hierarchy_) == int(
        np.sum(model.labels_ >= 0))


def test_tiny_max_eps_leaves_everything_noise():
    # No point has 5 neighbours within 0.01, so nothing is core and no
    # cluster can form.
    x = _two_blobs()
    model = OPTICS(min_samples=5, max_eps=0.01).fit(x)
    assert model.n_clusters_ == 0
    assert np.all(model.labels_ == -1)
    assert np.all(np.isinf(model.core_distances_))


def test_deterministic():
    x = _two_blobs()
    a = OPTICS(min_samples=5).fit(x).labels_
    b = OPTICS(min_samples=5).fit(x).labels_
    assert np.array_equal(a, b)


def test_duplicate_points_cluster_together():
    x = np.array([[0.0, 0.0]] * 6 + [[5.0, 5.0]] * 6)
    model = OPTICS(min_samples=5).fit(x)
    assert model.n_clusters_ == 2
    assert len(np.unique(model.labels_[:6])) == 1
    assert len(np.unique(model.labels_[6:])) == 1
    assert np.all(model.core_distances_[:6] == 0.0)
