import numpy as np
import pytest

from kmeans_clustering.model import KMeans


def _blobs():
    rng = np.random.default_rng(0)
    n = 30
    blobs = [
        rng.normal(loc=[0.0, 0.0], scale=0.5, size=(n, 2)),
        rng.normal(loc=[8.0, 0.0], scale=0.5, size=(n, 2)),
        rng.normal(loc=[4.0, 7.0], scale=0.5, size=(n, 2)),
    ]
    return np.vstack(blobs)


def test_recovers_three_blobs():
    x = _blobs()
    model = KMeans(n_clusters=3, random_state=0).fit(x)
    # Each true blob (30 points) should map to a single cluster.
    for start in (0, 30, 60):
        labels = model.labels_[start:start + 30]
        assert len(np.unique(labels)) == 1


def test_inertia_matches_manual_calculation():
    x = _blobs()
    model = KMeans(n_clusters=3, random_state=0).fit(x)
    manual = float(np.sum((x - model.cluster_centers_[model.labels_]) ** 2))
    assert model.inertia_ == pytest.approx(manual)


def test_more_clusters_lower_or_equal_inertia():
    x = _blobs()
    small = KMeans(n_clusters=2, random_state=0).fit(x).inertia_
    big = KMeans(n_clusters=4, random_state=0).fit(x).inertia_
    assert big <= small


def test_single_cluster_centre_is_mean():
    x = _blobs()
    model = KMeans(n_clusters=1, random_state=0).fit(x)
    assert np.allclose(model.cluster_centers_[0], x.mean(axis=0))
    assert np.all(model.labels_ == 0)


def test_predict_assigns_to_nearest_centre():
    x = _blobs()
    model = KMeans(n_clusters=3, random_state=0).fit(x)
    preds = model.predict(np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]]))
    assert len(np.unique(preds)) == 3


def test_fit_predict_returns_labels():
    x = _blobs()
    labels = KMeans(n_clusters=3, random_state=0).fit_predict(x)
    assert labels.shape == (90,)
    assert set(labels.tolist()) <= {0, 1, 2}


def test_deterministic_with_random_state():
    x = _blobs()
    a = KMeans(n_clusters=3, random_state=7).fit(x)
    b = KMeans(n_clusters=3, random_state=7).fit(x)
    assert np.array_equal(a.labels_, b.labels_)
    assert a.inertia_ == b.inertia_


def test_rejects_bad_n_clusters():
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    with pytest.raises(ValueError):
        KMeans(n_clusters=-2)


def test_rejects_more_clusters_than_samples():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        KMeans(n_clusters=3).fit(x)


def test_rejects_empty_data():
    with pytest.raises(ValueError):
        KMeans().fit(np.empty((0, 2)))


def test_rejects_nan_values():
    with pytest.raises(ValueError):
        KMeans().fit(np.array([[0.0, np.nan], [1.0, 1.0]]))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        KMeans().predict(np.array([[0.0, 0.0]]))


def test_rejects_wrong_feature_count():
    x = _blobs()
    model = KMeans(n_clusters=3, random_state=0).fit(x)
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0, 0.0]]))
