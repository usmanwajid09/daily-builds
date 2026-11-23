import numpy as np
import pytest

from nearest_centroid.model import NearestCentroid


def _blobs(seed=0, n=40):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[0.0, 0.0], scale=0.8, size=(n, 2))
    x1 = rng.normal(loc=[5.0, 4.0], scale=0.8, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_binary_accuracy_high():
    x, y = _blobs()
    model = NearestCentroid().fit(x, y)
    assert model.score(x, y) > 0.95


def test_predicts_nearby_class():
    x, y = _blobs()
    model = NearestCentroid().fit(x, y)
    assert model.predict(np.array([[0.1, 0.1]]))[0] == 0
    assert model.predict(np.array([[5.1, 4.1]]))[0] == 1


def test_multiclass():
    rng = np.random.default_rng(2)
    xs, ys = [], []
    for i, c in enumerate([[0, 0], [6, 0], [3, 6]]):
        xs.append(rng.normal(loc=c, scale=0.8, size=(30, 2)))
        ys.append(np.full(30, i))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    model = NearestCentroid().fit(x, y)
    assert model.centroids_.shape == (3, 2)
    assert model.score(x, y) > 0.95


def test_centroids_are_class_means():
    x = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]])
    y = np.array([0, 0, 1, 1])
    model = NearestCentroid().fit(x, y)
    assert np.allclose(model.centroids_[0], [1.0, 0.0])
    assert np.allclose(model.centroids_[1], [11.0, 10.0])


def test_shrinkage_pulls_toward_overall_mean():
    x = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]])
    y = np.array([0, 0, 1, 1])
    overall = np.array([6.0, 5.0])
    model = NearestCentroid(shrinkage=0.5).fit(x, y)
    assert np.allclose(model.centroids_[0], 0.5 * np.array([1.0, 0.0]) + 0.5 * overall)
    assert np.allclose(model.centroids_[1], 0.5 * np.array([11.0, 10.0]) + 0.5 * overall)


def test_full_shrinkage_collapses_centroids():
    x, y = _blobs()
    model = NearestCentroid(shrinkage=1.0).fit(x, y)
    assert np.allclose(model.centroids_[0], model.centroids_[1])
    assert np.allclose(model.centroids_[0], model.overall_mean_)


def test_shrinkage_still_accurate_when_mild():
    x, y = _blobs()
    model = NearestCentroid(shrinkage=0.3).fit(x, y)
    assert model.score(x, y) > 0.9


def test_manhattan_metric():
    x, y = _blobs()
    model = NearestCentroid(metric="manhattan").fit(x, y)
    assert model.score(x, y) > 0.95
    assert model.predict(np.array([[0.1, 0.1]]))[0] == 0


def test_proba_rows_sum_to_one():
    x, y = _blobs()
    model = NearestCentroid().fit(x, y)
    probs = model.predict_proba(x[:10])
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_proba_nearest_class_highest():
    x, y = _blobs()
    model = NearestCentroid().fit(x, y)
    probs = model.predict_proba(np.array([[0.1, 0.1]]))
    assert probs[0, 0] > probs[0, 1]


def test_fit_returns_self():
    x, y = _blobs()
    model = NearestCentroid()
    assert model.fit(x, y) is model


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        NearestCentroid().predict(np.array([[0.0, 0.0]]))


def test_bad_shrinkage_raises():
    with pytest.raises(ValueError, match="shrinkage"):
        NearestCentroid(shrinkage=-0.1)
    with pytest.raises(ValueError, match="shrinkage"):
        NearestCentroid(shrinkage=1.5)


def test_bad_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        NearestCentroid(metric="cosine")


def test_mismatched_xy_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="samples"):
        NearestCentroid().fit(x, y[:-1])


def test_single_class_raises():
    with pytest.raises(ValueError, match="at least 2 classes"):
        NearestCentroid().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1, 1]))


def test_wrong_feature_count_raises():
    x, y = _blobs()
    model = NearestCentroid().fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.array([[0.0, 0.0, 0.0]]))


def test_nonfinite_X_raises():
    x = np.array([[0.0, np.nan], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        NearestCentroid().fit(x, y)


def test_string_labels():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array(["a", "a", "a", "b", "b", "b"])
    model = NearestCentroid().fit(x, y)
    assert set(model.predict(x)) == {"a", "b"}
