import numpy as np
import pytest

from gaussian_mixture.model import GaussianMixture


def _blobs(seed=42, n=150):
    rng = np.random.default_rng(seed)
    x = np.vstack([
        rng.normal(loc=[0.0, 0.0], scale=0.8, size=(n, 2)),
        rng.normal(loc=[5.0, 4.0], scale=0.8, size=(n, 2)),
    ])
    y = np.array([0] * n + [1] * n)
    return x, y


def test_recovers_blob_means():
    x, _y = _blobs()
    model = GaussianMixture(n_components=2, random_state=42).fit(x)
    means = sorted(model.means_.tolist())
    assert np.allclose(means[0], [0.0, 0.0], atol=0.5)
    assert np.allclose(means[1], [5.0, 4.0], atol=0.5)


def test_clustering_accuracy():
    x, y = _blobs()
    model = GaussianMixture(n_components=2, random_state=42).fit(x)
    preds = model.predict(x)
    acc = max(float(np.mean(preds == y)), 1.0 - float(np.mean(preds == y)))
    assert acc >= 0.95


def test_predict_proba_sums_to_one():
    x, _y = _blobs()
    model = GaussianMixture(n_components=2, random_state=0).fit(x)
    proba = model.predict_proba(x)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_weights_sum_to_one():
    x, _y = _blobs()
    model = GaussianMixture(n_components=2, random_state=0).fit(x)
    assert np.isclose(model.weights_.sum(), 1.0)
    assert np.allclose(model.weights_, [0.5, 0.5], atol=0.1)


def test_log_likelihood_is_finite_and_beats_one_component():
    x, _y = _blobs()
    model = GaussianMixture(n_components=2, random_state=0).fit(x)
    single = GaussianMixture(n_components=1, random_state=0).fit(x)
    assert np.isfinite(model.log_likelihood_)
    assert np.isfinite(model.score(x))
    # Two components fit two blobs better than one component can.
    assert model.log_likelihood_ > single.log_likelihood_


def test_confident_assignments_on_separated_blobs():
    x, _y = _blobs()
    model = GaussianMixture(n_components=2, random_state=0).fit(x)
    proba = model.predict_proba(x)
    assert float(proba.max(axis=1).mean()) >= 0.95


def test_single_component():
    x, _y = _blobs()
    model = GaussianMixture(n_components=1, random_state=0).fit(x)
    assert np.all(model.predict(x) == 0)
    assert np.allclose(model.means_[0], x.mean(axis=0), atol=0.5)


def test_three_components():
    x, _y = _blobs()
    model = GaussianMixture(n_components=3, random_state=0).fit(x)
    assert model.means_.shape == (3, 2)
    assert np.isclose(model.weights_.sum(), 1.0)


def test_one_dimensional_data():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0, 1, 100), rng.normal(6, 1, 100)]).reshape(-1, 1)
    model = GaussianMixture(n_components=2, random_state=5).fit(x)
    means = sorted(model.means_.flatten().tolist())
    assert np.allclose(means, [0.0, 6.0], atol=0.5)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GaussianMixture().predict(np.array([[0.0, 0.0]]))


def test_too_many_components_raises():
    x = np.random.default_rng(0).normal(size=(5, 2))
    with pytest.raises(ValueError, match="n_components"):
        GaussianMixture(n_components=6).fit(x)


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        GaussianMixture().fit(np.zeros((0, 2)))


def test_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        GaussianMixture().fit(np.array([[0.0, np.nan], [1.0, 1.0]]))


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="n_components"):
        GaussianMixture(n_components=0)
    with pytest.raises(ValueError, match="max_iter"):
        GaussianMixture(max_iter=0)
    with pytest.raises(ValueError, match="reg_covar"):
        GaussianMixture(reg_covar=0)
