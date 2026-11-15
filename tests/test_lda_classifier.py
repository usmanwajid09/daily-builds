import numpy as np
import pytest

from lda_classifier.model import LDAClassifier


def _data(seed=42, n=60):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [4.0, 3.0], [0.0, 5.0]])
    xs, ys = [], []
    for k, c in enumerate(centres):
        xs.append(rng.normal(loc=c, scale=1.0, size=(n, 2)))
        ys.append(np.full(n, k))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_high_accuracy_on_separated_blobs():
    x, y = _data()
    model = LDAClassifier().fit(x, y)
    assert model.score(x, y) > 0.9


def test_predict_returns_known_classes():
    x, y = _data()
    model = LDAClassifier().fit(x, y)
    preds = model.predict(x)
    assert set(np.unique(preds)) <= {0, 1, 2}
    assert preds.shape == y.shape


def test_predict_proba_sums_to_one():
    x, y = _data()
    model = LDAClassifier().fit(x, y)
    probs = model.predict_proba(x)
    assert probs.shape == (len(y), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_matches_highest_proba():
    x, y = _data()
    model = LDAClassifier().fit(x, y)
    preds = model.predict(x)
    assert np.array_equal(preds, model.classes_[np.argmax(model.predict_proba(x), axis=1)])


def test_priors_estimated_from_frequencies():
    x, y = _data()
    model = LDAClassifier().fit(x, y)
    assert np.allclose(model.priors_, [1 / 3, 1 / 3, 1 / 3])


def test_custom_priors_used():
    x, y = _data()
    model = LDAClassifier(priors=[0.6, 0.2, 0.2]).fit(x, y)
    assert np.allclose(model.priors_, [0.6, 0.2, 0.2])


def test_shrinkage_keeps_accuracy():
    x, y = _data()
    model = LDAClassifier(shrinkage=0.5).fit(x, y)
    assert model.score(x, y) > 0.9
    assert np.isfinite(model.covariance_).all()


def test_handles_string_labels():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(loc=[0, 0], scale=1, size=(40, 2)),
                   rng.normal(loc=[5, 5], scale=1, size=(40, 2))])
    y = np.array(["cat"] * 40 + ["dog"] * 40)
    model = LDAClassifier().fit(x, y)
    assert set(model.classes_) == {"cat", "dog"}
    assert model.score(x, y) > 0.95


def test_one_class_raises():
    x, y = _data()
    with pytest.raises(ValueError, match="at least 2 classes"):
        LDAClassifier().fit(x, np.zeros_like(y))


def test_bad_shrinkage_raises():
    with pytest.raises(ValueError, match="shrinkage"):
        LDAClassifier(shrinkage=1.5)
    with pytest.raises(ValueError, match="shrinkage"):
        LDAClassifier(shrinkage=-0.1)


def test_bad_priors_raise():
    with pytest.raises(ValueError, match="priors"):
        LDAClassifier(priors=[0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="priors"):
        LDAClassifier(priors=[-0.2, 1.2])


def test_priors_wrong_length_raises():
    x, y = _data()
    with pytest.raises(ValueError, match="priors"):
        LDAClassifier(priors=[0.5, 0.5]).fit(x, y)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        LDAClassifier().predict(np.zeros((2, 2)))


def test_mismatched_shapes_raise():
    x, y = _data()
    with pytest.raises(ValueError, match="same number of samples"):
        LDAClassifier().fit(x, y[:10])
    with pytest.raises(ValueError, match="2D"):
        LDAClassifier().fit(x.ravel(), y)


def test_nan_input_raises():
    x, y = _data()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        LDAClassifier().fit(x, y)


def test_singular_covariance_handled():
    # Duplicate a feature so the covariance is singular.
    x, y = _data()
    x = np.c_[x, x[:, 0]]
    model = LDAClassifier().fit(x, y)
    assert model.score(x, y) > 0.9
