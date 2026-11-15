import numpy as np
import pytest

from eigenfaces.model import Eigenfaces


def make_faces(seed=0, size=12, n_identities=3, per_identity=4):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for ident in range(n_identities):
        base = np.zeros((size, size))
        for _ in range(3):
            cy, cx = rng.integers(0, size, 2)
            yy, xx = np.mgrid[0:size, 0:size]
            base += rng.uniform(0.4, 1.0) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 6.0)
        for _ in range(per_identity):
            images.append(base + rng.normal(scale=0.05, size=(size, size)))
            labels.append(ident)
    return np.array(images), np.array(labels)


def test_fit_sets_components_and_mean():
    images, labels = make_faces()
    model = Eigenfaces(n_components=5).fit(images, labels)
    assert model.components_.shape == (5, 12 * 12)
    assert model.mean_face_.shape == (12 * 12,)
    assert model.explained_variance_ratio_.shape == (5,)


def test_variance_ratios_sum_below_one():
    images, _ = make_faces()
    model = Eigenfaces(n_components=5).fit(images)
    assert 0 < model.explained_variance_ratio_.sum() <= 1.0


def test_predict_recognises_training_faces():
    images, labels = make_faces()
    model = Eigenfaces(n_components=5).fit(images, labels)
    assert np.array_equal(model.predict(images), labels)


def test_predict_noisy_copy_of_known_identity():
    images, labels = make_faces()
    model = Eigenfaces(n_components=5).fit(images, labels)
    rng = np.random.default_rng(7)
    query = images[5] + rng.normal(scale=0.05, size=images[5].shape)
    assert model.predict(query)[0] == labels[5]


def test_reconstruct_shape_and_low_error():
    images, _ = make_faces()
    model = Eigenfaces(n_components=8).fit(images)
    rebuilt = model.reconstruct(images)
    assert rebuilt.shape == images.shape
    assert np.mean((rebuilt - images) ** 2) < 0.01


def test_reconstruction_error_decreases_with_components():
    images, _ = make_faces()
    err2 = Eigenfaces(n_components=2).fit(images).reconstruction_error(images).mean()
    err8 = Eigenfaces(n_components=8).fit(images).reconstruction_error(images).mean()
    assert err8 < err2


def test_transform_shape():
    images, _ = make_faces()
    model = Eigenfaces(n_components=4).fit(images)
    assert model.transform(images).shape == (len(images), 4)
    assert model.transform(images[0]).shape == (1, 4)


def test_invalid_n_components_zero():
    with pytest.raises(ValueError):
        Eigenfaces(n_components=0)


def test_too_many_components_raises():
    images, _ = make_faces()
    with pytest.raises(ValueError):
        Eigenfaces(n_components=100).fit(images)


def test_single_image_fit_raises():
    images, _ = make_faces()
    with pytest.raises(ValueError):
        Eigenfaces(n_components=1).fit(images[:1])


def test_mismatched_labels_raises():
    images, _ = make_faces()
    with pytest.raises(ValueError):
        Eigenfaces(n_components=2).fit(images, labels=[0, 1])


def test_predict_before_fit_raises():
    images, _ = make_faces()
    with pytest.raises(ValueError):
        Eigenfaces(n_components=2).predict(images)


def test_wrong_image_size_raises():
    images, _ = make_faces()
    model = Eigenfaces(n_components=2).fit(images)
    with pytest.raises(ValueError):
        model.predict(np.zeros((8, 8)))


def test_nonfinite_images_raises():
    images, _ = make_faces()
    images[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        Eigenfaces(n_components=2).fit(images)
