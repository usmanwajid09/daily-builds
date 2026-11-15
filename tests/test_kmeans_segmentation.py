import numpy as np
import pytest

from kmeans_segmentation.segment import KMeansSegmenter, segment_image


def make_blocks(seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((30, 30, 3))
    img[:15, :15] = [0.9, 0.1, 0.1]
    img[:15, 15:] = [0.1, 0.9, 0.1]
    img[15:, :] = [0.1, 0.1, 0.9]
    return np.clip(img + rng.normal(scale=0.03, size=img.shape), 0, 1)


def test_segment_image_returns_shapes():
    img = make_blocks()
    labels, segmented, centers = segment_image(img, n_clusters=3, seed=0)
    assert labels.shape == (30, 30)
    assert segmented.shape == (30, 30, 3)
    assert centers.shape == (3, 3)


def test_three_blocks_recovered():
    img = make_blocks()
    labels, _, centers = segment_image(img, n_clusters=3, seed=0)
    sizes = np.bincount(labels.ravel(), minlength=3)
    assert sorted(sizes.tolist()) == [225, 225, 450]
    for expected in ([0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]):
        assert np.min(np.linalg.norm(centers - expected, axis=1)) < 0.05


def test_segmented_has_only_centre_colours():
    img = make_blocks()
    _, segmented, centers = segment_image(img, n_clusters=3, seed=0)
    uniq = np.unique(segmented.reshape(-1, 3), axis=0)
    assert len(uniq) == 3
    for colour in uniq:
        assert np.min(np.linalg.norm(centers - colour, axis=1)) < 1e-9


def test_grayscale_image():
    rng = np.random.default_rng(1)
    img = np.zeros((20, 20))
    img[:, :10] = 0.2
    img[:, 10:] = 0.8
    img = np.clip(img + rng.normal(scale=0.02, size=img.shape), 0, 1)
    labels, segmented, centers = segment_image(img, n_clusters=2, seed=0)
    assert labels.shape == (20, 20)
    assert segmented.shape == (20, 20)
    assert centers.shape == (2, 1)


def test_uint8_input_accepted():
    img = (make_blocks() * 255).astype(np.uint8)
    labels, segmented, _ = segment_image(img, n_clusters=3, seed=0)
    assert labels.shape == (30, 30)
    assert segmented.max() <= 1.0


def test_cluster_sizes_sum_to_pixels():
    seg = KMeansSegmenter(n_clusters=3, seed=0).fit(make_blocks())
    assert seg.cluster_sizes().sum() == 30 * 30
    assert seg.n_iter_ >= 1


def test_segment_method_matches_labels():
    seg = KMeansSegmenter(n_clusters=3, seed=0).fit(make_blocks())
    segmented = seg.segment()
    assert np.allclose(segmented, seg.centers_[seg.labels_])


def test_deterministic_with_seed():
    img = make_blocks()
    a = segment_image(img, n_clusters=3, seed=5)[0]
    b = segment_image(img, n_clusters=3, seed=5)[0]
    assert np.array_equal(a, b)


def test_invalid_n_clusters_one():
    with pytest.raises(ValueError):
        KMeansSegmenter(n_clusters=1)


def test_too_many_clusters_raises():
    with pytest.raises(ValueError):
        segment_image(np.zeros((2, 2, 3)), n_clusters=5, seed=0)


def test_wrong_dimensions_raises():
    with pytest.raises(ValueError):
        segment_image(np.zeros((10,)), n_clusters=2, seed=0)


def test_segment_before_fit_raises():
    with pytest.raises(ValueError):
        KMeansSegmenter(n_clusters=3).segment()


def test_mismatched_segment_shape_raises():
    seg = KMeansSegmenter(n_clusters=3, seed=0).fit(make_blocks())
    with pytest.raises(ValueError):
        seg.segment(np.zeros((5, 5, 3)))
