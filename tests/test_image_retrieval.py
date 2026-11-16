import numpy as np
import pytest

from image_retrieval import (
    HIST_SIZE,
    ImageIndex,
    color_histogram,
    histogram_correlation,
    make_blocks,
    make_gradient,
    make_solid,
    make_split,
)


def test_histogram_sums_to_one_and_has_expected_size():
    hist = color_histogram(make_solid((10, 20, 30)))
    assert hist.shape == (HIST_SIZE,)
    assert hist.sum() == pytest.approx(1.0)
    assert (hist >= 0).all()


def test_identical_images_correlate_at_one():
    img = make_gradient((255, 120, 40), (40, 60, 200))
    h = color_histogram(img)
    assert histogram_correlation(h, h) == pytest.approx(1.0)


def test_very_different_images_correlate_low():
    h_red = color_histogram(make_solid((255, 0, 0)))
    h_blue = color_histogram(make_solid((0, 0, 255)))
    assert histogram_correlation(h_red, h_blue) < 0.0


def test_query_returns_indexed_image_first_with_score_one():
    index = ImageIndex()
    img_a = make_gradient((255, 120, 40), (255, 180, 90))
    img_b = make_solid((20, 90, 180))
    index.add("a", img_a).add("b", img_b)
    results = index.query(img_a, top_k=2)
    assert results[0] == ("a", pytest.approx(1.0))
    assert results[1][0] == "b"


def test_query_respects_top_k_and_sorts_descending():
    index = ImageIndex()
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        index.add(f"img-{i}", make_solid(color))
    results = index.query(make_solid((250, 10, 10)), top_k=2)
    assert len(results) == 2
    assert results[0][1] >= results[1][1]
    assert results[0][0] == "img-0"


def test_similar_colours_rank_above_dissimilar():
    index = ImageIndex()
    index.add("red", make_solid((255, 0, 0)))
    index.add("blue", make_solid((0, 0, 255)))
    results = index.query(make_solid((240, 20, 20)), top_k=2)
    assert results[0][0] == "red"


def test_query_on_empty_index_raises():
    with pytest.raises(ValueError):
        ImageIndex().query(make_solid((1, 2, 3)))


def test_invalid_image_shape_raises():
    with pytest.raises(ValueError):
        color_histogram(np.zeros((64, 64)))


def test_out_of_range_values_raise():
    with pytest.raises(ValueError):
        color_histogram(np.full((8, 8, 3), 300.0))


def test_mismatched_histogram_shapes_raise():
    with pytest.raises(ValueError):
        histogram_correlation(np.zeros(8), np.zeros(16))


def test_invalid_top_k_raises():
    index = ImageIndex().add("a", make_solid((1, 2, 3)))
    with pytest.raises(ValueError):
        index.query(make_solid((1, 2, 3)), top_k=0)


def test_split_and_blocks_produce_expected_shapes():
    assert make_split((255, 0, 0), (0, 0, 255)).shape == (64, 64, 3)
    assert make_blocks([(255, 0, 0), (0, 255, 0)]).shape == (64, 64, 3)
    assert len(ImageIndex()) == 0
