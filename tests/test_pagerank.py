import numpy as np
import pytest

from pagerank import pagerank, pagerank_from_edges

# Demo graph: A -> B, C; B -> C; C -> A; D -> C
DEMO_A = np.array(
    [[0, 1, 1, 0],
     [0, 0, 1, 0],
     [1, 0, 0, 0],
     [0, 0, 1, 0]],
    dtype=float,
)


def test_scores_sum_to_one():
    scores = pagerank(DEMO_A)
    assert scores.sum() == pytest.approx(1.0)


def test_all_scores_positive():
    scores = pagerank(DEMO_A)
    assert np.all(scores > 0)


def test_most_linked_node_ranks_top():
    scores = pagerank(DEMO_A)
    assert int(np.argmax(scores)) == 2  # node C


def test_demo_headline_c_score_above_quarter():
    scores = pagerank(DEMO_A)
    assert scores[2] > 0.3


def test_symmetric_two_node_cycle_is_uniform():
    scores = pagerank([[0, 1], [1, 0]])
    assert scores == pytest.approx([0.5, 0.5])


def test_single_node_scores_one():
    assert pagerank([[0]]) == pytest.approx([1.0])
    assert pagerank([[1]]) == pytest.approx([1.0])


def test_dangling_node_handled():
    # Node 1 has no outgoing links; scores must still sum to 1.
    scores = pagerank([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert scores.sum() == pytest.approx(1.0)
    assert np.all(scores > 0)


def test_all_dangling_nodes_are_uniform():
    scores = pagerank(np.zeros((3, 3)))
    assert scores == pytest.approx([1.0 / 3] * 3)


def test_from_edges_matches_matrix():
    edges = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)]
    assert pagerank_from_edges(edges, 4) == pytest.approx(pagerank(DEMO_A))


def test_damping_changes_scores():
    low = pagerank(DEMO_A, damping=0.5)
    high = pagerank(DEMO_A, damping=0.95)
    assert not np.allclose(low, high)
    assert low.sum() == pytest.approx(1.0)
    assert high.sum() == pytest.approx(1.0)


def test_empty_matrix_raises():
    with pytest.raises(ValueError, match="at least one node"):
        pagerank(np.zeros((0, 0)))


def test_non_square_matrix_raises():
    with pytest.raises(ValueError, match="square"):
        pagerank([[0, 1, 0], [1, 0, 1]])


def test_negative_entries_raise():
    with pytest.raises(ValueError, match="negative"):
        pagerank([[0, -1], [1, 0]])


def test_nan_entries_raise():
    with pytest.raises(ValueError, match="finite"):
        pagerank([[0, float("nan")], [1, 0]])


def test_bad_damping_raises():
    with pytest.raises(ValueError, match="damping"):
        pagerank(DEMO_A, damping=1.0)
    with pytest.raises(ValueError, match="damping"):
        pagerank(DEMO_A, damping=0.0)


def test_edge_out_of_range_raises():
    with pytest.raises(ValueError, match="outside the node range"):
        pagerank_from_edges([(0, 5)], n_nodes=3)


def test_edge_with_bad_pair_raises():
    with pytest.raises(ValueError, match="pair"):
        pagerank_from_edges([(0,)], n_nodes=2)


def test_bad_n_nodes_raises():
    with pytest.raises(ValueError, match="n_nodes"):
        pagerank_from_edges([], n_nodes=0)
