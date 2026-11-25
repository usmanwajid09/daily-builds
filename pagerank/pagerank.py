"""PageRank by power iteration, implemented with numpy.

Given a directed graph as an adjacency matrix ``A`` where ``A[i, j] = 1``
means node ``i`` links to node ``j``, the PageRank vector is the stationary
distribution of a random surfer who follows a random outgoing link with
probability ``damping`` and teleports to a uniformly random node otherwise.

Dangling nodes (no outgoing links) are treated as linking to every node,
so the transition matrix stays stochastic.
"""

import numpy as np

__all__ = ["pagerank", "pagerank_from_edges"]


def _validate_adjacency(adjacency):
    if isinstance(adjacency, (str, bytes)):
        raise TypeError("adjacency must be a 2-D array of numbers, got a string")
    try:
        A = np.asarray(adjacency, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("adjacency must be a 2-D array of numbers")
    if A.ndim != 2:
        raise ValueError(f"adjacency must be 2-D, got shape {A.shape}")
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"adjacency must be square, got shape {A.shape}"
        )
    if A.shape[0] == 0:
        raise ValueError("adjacency must contain at least one node")
    if not np.all(np.isfinite(A)):
        raise ValueError("adjacency must contain only finite numbers")
    if np.any(A < 0):
        raise ValueError("adjacency must not contain negative entries")
    return A


def pagerank(adjacency, damping=0.85, max_iter=100, tol=1e-6):
    """Compute PageRank scores for a graph given as an adjacency matrix.

    Returns a 1-D array of scores that sums to 1, ordered by node index.
    ``A[i, j]`` is the weight of the link from node ``i`` to node ``j``.
    """
    if not isinstance(damping, (int, float, np.floating)) or not (
        0.0 < float(damping) < 1.0
    ):
        raise ValueError(f"damping must be strictly between 0 and 1, got {damping!r}")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be an integer >= 1, got {max_iter!r}")
    if not isinstance(tol, (int, float, np.floating)) or float(tol) <= 0:
        raise ValueError(f"tol must be positive, got {tol!r}")

    A = _validate_adjacency(adjacency)
    n = A.shape[0]
    out_degree = A.sum(axis=1)
    # Row-stochastic transition matrix; dangling rows link everywhere.
    M = np.empty_like(A)
    for i in range(n):
        if out_degree[i] > 0:
            M[i] = A[i] / out_degree[i]
        else:
            M[i] = 1.0 / n
    teleport = (1.0 - float(damping)) / n
    rank = np.full(n, 1.0 / n)
    for _ in range(int(max_iter)):
        new_rank = float(damping) * (M.T @ rank) + teleport
        if np.linalg.norm(new_rank - rank, ord=1) < float(tol):
            rank = new_rank
            break
        rank = new_rank
    rank = rank / rank.sum()
    return rank


def pagerank_from_edges(edges, n_nodes, **kwargs):
    """Compute PageRank from a list of ``(source, target)`` edge tuples.

    ``n_nodes`` is the number of nodes (indices ``0 .. n_nodes - 1``).
    Duplicate edges count once per occurrence, i.e. they add weight.
    """
    if not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1:
        raise ValueError(f"n_nodes must be an integer >= 1, got {n_nodes!r}")
    try:
        edges = list(edges)
    except TypeError:
        raise TypeError("edges must be an iterable of (source, target) pairs")
    A = np.zeros((n_nodes, n_nodes))
    for edge in edges:
        try:
            src, dst = edge
        except (TypeError, ValueError):
            raise ValueError(
                f"each edge must be a (source, target) pair, got {edge!r}"
            )
        if not isinstance(src, (int, np.integer)) or not isinstance(
            dst, (int, np.integer)
        ):
            raise TypeError(
                f"edge endpoints must be integers, got {edge!r}"
            )
        if not (0 <= src < n_nodes) or not (0 <= dst < n_nodes):
            raise ValueError(
                f"edge {edge!r} is outside the node range 0..{n_nodes - 1}"
            )
        A[src, dst] += 1.0
    return pagerank(A, **kwargs)
