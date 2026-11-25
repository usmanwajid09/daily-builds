# PageRank

PageRank by **power iteration**, implemented from scratch with numpy —
the algorithm behind Google's original search ranking.

## How it works

A random surfer follows a random outgoing link with probability
`damping` (default 0.85) and teleports to a uniformly random node
otherwise. The PageRank vector is the stationary distribution of that
walk, found by iterating `rank = damping * Mᵀ @ rank + (1 - damping) / n`
until the change drops below `tol`, where `M` is the row-stochastic
transition matrix built from the adjacency matrix.

**Dangling nodes** (pages with no outgoing links) are treated as linking
to every node, so the transition matrix stays stochastic and the scores
always sum to 1.

## Usage

```python
from pagerank import pagerank, pagerank_from_edges

# Adjacency matrix: A[i, j] = 1 means node i links to node j
scores = pagerank([[0, 1, 1, 0],
                   [0, 0, 1, 0],
                   [1, 0, 0, 0],
                   [0, 0, 1, 0]])
print(scores)  # sums to 1; the most-linked node scores highest

# Or from an edge list
scores = pagerank_from_edges([(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)], n_nodes=4)
```

Bad inputs (non-square or empty matrices, negative entries, damping
outside (0, 1), out-of-range edges) raise `ValueError`/`TypeError` with
a clear message.

## Run the demo

```bash
python pagerank/demo.py
```

Ranks a 4-node web (A → B, C; B → C; C → A; D → C) and prints the scores,
highest first — node C, the most-linked page, ranks top.

## Run the tests

```bash
pytest tests/test_pagerank.py -q
```
