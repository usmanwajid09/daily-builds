"""Rank a small web of pages with PageRank.

Graph (4 nodes): A links to B and C; B links to C; C links to A;
D links to C only. Node C is the most-linked page, so it should rank
highest. Prints the score of every node, highest first.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pagerank import pagerank_from_edges

NODES = ["A", "B", "C", "D"]
EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A"), ("D", "C")]


def main():
    index = {name: i for i, name in enumerate(NODES)}
    edges = [(index[s], index[t]) for s, t in EDGES]
    scores = pagerank_from_edges(edges, n_nodes=len(NODES), damping=0.85)
    order = np.argsort(-scores)
    print("Graph: A -> B, C;  B -> C;  C -> A;  D -> C")
    print(f"Nodes: {NODES}, damping 0.85, scores sum to {scores.sum():.6f}\n")
    for i in order:
        print(f"  {NODES[i]}: {scores[i]:.4f}")
    top = NODES[int(order[0])]
    print(f"\nTop-ranked node: {top} (score {scores[order[0]]:.4f})")


if __name__ == "__main__":
    main()
