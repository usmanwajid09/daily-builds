"""Public API for the mds_embedding package."""

from mds_embedding.model import (
    ClassicalMDS,
    double_centre,
    kruskal_stress,
    pairwise_distances,
)

__all__ = [
    "ClassicalMDS",
    "double_centre",
    "kruskal_stress",
    "pairwise_distances",
]
