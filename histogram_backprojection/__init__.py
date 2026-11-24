"""Histogram backprojection for colour-based object localisation."""

from histogram_backprojection.backprojection import (
    backproject,
    build_histogram,
    localise,
    locate_centroid,
    locate_peak,
)

__all__ = [
    "backproject",
    "build_histogram",
    "localise",
    "locate_centroid",
    "locate_peak",
]
