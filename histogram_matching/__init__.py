"""Histogram specification: match one image's histogram to another's."""

from histogram_matching.histogram_matching import (
    compute_cdf,
    compute_histogram,
    match_histogram,
)

__all__ = ["compute_cdf", "compute_histogram", "match_histogram"]
