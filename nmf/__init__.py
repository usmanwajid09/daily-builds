"""Non-negative matrix factorization (NMF) from scratch.

Factorize a non-negative matrix ``V`` into two non-negative matrices
``W`` and ``H`` such that ``V ≈ W @ H``, using the Lee-Seung
multiplicative update rules. Only numpy is used.
"""

from .model import NMF

__all__ = ["NMF"]
