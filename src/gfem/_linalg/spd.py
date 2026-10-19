"""A sparse symmetric positive definite factor on the host (NGSolve ``sparsecholesky``).

For small auxiliary systems assembled in NumPy / SciPy, such as the boundary
morph's mesh extension: factorize once, then solve many right-hand sides.
"""

from __future__ import annotations

import threading

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp


class SpdFactor:
    """A Cholesky factor of a sparse SPD matrix, reused for every solve."""

    def __init__(self, matrix: sp.sparray | sp.spmatrix) -> None:
        """Factorize ``matrix`` (square, symmetric positive definite)."""
        coo = sp.coo_array(matrix)
        n, m = coo.shape
        if n != m:
            msg = f"SpdFactor needs a square matrix, got shape {coo.shape}."
            raise ValueError(msg)
        self.n = n
        self._matrix = ngs.la.SparseMatrixd.CreateFromCOO(
            coo.row.astype(np.int64),
            coo.col.astype(np.int64),
            np.asarray(coo.data, dtype=float),
            n,
            n,
        )
        self._inverse = self._matrix.Inverse(inverse="sparsecholesky")
        self._lock = threading.Lock()

    def solve(self, b: np.ndarray) -> np.ndarray:
        """The solution x of ``matrix @ x = b`` for one right-hand side."""
        b = np.asarray(b, dtype=float)
        with self._lock:
            rhs = self._matrix.CreateVector()
            rhs.FV().NumPy()[:] = b
            x = self._matrix.CreateVector()
            x.data = self._inverse * rhs
            return x.FV().NumPy().copy()
