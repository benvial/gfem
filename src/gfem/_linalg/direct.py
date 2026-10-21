"""The ``direct`` backend: NGSolve sparse Cholesky (LDL^T) factorization."""

from __future__ import annotations

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from gfem._linalg import Operator, Symmetry
from gfem._linalg.base import NgsHandle, host_bytes, same_pattern

INVERSE = "sparsecholesky"


class DirectHandle(NgsHandle):
    """One sparse Cholesky factorization, reused for every right-hand side.

    Sparse Cholesky factors ``A = L D L^T``, so it needs ``A^T = A``: a real
    symmetric operator (``"hermitian"`` on a real space) or a complex
    symmetric one. Complex Hermitian and general operators are refused.

    `refactor` factorizes another operator's values in place (NGSolve's
    ``Update()``): a `gfem._linalg.Refactorable` handle.
    """

    kind = "direct"

    def __init__(
        self,
        matrix: ngs.BaseMatrix,
        freedofs: ngs.BitArray | None,
        symmetry: Symmetry,
        *,
        is_complex: bool,
    ) -> None:
        """Factorize an assembled matrix (see `prepare` and `factorize`)."""
        if symmetry == "general" or (symmetry == "hermitian" and is_complex):
            self._refuse(symmetry)
        super().__init__(matrix, freedofs, symmetry, is_complex=is_complex)
        self._solver = self._factor(matrix, freedofs)

    @staticmethod
    def _factor(matrix: ngs.BaseMatrix, freedofs: ngs.BitArray | None):
        if freedofs is None:
            return matrix.Inverse(inverse=INVERSE)
        return matrix.Inverse(freedofs, inverse=INVERSE)

    @classmethod
    def from_operator(cls, operator: Operator, symmetry: Symmetry) -> DirectHandle:
        """Assemble and factorize (see `gfem._linalg.prepare`)."""
        form = operator.form
        if symmetry == "general" or (symmetry == "hermitian" and form.space.is_complex):
            # refuse before paying for the assembly
            cls._refuse(symmetry)
        form.Assemble()
        return cls(
            form.mat,
            operator.freedofs,
            symmetry,
            is_complex=bool(form.space.is_complex),
        )

    @classmethod
    def from_scipy(cls, matrix: sp.sparray | sp.spmatrix) -> DirectHandle:
        """Factorize a real symmetric SciPy matrix (see `gfem._linalg.factorize`)."""
        coo = sp.coo_array(matrix)
        n, m = coo.shape
        if n != m:
            raise ValueError(f"factorize needs a square matrix, got shape {coo.shape}.")
        assembled = ngs.la.SparseMatrixd.CreateFromCOO(
            coo.row.astype(np.int64),
            coo.col.astype(np.int64),
            np.asarray(coo.data, dtype=float),
            n,
            n,
        )
        return cls(assembled, None, "hermitian", is_complex=False)

    def refactor(self, operator: Operator) -> None:
        """Assemble ``operator`` and refactorize (see `gfem._linalg.Refactorable`).

        The new values are copied into the factored matrix, and NGSolve's
        ``Update()`` factorizes them again; a changed pattern is factorized
        anew.
        """
        matrix = self._matrix
        if self._solver is None or matrix is None:
            raise RuntimeError("This solver handle was freed.")
        form = operator.form
        form.Assemble()
        with self._lock:
            if same_pattern(matrix, form.mat):
                matrix.AsVector().data = form.mat.AsVector()
                self._solver.Update()
            else:
                self._matrix = form.mat
                self._solver = self._factor(form.mat, operator.freedofs)

    @staticmethod
    def _refuse(symmetry: Symmetry) -> None:
        raise ValueError(
            "The direct solver (sparse Cholesky, A = L D L^T) needs A^T = A: a "
            f"real symmetric or complex symmetric operator, not {symmetry!r}."
        )

    @property
    def _inverse(self):
        return self._solver

    @property
    def nbytes(self) -> int:
        """Host memory of the matrix and the factors, in bytes."""
        if self._solver is None or self._matrix is None:
            return 0
        return host_bytes(self._matrix) + host_bytes(self._solver)
