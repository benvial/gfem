"""The ``direct`` backend: a symmetric LDL^T factorization on the CPU.

With the ``gfem[mkl]`` extra installed, NGSolve's MKL PARDISO factors with a
METIS nested-dissection ordering and the symmetric matrix type; it fills
about a third less than NGSolve's sparse Cholesky (minimum degree) and
factors several times faster. Without it, NGSolve's sparse Cholesky. Both
give the same answers to the direct solve's accuracy.

Installing ``mkl`` switches NGSolve's process-wide default inverse to PARDISO
(an LU with minimum degree), so every inverse gfem creates names its type.
"""

from __future__ import annotations

import functools

import ngsolve as ngs
import numpy as np
import pyngcore
import scipy.sparse as sp

from gfem._linalg import Operator, Symmetry
from gfem._linalg.base import NgsHandle, host_bytes, same_pattern

#: NGSolve's sparse Cholesky: the factorization without MKL, and the BDDC
#: coarse solve's.
INVERSE = "sparsecholesky"
#: NGSolve's MKL PARDISO wrapper and its flags: METIS ordering, and the
#: symmetric matrix type (gfem stores the full matrix, so NGSolve would
#: otherwise factor it as unsymmetric LU, with twice the fill).
PARDISO = "pardiso"
PARDISO_FLAGS = {"ordering": "metis", "symmetric": True}


@functools.cache
def pardiso_available() -> bool:
    """Whether NGSolve found MKL PARDISO in a pip ``mkl`` (the ``mkl`` extra)."""
    try:
        from ngsolve.solvers import mkl_pardiso
    except ImportError:
        return False
    return (
        mkl_pardiso._pardiso is not None  # noqa: SLF001
        and mkl_pardiso._mkl_free_buffers is not None  # noqa: SLF001
    )


class DirectHandle(NgsHandle):
    """One LDL^T factorization, reused for every right-hand side.

    MKL PARDISO when the ``mkl`` extra is installed, NGSolve's sparse
    Cholesky otherwise (see the module docstring). Both factor
    ``A = L D L^T``, so they need ``A^T = A``: a real symmetric operator
    (``"hermitian"`` on a real space) or a complex symmetric one. Complex
    Hermitian and general operators are refused.

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
        inverse = INVERSE
        if pardiso_available():
            inverse = PARDISO
            matrix.inverse_flags = pyngcore.Flags(PARDISO_FLAGS)  # ty: ignore[unresolved-attribute]
        if freedofs is None:
            return matrix.Inverse(inverse=inverse)
        return matrix.Inverse(freedofs, inverse=inverse)

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
            "The direct solver (A = L D L^T) needs A^T = A: a "
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
        factors = host_bytes(self._solver)
        params = getattr(self._solver, "_params", None)
        if not factors and params is not None:
            # NGSolve's PARDISO wrapper reports no memory: take PARDISO's own
            # permanent (iparm[15]) and factor (iparm[16]) memory, in KiB
            factors = 1024 * (int(params[15]) + int(params[16]))
        return host_bytes(self._matrix) + factors
