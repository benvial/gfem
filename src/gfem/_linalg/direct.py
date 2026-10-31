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
from collections.abc import Callable
from typing import Any, Protocol, cast

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from gfem._linalg import Symmetry
from gfem._linalg.base import PlannedHandle, equals_its_transpose, host_bytes
from gfem._linalg.pardiso import (  # noqa: F401 (the first two re-exported)
    PARDISO_FLAGS,
    PardisoFactors,
    pardiso_available,
    pardiso_inverse,
)

#: NGSolve's sparse Cholesky: the factorization without MKL, also the BDDC
#: coarse factorization's.
INVERSE = "sparsecholesky"


@functools.cache
def _real_sparse_matrix() -> Any:
    """NGSolve's ``SparseMatrix<double>`` class.

    NGSolve names it after the C++ ``typeid``: ``SparseMatrixd`` with GCC and
    Clang, another name with MSVC (Windows), so it is found by what its
    ``CreateFromCOO`` returns.
    """
    for cls in vars(ngs.la).values():
        if not (isinstance(cls, type) and "CreateFromCOO" in vars(cls)):
            continue
        signature = cls.CreateFromCOO.__doc__ or ""
        if signature.rstrip().endswith("SparseMatrixTM<double>"):
            return cls
    raise RuntimeError("ngsolve.la has no real SparseMatrix with CreateFromCOO")


class Factors(Protocol):
    """A direct handle's inverse, and how it factorizes new values on its plan."""

    #: The inverse a solve applies.
    inverse: Any

    def new_values(
        self, matrix: ngs.BaseMatrix, write: Callable[[ngs.BaseMatrix], None]
    ) -> None:
        """``write(matrix)``, then factorize its new values on the plan."""
        ...

    @property
    def nbytes(self) -> int:
        """Host memory of the factors and whatever else this holds, in bytes."""
        ...


class UpdateFactors:
    """An inverse NGSolve refactorizes itself (``Update()``): sparse Cholesky."""

    def __init__(self, inverse: ngs.BaseMatrix) -> None:
        """Hold ``inverse``."""
        self.inverse = inverse

    def new_values(
        self, matrix: ngs.BaseMatrix, write: Callable[[ngs.BaseMatrix], None]
    ) -> None:
        """``write(matrix)``, then ``Update()`` the inverse."""
        write(matrix)
        self.inverse.Update()

    @property
    def nbytes(self) -> int:
        """Host memory NGSolve reports for the factors, in bytes."""
        return host_bytes(self.inverse)


class DirectHandle(PlannedHandle):
    """One LDL^T factorization, reused for every right-hand side.

    MKL PARDISO when the ``mkl`` extra is installed, NGSolve's sparse
    Cholesky otherwise (see the module docstring). Both factor
    ``A = L D L^T``, so they need ``A^T = A``: a real symmetric operator
    (``"hermitian"`` on a real space) or a complex symmetric one. Complex
    Hermitian and general operators are refused.

    An operator with a piece sum (``Operator.piece_sum``) is summed, never
    assembled: into a new matrix at creation, into the handle's own matrix at
    `refactorize`.

    `refactorize` factorizes another operator's values in place: a
    `gfem._linalg.Refactorable` handle. PARDISO gathers them straight into
    its wrapper's buffer and runs its numeric factorization (phase 22);
    sparse Cholesky, or a wrapper without the private attributes PARDISO's
    `Factors` adapter (`gfem._linalg.pardiso.PardisoFactors`) reads, runs
    NGSolve's ``Update()``. The gather index is built at the first
    refactorization, so a handle factored once never builds it.

    A PARDISO handle refactorizes and solves on the thread count it was
    factored with, whatever TaskManager the caller is in: a solve outside the
    TaskManager it was factored in stays parallel, and one factored outside
    any TaskManager stays on one thread (`gfem._linalg.pardiso`).
    """

    kind = "direct"
    sums_pieces = True

    def __init__(
        self,
        matrix: ngs.BaseMatrix,
        freedofs: ngs.BitArray | None,
        symmetry: Symmetry,
        *,
        is_complex: bool,
    ) -> None:
        """Factorize an assembled matrix (see `prepare` and `factorize`).

        ``symmetry`` is refused before (`from_operator`; `from_scipy`'s real
        symmetric matrix always passes).
        """
        super().__init__(matrix, freedofs, symmetry, is_complex=is_complex)
        self._factors: Factors | None = self._factor(matrix, freedofs)
        self._solver = self._factors.inverse

    def _factor(self, matrix: ngs.BaseMatrix, freedofs: ngs.BitArray | None) -> Factors:
        free = matrix.height if freedofs is None else freedofs.NumSet()
        # gfem's registered PARDISO type, which applies `PARDISO_FLAGS` and
        # runs its calls (phases 12, 22 and 33) at OpenMP block time 0; never
        # on an empty matrix, which PARDISO refuses
        pardiso = (
            pardiso_inverse(self.symmetry, is_complex=self._complex)
            if free > 0
            else None
        )
        inverse = pardiso or INVERSE
        if freedofs is None or free == matrix.height:
            # every DOF free: NGSolve factors the matrix itself instead of
            # projecting it on the free DOFs first
            solver = matrix.Inverse(inverse=inverse)
        else:
            solver = matrix.Inverse(freedofs, inverse=inverse)
        if pardiso is None:
            return UpdateFactors(solver)
        return PardisoFactors(solver, self._free)

    @classmethod
    def from_scipy(cls, matrix: sp.sparray | sp.spmatrix) -> DirectHandle:
        """Factorize a real symmetric SciPy matrix (see `gfem._linalg.factorize`)."""
        coo = sp.coo_array(matrix)
        n, m = coo.shape
        if n != m:
            raise ValueError(f"factorize needs a square matrix, got shape {coo.shape}.")
        assembled = _real_sparse_matrix().CreateFromCOO(
            coo.row.astype(np.int64),
            coo.col.astype(np.int64),
            np.asarray(coo.data, dtype=float),
            n,
            n,
        )
        return cls(assembled, None, "hermitian", is_complex=False)

    def _new_values(self, write: Callable[[ngs.BaseMatrix], None]) -> None:
        """Write the held matrix and factorize it on the plan; under the lock."""
        matrix = self._live()
        # a live handle holds its factors (`_release` drops them)
        cast("Factors", self._factors).new_values(matrix, write)

    def _new_plan(self, matrix: ngs.BaseMatrix, freedofs: ngs.BitArray | None) -> None:
        """Factorize ``matrix`` anew and hold it; under the lock.

        Factors before it holds anything: a failure leaves the handle as it was.
        """
        factors = self._factor(matrix, freedofs)
        self._matrix, self._solver, self._factors = matrix, factors.inverse, factors

    @classmethod
    def _admit(cls, symmetry: Symmetry, *, is_complex: bool) -> None:
        if not equals_its_transpose(symmetry, is_complex=is_complex):
            raise ValueError(
                "The direct solver (A = L D L^T) needs A^T = A: a "
                f"real symmetric or complex symmetric operator, not {symmetry!r}."
            )

    @property
    def _inverse(self):
        return self._solver

    @property
    def nbytes(self) -> int:
        """Host memory of the matrix and the factors, in bytes.

        PARDISO's factors count the gather index once built (at the first
        `refactorize`).
        """
        if self._factors is None or self._matrix is None:
            return 0
        return host_bytes(self._matrix) + self._factors.nbytes

    def _release(self) -> None:
        """Drop the factors; under the lock."""
        self._factors = None
