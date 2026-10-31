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
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from gfem._linalg import Symmetry
from gfem._linalg.base import PlannedHandle, equals_its_transpose, host_bytes
from gfem._linalg.pardiso import (  # noqa: F401 (the first two re-exported)
    PARDISO_FLAGS,
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
    sparse Cholesky, or a wrapper without the private attributes this reads,
    runs NGSolve's ``Update()``. The gather index is built at the first
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
        """Factorize an assembled matrix (see `prepare` and `factorize`)."""
        self._admit(symmetry, is_complex=is_complex)
        super().__init__(matrix, freedofs, symmetry, is_complex=is_complex)
        self._solver = self._factor(matrix, freedofs, symmetry, is_complex=is_complex)
        #: Built at the first `refactorize` (`_gather_built`); ``None`` once
        #: built refactorizes by ``Update()``.
        self._gather: np.ndarray | None = None
        self._gather_built = False

    @staticmethod
    def _factor(
        matrix: ngs.BaseMatrix,
        freedofs: ngs.BitArray | None,
        symmetry: Symmetry,
        *,
        is_complex: bool,
    ):
        free = matrix.height if freedofs is None else freedofs.NumSet()
        # gfem's registered PARDISO type, which applies `PARDISO_FLAGS` and
        # runs its calls (phases 12, 22 and 33) at OpenMP block time 0; never
        # on an empty matrix, which PARDISO refuses
        pardiso = pardiso_inverse(symmetry, is_complex=is_complex) if free > 0 else None
        inverse = pardiso or INVERSE
        if freedofs is None or free == matrix.height:
            # every DOF free: NGSolve factors the matrix itself instead of
            # projecting it on the free DOFs first
            return matrix.Inverse(inverse=inverse)
        return matrix.Inverse(freedofs, inverse=inverse)

    def _ensure_gather(self, matrix, solver) -> None:
        """Build the gather index if not yet built; under the lock.

        Call it before ``matrix`` takes new values: the wrapper's buffer
        then holds the factored matrix's free upper triangle, which the index
        is checked against.
        """
        if not self._gather_built:
            self._gather = self._gather_index(matrix, solver)
            self._gather_built = True

    def _gather_index(self, matrix, solver) -> np.ndarray | None:
        """Where PARDISO's buffer reads the factored matrix's values, or ``None``.

        NGSolve's wrapper factors the free-row, free-column upper triangle in
        CSR order (its ``get_csr``): these are that triangle's positions in
        the matrix's value vector. Checked once against the wrapper's buffer
        of the factored values; ``None`` (refactorize by ``Update()``) on a
        mismatch, for a solver without the wrapper's ``_data_np`` and
        ``_call_pardiso``, or when reading the matrix or the wrapper fails:
        it never raises.
        """
        try:
            return self._checked_gather_index(matrix, solver)
        except Exception:  # noqa: BLE001 - any failure falls back to Update()
            return None

    def _checked_gather_index(self, matrix, solver) -> np.ndarray | None:
        buffer = getattr(solver, "_data_np", None)
        if not isinstance(buffer, np.ndarray) or not hasattr(solver, "_call_pardiso"):
            return None
        data, cols, rowptr = matrix.CSR()
        cols = np.asarray(cols).astype(np.int64)
        rowptr = np.asarray(rowptr).astype(np.int64)
        rows = np.repeat(np.arange(matrix.height), np.diff(rowptr))
        free = self._free
        index = np.flatnonzero(free[rows] & free[cols] & (cols >= rows))
        index = index.astype(np.int32 if len(cols) < 2**31 else np.int64)
        if index.size != buffer.size or not np.array_equal(
            np.asarray(data)[index], buffer
        ):
            return None
        return index

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
        """Write the held matrix and factorize it on the plan; under the lock.

        The first call builds the gather index, before the values change.
        """
        matrix, solver = self._live(), self._solver
        self._ensure_gather(matrix, solver)
        write(matrix)
        self._refactor_values(matrix, solver)

    def _new_plan(self, matrix: ngs.BaseMatrix, freedofs: ngs.BitArray | None) -> None:
        """Factorize ``matrix`` anew and hold it; under the lock.

        Factors before it holds anything: a failure leaves the handle as it was.
        """
        solver = self._factor(matrix, freedofs, self.symmetry, is_complex=self._complex)
        self._matrix, self._solver = matrix, solver
        self._gather, self._gather_built = None, False

    def _refactor_values(self, matrix, solver) -> None:
        """Factorize ``matrix``'s new values on ``solver``'s plan; under the lock."""
        if self._gather is None:
            solver.Update()
            return
        # the index is in range (checked when built); "clip" writes straight
        # into the buffer, "raise" through a temporary copy
        buffer = solver._data_np  # noqa: SLF001
        values = np.asarray(matrix.CSR()[0])
        np.take(values, self._gather, out=buffer, mode="clip")
        solver._call_pardiso(22)  # noqa: SLF001

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
        """Host memory of the matrix, the factors and the gather index, in bytes.

        The gather index counts once built (at the first `refactorize`).
        """
        if self._solver is None or self._matrix is None:
            return 0
        factors = host_bytes(self._solver)
        params = getattr(self._solver, "_params", None)
        if not factors and params is not None:
            # NGSolve's PARDISO wrapper reports no memory: take PARDISO's own
            # permanent (iparm[15]) and factor (iparm[16]) memory, in KiB
            factors = 1024 * (int(params[15]) + int(params[16]))
        gather = 0 if self._gather is None else self._gather.nbytes
        return host_bytes(self._matrix) + factors + gather

    def _release(self) -> None:
        """Drop the gather index; under the lock."""
        self._gather = None
        self._pattern = None
