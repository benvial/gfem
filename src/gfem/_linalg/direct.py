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
#: NGSolve's MKL PARDISO wrapper and its flags: MKL's OpenMP parallel METIS
#: ordering (the same nested dissection and fill as serial METIS, with a
#: faster analysis, phase 11, at 2-4 threads), the symmetric matrix type
#: (gfem stores the full matrix, so NGSolve would otherwise factor it as
#: unsymmetric LU, with twice the fill), and no iterative refinement.
#: NGSolve's wrapper asks for up to 16 refinement steps with no
#: backward-error stop: they stagnate without lowering the residual and make
#: a solve cost about 4.5 substitutions instead of one. 0 is MKL's default
#: (two steps only after a perturbed pivot).
PARDISO = "pardiso"
PARDISO_FLAGS = {"ordering": "parallel_metis", "symmetric": True, "refinement": 0}


@functools.cache
def pardiso_available() -> bool:
    """Whether NGSolve found MKL PARDISO in a pip ``mkl`` (the ``mkl`` extra).

    Reads private attributes of NGSolve's wrapper: if they are missing (another
    NGSolve version), the direct solver stays sparse Cholesky.
    """
    try:
        from ngsolve.solvers import mkl_pardiso
    except ImportError:
        return False
    return (
        getattr(mkl_pardiso, "_pardiso", None) is not None
        and getattr(mkl_pardiso, "_mkl_free_buffers", None) is not None
    )


class DirectHandle(NgsHandle):
    """One LDL^T factorization, reused for every right-hand side.

    MKL PARDISO when the ``mkl`` extra is installed, NGSolve's sparse
    Cholesky otherwise (see the module docstring). Both factor
    ``A = L D L^T``, so they need ``A^T = A``: a real symmetric operator
    (``"hermitian"`` on a real space) or a complex symmetric one. Complex
    Hermitian and general operators are refused.

    `refactorize` factorizes another operator's values in place: a
    `gfem._linalg.Refactorable` handle. PARDISO gathers them straight into
    its wrapper's buffer and runs its numeric factorization (phase 22);
    sparse Cholesky, or a wrapper without the private attributes this reads,
    runs NGSolve's ``Update()``.
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
        self._gather = self._gather_index(matrix, self._solver)

    @staticmethod
    def _factor(matrix: ngs.BaseMatrix, freedofs: ngs.BitArray | None):
        inverse = INVERSE
        free = matrix.height if freedofs is None else freedofs.NumSet()
        if pardiso_available() and free > 0:  # PARDISO refuses an empty matrix
            inverse = PARDISO
            matrix.inverse_flags = pyngcore.Flags(PARDISO_FLAGS)  # ty: ignore[unresolved-attribute]
        if freedofs is None:
            return matrix.Inverse(inverse=inverse)
        return matrix.Inverse(freedofs, inverse=inverse)

    def _gather_index(self, matrix, solver) -> np.ndarray | None:
        """Where PARDISO's buffer reads ``matrix``'s values, or ``None``.

        NGSolve's wrapper factors the free-row, free-column upper triangle in
        CSR order (its ``get_csr``): these are that triangle's positions in
        ``matrix``'s value vector. Checked once against the buffer the wrapper
        analysed; ``None`` (refactorize by ``Update()``) on a mismatch or for
        a solver without the wrapper's ``_data_np`` and ``_call_pardiso``.
        """
        buffer = getattr(solver, "_data_np", None)
        if not isinstance(buffer, np.ndarray) or not hasattr(solver, "_call_pardiso"):
            return None
        data, cols, rowptr = matrix.CSR()
        cols = np.asarray(cols).astype(np.int64)
        rowptr = np.asarray(rowptr).astype(np.int64)
        rows = np.repeat(np.arange(matrix.height), np.diff(rowptr))
        free = self._free
        index = np.flatnonzero(free[rows] & free[cols] & (cols >= rows))
        index = index.astype(np.int32 if matrix.nze < 2**31 else np.int64)
        if index.size != buffer.size or not np.array_equal(
            np.asarray(data)[index], buffer
        ):
            return None
        return index

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

    def refactorize(self, operator: Operator) -> None:
        """Assemble ``operator`` and refactorize (see `gfem._linalg.Refactorable`).

        The new values are copied into the factored matrix (`apply` reads
        it). PARDISO gathers them into its buffer and factorizes them again
        on its analysis (phase 22), as ``Update()`` would after projecting
        the matrix and extracting its triangle once more; without the
        gather index, ``Update()``. A changed pattern is factorized anew.
        """
        matrix = self._matrix
        if self._solver is None or matrix is None:
            raise RuntimeError("This solver handle was freed.")
        form = operator.form
        form.Assemble()
        with self._lock:
            if not same_pattern(matrix, form.mat):
                self._matrix = form.mat
                self._solver = self._factor(form.mat, operator.freedofs)
                self._gather = self._gather_index(form.mat, self._solver)
                return
            matrix.AsVector().data = form.mat.AsVector()
            if self._gather is None:
                self._solver.Update()
                return
            solver = self._solver
            # the index is in range (checked at creation); "clip" writes
            # straight into the buffer, "raise" through a temporary copy
            buffer = solver._data_np  # noqa: SLF001
            values = np.asarray(form.mat.CSR()[0])
            np.take(values, self._gather, out=buffer, mode="clip")
            solver._call_pardiso(22)  # noqa: SLF001

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
        """Host memory of the matrix, the factors and the gather index, in bytes."""
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

    def free(self) -> None:
        """Drop the backend, the matrix and the gather index."""
        super().free()
        self._gather = None
