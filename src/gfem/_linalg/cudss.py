"""The GPU direct solver: NVIDIA cuDSS through nvmath-python.

The host holds every array. The handle copies the lower triangle of the
free-DOF matrix to the GPU once, factorizes it there, and keeps the factors
on the device (or in host memory, cuDSS's hybrid memory mode, when the
factorization would not fit). Each `solve` sends the right-hand sides and
returns NumPy arrays.

cuDSS solves a fixed number of right-hand-side columns per solver object, so
the handle solves ``width`` columns at a time: wider blocks go in chunks,
narrower ones are zero-padded. All of them reuse the same factors.

This is the only module that imports nvmath.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from gfem._linalg import Operator, Symmetry
from gfem._linalg.base import NgsHandle, host_bytes

_INT32_MAX = np.iinfo(np.int32).max


def _nvmath():
    import nvmath.sparse.advanced as nsa

    return nsa


def _free_device_memory(device_id: int) -> int:
    """Bytes free on GPU ``device_id`` (in its primary context)."""
    from cuda.bindings import driver

    _, device = driver.cuDeviceGet(device_id)
    _, ctx = driver.cuDevicePrimaryCtxRetain(device)
    try:
        driver.cuCtxPushCurrent(ctx)
        try:
            _, free, _ = driver.cuMemGetInfo()
        finally:
            driver.cuCtxPopCurrent()
    finally:
        driver.cuDevicePrimaryCtxRelease(device)
    return int(free)


def lower_triangle(matrix: sp.csr_matrix, free: np.ndarray) -> sp.csr_matrix:
    """The lower triangle of the free-DOF block, with int32 indices.

    Raises:
        ValueError: The triangle has more nonzeros than int32 indexes.
    """
    idx = np.flatnonzero(free)
    block = matrix[idx][:, idx]
    lower = sp.tril(block, format="csr")
    if lower.nnz > _INT32_MAX:
        raise ValueError(
            f"The matrix has {lower.nnz} nonzeros in its lower triangle, more "
            "than the GPU solver's 32-bit indices allow; use device='cpu'."
        )
    lower.sort_indices()
    lower.indices = lower.indices.astype(np.int32)
    lower.indptr = lower.indptr.astype(np.int32)
    return lower


class CudssHandle(NgsHandle):
    """One cuDSS factorization on a GPU, reused for every right-hand side.

    ``"complex_symmetric"`` and real ``"hermitian"`` operators are factored
    ``A = L D L^T`` (no conjugate), complex ``"hermitian"`` ones
    ``A = L D L^H``. ``"general"`` is refused. Precision is float64 /
    complex128.
    """

    kind = "cudss"

    def __init__(
        self,
        matrix,
        freedofs,
        symmetry: Symmetry,
        *,
        is_complex: bool,
        device_id: int = 0,
        width: int = 1,
    ) -> None:
        """Factorize an assembled NGSolve matrix on GPU ``device_id``."""
        if symmetry == "general":
            self._refuse()
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}.")
        super().__init__(matrix, freedofs, symmetry, is_complex=is_complex)
        data, cols, rowptr = matrix.CSR()
        full = sp.csr_matrix(
            (np.asarray(data), np.asarray(cols), np.asarray(rowptr)),
            shape=(self.ndof, self.ndof),
        )
        self._dtype = np.complex128 if is_complex else np.float64
        self._lower = lower_triangle(full.astype(self._dtype), self._free)
        self.width = width
        self.device_id = device_id
        self.hybrid_memory = False
        self._solver = self._factorize()

    @classmethod
    def from_operator(
        cls, operator: Operator, symmetry: Symmetry, *, device_id: int, width: int
    ) -> CudssHandle:
        """Assemble and factorize on the GPU (see `gfem._linalg.prepare`)."""
        if symmetry == "general":
            cls._refuse()  # before paying for the assembly
        form = operator.form
        form.Assemble()
        return cls(
            form.mat,
            operator.freedofs,
            symmetry,
            is_complex=bool(form.space.is_complex),
            device_id=device_id,
            width=width,
        )

    @staticmethod
    def _refuse() -> None:
        raise NotImplementedError(
            "The GPU solver (cuDSS) handles symmetric and Hermitian operators "
            "only; a general operator needs device='cpu'."
        )

    def _block(self) -> np.ndarray:
        return np.zeros(
            (self._lower.shape[0], self.width), dtype=self._dtype, order="F"
        )

    def _solver_object(self, hybrid: bool):
        nsa = _nvmath()
        types = nsa.DirectSolverMatrixType
        if self.symmetry == "hermitian" and self._complex:
            matrix_type = types.HERMITIAN
        else:
            matrix_type = types.SYMMETRIC
        options = nsa.DirectSolverOptions(
            sparse_system_type=matrix_type,
            sparse_system_view=nsa.DirectSolverMatrixViewType.LOWER,
        )
        execution = nsa.ExecutionCUDA(
            device_id=self.device_id,
            hybrid_memory_mode_options=nsa.HybridMemoryModeOptions(
                hybrid_memory_mode=hybrid
            ),
        )
        return nsa.DirectSolver(
            self._lower, self._block(), options=options, execution=execution
        )

    def _factorize(self):
        """Plan; switch to hybrid memory if the factors do not fit; factorize."""
        solver = self._solver_object(hybrid=False)
        try:
            estimates = solver.plan().memory_estimates
            if int(estimates.peak_device_memory) > _free_device_memory(self.device_id):
                solver.free()
                solver = self._solver_object(hybrid=True)
                estimates = solver.plan().memory_estimates
                self.hybrid_memory = True
            self._estimates = (
                int(estimates.permanent_device_memory),
                int(estimates.permanent_host_memory),
            )
            solver.factorize()
        except BaseException:
            solver.free()
            raise
        return solver

    def _inverse_map(self, B: np.ndarray) -> np.ndarray:
        cols, flat = self._columns(B)
        solver = self._solver
        if solver is None:
            raise RuntimeError("This solver handle was freed.")
        free = self._free
        rhs = cols[free]
        out = np.zeros(cols.shape, dtype=self._dtype)
        with self._lock:
            for start in range(0, rhs.shape[1], self.width):
                chunk = rhs[:, start : start + self.width]
                block = self._block()
                block[:, : chunk.shape[1]] = chunk
                solver.reset_operands(b=block)
                x = np.asarray(solver.solve())
                out[free, start : start + chunk.shape[1]] = x.reshape(block.shape)[
                    :, : chunk.shape[1]
                ]
        return out[:, 0] if flat else out

    @property
    def nbytes(self) -> int:
        """Host and device memory of the matrix and the factors, in bytes."""
        if self._solver is None or self._matrix is None:
            return 0
        lower = self._lower
        csr = lower.data.nbytes + lower.indices.nbytes + lower.indptr.nbytes
        return host_bytes(self._matrix) + 2 * csr + sum(self._estimates)

    def free(self) -> None:
        """Release the GPU factors and the matrix."""
        if self._solver is not None:
            self._solver.free()
        super().free()
