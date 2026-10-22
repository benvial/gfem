"""The GPU direct solver: NVIDIA cuDSS through nvmath-python.

The host holds every array. The handle copies the lower triangle of the
free-DOF matrix to the GPU once, factorizes it there, and keeps the factors
on the device (or in host memory, cuDSS's hybrid memory mode, when the
factorization would not fit). Each `solve` sends the right-hand sides and
returns NumPy arrays.

cuDSS solves a fixed number of right-hand-side columns per solver object, so
the handle solves ``width`` columns at a time: wider blocks go in chunks,
narrower ones are zero-padded. All of them reuse the same factors.

`CudssHandle.refactorize` factorizes another operator with the same pattern on
the handle's solver plan: the next frequency of a sweep gathers its values
into the triangle and factorizes, without planning again.

This is the only module that imports nvmath.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from gfem._linalg import Operator, Symmetry
from gfem._linalg.base import NgsHandle, host_bytes, same_pattern

_INT32_MAX = np.iinfo(np.int32).max


class _NoResetWarning(logging.Filter):
    """Drop nvmath's warning at each reset of the matrix values.

    It says a reset needs a new factorization; `CudssHandle.refactorize`
    factorizes right after, at every frequency of a sweep.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Resetting the LHS 'a'")


#: nvmath's logger for the handle's solvers (its default is the root logger).
_LOGGER = logging.getLogger(__name__)
_LOGGER.addFilter(_NoResetWarning())


def _nvmath():
    import nvmath.sparse.advanced as nsa

    return nsa


_THREADING_LAYER = "libcudss_mtlayer_gomp.so.0"


def _nvidia_roots() -> list[Path]:
    """Directories of the ``nvidia`` namespace package, where the wheels unpack."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    return [Path(p) for p in (spec.submodule_search_locations or [])] if spec else []


def threading_layer() -> str | None:
    """The cuDSS OpenMP threading layer of the installed wheel, or ``None``.

    Looks in ``nvidia/cu13/lib`` then ``nvidia/cu12/lib`` (the two wheel
    layouts); ``None`` when neither holds it, and cuDSS plans on one thread.
    """
    for root in _nvidia_roots():
        for layout in ("cu13", "cu12"):
            path = root / layout / "lib" / _THREADING_LAYER
            if path.is_file():
                return str(path)
    return None


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


def lower_pattern(
    rowptr: np.ndarray, cols: np.ndarray, free: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The free-DOF lower triangle of a CSR pattern, and where its values are.

    ``rowptr`` and ``cols`` are NGSolve's CSR pattern (sorted columns in each
    row). Returns the triangle's int32 ``indptr`` and ``indices``, in free-DOF
    numbering, and ``gather``, the positions in the CSR data of its values:
    ``np.take(data, gather)`` is the triangle's ``data``, for any values on
    this pattern.

    Raises:
        ValueError: The triangle has more nonzeros than int32 indexes.
    """
    n = len(rowptr) - 1
    rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(rowptr.astype(np.int64)))
    keep = free[rows] & free[cols] & (cols <= rows)
    gather = np.flatnonzero(keep)
    if len(gather) > _INT32_MAX:
        raise ValueError(
            f"The matrix has {len(gather)} nonzeros in its lower triangle, more "
            "than the GPU solver's 32-bit indices allow; use device='cpu'."
        )
    renumber = (np.cumsum(free) - 1).astype(np.int32)
    indices = renumber[cols[gather]]
    nfree = int(np.count_nonzero(free))
    indptr = np.zeros(nfree + 1, dtype=np.int32)
    np.cumsum(np.bincount(renumber[rows[gather]], minlength=nfree), out=indptr[1:])
    return indptr, indices, gather


class CudssHandle(NgsHandle):
    """One cuDSS factorization on a GPU, reused for every right-hand side.

    ``"complex_symmetric"`` and real ``"hermitian"`` operators are factored
    ``A = L D L^T`` (no conjugate), complex ``"hermitian"`` ones
    ``A = L D L^H``. ``"general"`` is refused. Precision is float64 /
    complex128.

    `refactorize` factorizes another operator's values on the same plan: a
    `gfem._linalg.Refactorable` handle.
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
        threads: int | None = None,
    ) -> None:
        """Factorize an assembled NGSolve matrix on GPU ``device_id``.

        ``threads=1`` keeps cuDSS's host-side planning on one thread; otherwise
        it uses the wheel's threading layer when there is one.
        """
        if symmetry == "general":
            self._refuse()
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}.")
        super().__init__(matrix, freedofs, symmetry, is_complex=is_complex)
        self._dtype = np.complex128 if is_complex else np.float64
        self._lower, self._gather = self._triangle(matrix)
        self.width = width
        self.device_id = device_id
        self._multithreading_lib = None if threads == 1 else threading_layer()
        self.hybrid_memory = False
        self._solver = self._factorize()

    @classmethod
    def from_operator(
        cls,
        operator: Operator,
        symmetry: Symmetry,
        *,
        device_id: int,
        width: int,
        threads: int | None = None,
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
            threads=threads,
        )

    def _triangle(self, matrix) -> tuple[sp.csr_matrix, np.ndarray]:
        """The free-DOF lower triangle of ``matrix`` and its gather index."""
        data, cols, rowptr = matrix.CSR()
        indptr, indices, gather = lower_pattern(
            np.asarray(rowptr), np.asarray(cols), self._free
        )
        values = np.take(np.asarray(data), gather).astype(self._dtype, copy=False)
        lower = sp.csr_matrix((values, indices, indptr), shape=(len(indptr) - 1,) * 2)
        return lower, gather

    def refactorize(self, operator: Operator) -> None:
        """Assemble ``operator`` and refactorize (see `gfem._linalg.Refactorable`).

        On the same pattern, gathers the new values into the triangle and
        factorizes on the existing plan, in the memory mode of the first
        plan. A changed pattern is planned again (hybrid memory stays on if
        it was).
        """
        solver = self._solver
        matrix = self._matrix
        if solver is None or matrix is None:
            raise RuntimeError("This solver handle was freed.")
        form = operator.form
        form.Assemble()
        with self._lock:
            if same_pattern(matrix, form.mat):
                data = np.asarray(form.mat.CSR()[0])
                np.take(data, self._gather, out=self._lower.data)
                self._matrix = form.mat
                solver.reset_operands(a=self._lower)
                solver.factorize()
                return
            self._matrix = form.mat
            self._lower, self._gather = self._triangle(form.mat)
            self._solver = None
            solver.free()
            self._solver = self._factorize()

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
            multithreading_lib=self._multithreading_lib,
            logger=_LOGGER,
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
        solver = self._solver_object(hybrid=self.hybrid_memory)
        try:
            estimates = solver.plan().memory_estimates
            if not self.hybrid_memory and int(
                estimates.peak_device_memory
            ) > _free_device_memory(self.device_id):
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

    def _inverse_map(
        self,
        B: np.ndarray,
        tolerance: float | None,  # noqa: ARG002
        x0: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """The inverse on the free rows of ``B`` (direct: no ``tolerance``, ``x0``)."""
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
        held = 2 * csr + self._gather.nbytes
        return host_bytes(self._matrix) + held + sum(self._estimates)

    def free(self) -> None:
        """Release the GPU factors and the matrix."""
        if self._solver is not None:
            self._solver.free()
        super().free()
