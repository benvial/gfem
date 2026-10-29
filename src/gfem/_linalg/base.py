"""What every NGSolve-backed handle shares: NumPy columns in, NumPy columns out."""

from __future__ import annotations

import threading

import ngsolve as ngs
import numpy as np

from gfem._linalg import Symmetry


def host_bytes(obj) -> int:
    """Host memory NGSolve reports for a matrix or an inverse, in bytes."""
    return sum(int(u.host) for u in obj.GetMemoryUsage())


def same_pattern(a, b) -> bool:
    """Whether two assembled sparse matrices have one sparsity pattern.

    Compares the sizes and nonzero counts, then the CSR row pointers and
    column indices (no copies: NGSolve's CSR arrays are views).
    """
    if a.height != b.height or a.nze != b.nze:
        return False
    _, cols_a, rowptr_a = a.CSR()
    _, cols_b, rowptr_b = b.CSR()
    return np.array_equal(np.asarray(rowptr_a), np.asarray(rowptr_b)) and (
        np.array_equal(np.asarray(cols_a), np.asarray(cols_b))
    )


def free_mask(freedofs: ngs.BitArray | None, ndof: int) -> np.ndarray:
    """The free DOFs as a NumPy boolean mask; ``None`` means every DOF is free.

    No per-DOF Python loop: every bit set gives ``np.ones``; otherwise an
    NGSolve vector is set to 1 on the BitArray (in C++) and compared with 0.

    Raises:
        ValueError: ``freedofs`` does not have ``ndof`` bits.
    """
    if freedofs is not None and len(freedofs) != ndof:
        raise ValueError(
            f"The free-DOF set has {len(freedofs)} bits, the space {ndof} DOFs."
        )
    if freedofs is None or freedofs.NumSet() == ndof:
        return np.ones(ndof, dtype=bool)
    marks = ngs.la.CreateVVector(len(freedofs))
    marks[:] = 0.0
    marks[freedofs] = 1.0
    return marks.FV().NumPy() != 0.0


def same_free_dofs(a: ngs.BitArray, b: ngs.BitArray) -> bool:
    """Whether two free-DOF sets are equal, without building masks.

    The same object short-circuits; otherwise equal lengths, equal counts
    and no bit of ``a`` outside ``b`` (``a`` within ``b`` and as many bits
    means ``a == b``).
    """
    return a is b or (
        len(a) == len(b) and a.NumSet() == b.NumSet() and (a & ~b).NumSet() == 0
    )


class NgsHandle:
    """A handle on an assembled NGSolve matrix with free DOFs.

    Subclasses set ``self._solver`` (an NGSolve operator applying the
    inverse on the free DOFs) and implement `nbytes`. Calls are serialized
    by a lock: NGSolve inverses keep work buffers, and JAX may call a host
    callback from several threads.
    """

    kind: str

    def __init__(
        self,
        matrix: ngs.BaseMatrix,
        freedofs: ngs.BitArray | None,
        symmetry: Symmetry,
        *,
        is_complex: bool,
    ) -> None:
        """Hold ``matrix``; ``freedofs=None`` means every DOF is free."""
        self.symmetry: Symmetry = symmetry
        self.ndof = int(matrix.height)
        self._matrix: ngs.BaseMatrix | None = matrix
        self._free = free_mask(freedofs, self.ndof)
        self._complex = is_complex
        self._solver = None
        self._lock = threading.RLock()

    def _columns(self, B: np.ndarray) -> tuple[np.ndarray, bool]:
        B = np.asarray(B)
        if B.shape[0] != self.ndof or B.ndim not in (1, 2):
            raise ValueError(
                f"Expected an array of shape ({self.ndof},) or ({self.ndof}, k), "
                f"got {B.shape}."
            )
        return (B[:, None], True) if B.ndim == 1 else (B, False)

    def _map(self, op, B: np.ndarray, after=None) -> np.ndarray:
        """``op`` applied to each column of ``B``.

        ``after(op, x, y)`` checks each column's ``y = op * x``, and may
        change ``y`` in place.
        """
        cols, flat = self._columns(B)
        dtype = np.complex128 if self._complex else np.float64
        out = np.zeros(cols.shape, dtype=dtype)
        matrix = self._matrix
        if matrix is None:
            raise RuntimeError("This solver handle was freed.")
        x = matrix.CreateColVector()
        y = matrix.CreateColVector()
        with self._lock:
            for j in range(cols.shape[1]):
                x.FV().NumPy()[:] = cols[:, j]
                y.data = op * x
                if after is not None:
                    after(op, x, y)
                out[:, j] = y.FV().NumPy()
        return out[:, 0] if flat else out

    def _inverse_map(
        self,
        B: np.ndarray,
        tolerance: float | None,  # noqa: ARG002
        x0: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Hook for subclasses: the inverse on the free rows of ``B``.

        ``x0`` (same shape as ``B``, zero on fixed rows) is an initial guess.
        Direct backends ignore ``tolerance`` and ``x0``.
        """
        return self._map(self._solver, B)

    def solve(
        self,
        B: np.ndarray,
        *,
        transpose: bool = False,
        tolerance: float | None = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """``A_ff X_f = B_f`` on the free DOFs; ``X`` is zero on fixed DOFs.

        ``transpose`` solves ``A^T X = B``: the same solve for a real or
        complex symmetric operator, the conjugate one for a complex
        Hermitian operator. ``tolerance`` is the relative residual of this
        solve for iterative backends (an adjoint solve on the forward's
        preconditioner); ``None`` is the prepared tolerance. ``x0``, the
        shape of ``B``, is an initial guess for iterative backends; its fixed
        rows are not read. Direct backends ignore both.

        Raises:
            RuntimeError: The handle was freed.
            NotImplementedError: ``transpose`` for a general operator.
        """
        if self._solver is None:
            raise RuntimeError("This solver handle was freed.")
        B = np.asarray(B)
        conjugate = transpose and self._complex and self.symmetry == "hermitian"
        if transpose and self.symmetry == "general":
            raise NotImplementedError(
                f"The {self.kind!r} solver has no transposed solve of a general "
                "operator."
            )
        if conjugate:
            guess = None if x0 is None else np.conj(x0)
            return np.conj(self.solve(np.conj(B), tolerance=tolerance, x0=guess))
        free = self._free if B.ndim == 1 else self._free[:, None]
        B = np.where(free, B, 0)
        if x0 is not None:
            x0 = np.asarray(x0)
            if x0.shape != B.shape:
                raise ValueError(
                    f"The initial guess has shape {x0.shape}, the right-hand "
                    f"sides {B.shape}."
                )
            x0 = np.where(free, x0, 0)
        if not self._complex and np.iscomplexobj(B):
            re = None if x0 is None else x0.real
            im = None if x0 is None else x0.imag
            return self._inverse_map(B.real, tolerance, re) + 1j * self._inverse_map(
                B.imag, tolerance, im
            )
        if x0 is not None and not self._complex:
            x0 = x0.real
        return self._inverse_map(B, tolerance, x0)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """``A X`` with the full assembled matrix.

        Raises:
            RuntimeError: The handle was freed.
        """
        if self._matrix is None:
            raise RuntimeError("This solver handle was freed.")
        return self._map(self._matrix, X)

    def free(self) -> None:
        """Drop the backend and the matrix."""
        self._solver = None
        self._matrix = None
