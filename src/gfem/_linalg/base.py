"""What every NGSolve-backed handle shares: NumPy columns in, NumPy columns out."""

from __future__ import annotations

import threading

import ngsolve as ngs
import numpy as np

from gfem._linalg import Symmetry


def host_bytes(obj) -> int:
    """Host memory NGSolve reports for a matrix or an inverse, in bytes."""
    return sum(int(u.host) for u in obj.GetMemoryUsage())


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
        self._free = (
            np.ones(self.ndof, dtype=bool)
            if freedofs is None
            else np.array(list(freedofs), dtype=bool)  # ty: ignore[invalid-argument-type]
        )
        self._complex = is_complex
        self._solver = None
        self._lock = threading.Lock()

    def _columns(self, B: np.ndarray) -> tuple[np.ndarray, bool]:
        B = np.asarray(B)
        if B.shape[0] != self.ndof or B.ndim not in (1, 2):
            raise ValueError(
                f"Expected an array of shape ({self.ndof},) or ({self.ndof}, k), "
                f"got {B.shape}."
            )
        return (B[:, None], True) if B.ndim == 1 else (B, False)

    def _map(self, op, B: np.ndarray, after=None) -> np.ndarray:
        """``op`` applied to each column of ``B``; ``after(op)`` checks each."""
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
                    after(op)
                out[:, j] = y.FV().NumPy()
        return out[:, 0] if flat else out

    def _inverse_map(self, B: np.ndarray) -> np.ndarray:
        """Hook for subclasses: the inverse on the free rows of ``B``."""
        return self._map(self._solver, B)

    def solve(self, B: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """``A_ff X_f = B_f`` on the free DOFs; ``X`` is zero on fixed DOFs.

        ``transpose`` solves ``A^T X = B``: the same solve for a real or
        complex symmetric operator, the conjugate one for a complex
        Hermitian operator.

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
            return np.conj(self.solve(np.conj(B)))
        B = np.where(self._free if B.ndim == 1 else self._free[:, None], B, 0)
        if not self._complex and np.iscomplexobj(B):
            return self._inverse_map(B.real) + 1j * self._inverse_map(B.imag)
        return self._inverse_map(B)

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
