"""The ``direct`` backend: NGSolve sparse Cholesky (LDL^T) factorization."""

from __future__ import annotations

import numpy as np

from gfem._linalg import Operator, Symmetry

INVERSE = "sparsecholesky"


def _host_bytes(obj) -> int:
    return sum(int(u.host) for u in obj.GetMemoryUsage())


class DirectHandle:
    """One sparse Cholesky factorization, reused for every right-hand side.

    Sparse Cholesky factors ``A = L D L^T``, so it needs ``A^T = A``: a real
    symmetric operator (``"hermitian"`` on a real space) or a complex
    symmetric one. Complex Hermitian and general operators are refused.
    """

    kind = "direct"

    def __init__(self, operator: Operator, symmetry: Symmetry) -> None:
        """Assemble and factorize (see `gfem._linalg.prepare`)."""
        form = operator.form
        if symmetry == "general" or (symmetry == "hermitian" and form.space.is_complex):
            raise ValueError(
                "The direct solver (sparse Cholesky, A = L D L^T) needs A^T = A: a "
                f"real symmetric or complex symmetric operator, not {symmetry!r}."
            )
        form.Assemble()
        self.symmetry: Symmetry = symmetry
        self.ndof = int(form.space.ndof)
        self._matrix = form.mat
        self._free = np.array(list(operator.freedofs), dtype=bool)  # ty: ignore[invalid-argument-type]
        self._inverse = form.mat.Inverse(operator.freedofs, inverse=INVERSE)
        self._complex = bool(form.space.is_complex)
        self._form = form

    @property
    def nbytes(self) -> int:
        """Host memory of the matrix and the factors, in bytes."""
        if self._inverse is None:
            return 0
        return _host_bytes(self._matrix) + _host_bytes(self._inverse)

    def _columns(self, B: np.ndarray) -> tuple[np.ndarray, bool]:
        B = np.asarray(B)
        if B.shape[0] != self.ndof or B.ndim not in (1, 2):
            raise ValueError(
                f"Expected an array of shape ({self.ndof},) or ({self.ndof}, k), "
                f"got {B.shape}."
            )
        return (B[:, None], True) if B.ndim == 1 else (B, False)

    def _map(self, op, B: np.ndarray) -> np.ndarray:
        cols, flat = self._columns(B)
        dtype = np.complex128 if self._complex else np.float64
        out = np.zeros(cols.shape, dtype=dtype)
        matrix = self._matrix
        if matrix is None:
            raise RuntimeError("This solver handle was freed.")
        x = matrix.CreateColVector()
        y = matrix.CreateColVector()
        for j in range(cols.shape[1]):
            x.FV().NumPy()[:] = cols[:, j]
            y.data = op * x
            out[:, j] = y.FV().NumPy()
        return out[:, 0] if flat else out

    def solve(self, B: np.ndarray, *, transpose: bool = False) -> np.ndarray:  # noqa: ARG002
        """``A_ff X_f = B_f`` on the free DOFs; ``X`` is zero on fixed DOFs.

        ``transpose`` is free for symmetric operators (A^T = A).

        Raises:
            RuntimeError: The handle was freed.
        """
        if self._inverse is None:
            raise RuntimeError("This solver handle was freed.")
        B = np.where(self._free if np.ndim(B) == 1 else self._free[:, None], B, 0)
        if not self._complex and np.iscomplexobj(B):
            return self._map(self._inverse, B.real) + 1j * self._map(
                self._inverse, B.imag
            )
        return self._map(self._inverse, B)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """``A X`` with the full assembled matrix.

        Raises:
            RuntimeError: The handle was freed.
        """
        if self._matrix is None:
            raise RuntimeError("This solver handle was freed.")
        return self._map(self._matrix, X)

    def free(self) -> None:
        """Drop the factors and the matrix."""
        self._inverse = None
        self._matrix = None
        self._form = None
