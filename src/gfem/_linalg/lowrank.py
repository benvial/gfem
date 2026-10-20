"""A prepared operator plus a low-rank term: ``A + U C Uᵀ`` by Woodbury.

``low_rank_update(handle, U, C)`` wraps a prepared handle of ``A`` into a
handle of ``A + U C Uᵀ`` (``U`` ``(ndof, p)`` with few columns, ``C`` ``(p,
p)``), never assembling the update: a wave port's modal correction is
dense on the port face, so it stays matrix-free, as in Palace. A solve
uses the Sherman-Morrison-Woodbury identity

    (A + U C Uᵀ)⁻¹ B = Y - Z (I + C Uᵀ Z)⁻¹ C Uᵀ Y,   Y = A⁻¹ B,  Z = A⁻¹ U,

so it costs ``p`` extra columns of the base solve, once per handle (``Z``
is kept for later solves, per direction and tolerance). The first solve
puts ``B`` and ``U`` in one call of the base handle. ``C`` need not be
invertible.

``U`` must vanish on the fixed DOFs (the update acts on the free ones, like
the base solve).
"""

from __future__ import annotations

import numpy as np

from gfem._linalg import Handle

__all__ = ["LowRankUpdate"]


class LowRankUpdate:
    """A `gfem._linalg.Handle` of ``A + U C Uᵀ`` (see the module docstring).

    Attributes:
        kind: The base handle's backend.
        symmetry: The base handle's symmetry (the update is the caller's).
        ndof: The number of DOFs.
        vectors: ``U``.
        coefficients: ``C``.
    """

    def __init__(self, handle: Handle, vectors: np.ndarray, coefficients) -> None:
        """Wrap ``handle`` (of ``A``) with the update ``U C Uᵀ``."""
        vectors = np.asarray(vectors)
        coefficients = np.asarray(coefficients)
        p = vectors.shape[1]
        if vectors.shape[0] != handle.ndof or coefficients.shape != (p, p):
            raise ValueError(
                f"Expected U of shape ({handle.ndof}, p) and C of shape (p, p), "
                f"got {vectors.shape} and {coefficients.shape}."
            )
        self.kind = handle.kind
        self.symmetry = handle.symmetry
        self.ndof = handle.ndof
        self.vectors = vectors
        self.coefficients = coefficients
        self._base: Handle | None = handle
        self._z: dict[tuple[bool, float | None], np.ndarray] = {}
        self._iterations: list[int] = []

    @property
    def iterations(self) -> list[int]:
        """Krylov iterations of every right-hand side solved so far.

        The update's own columns are not counted; empty for direct
        backends.
        """
        return list(self._iterations)

    @property
    def nbytes(self) -> int:
        """Host memory of the base handle, ``U`` and the kept ``A⁻¹ U``."""
        if self._base is None:
            return 0
        kept = sum(z.nbytes for z in self._z.values())
        return int(self._base.nbytes) + self.vectors.nbytes + kept

    def solve(
        self,
        B: np.ndarray,
        *,
        transpose: bool = False,
        tolerance: float | None = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve ``(A + U C Uᵀ) X = B`` (``transpose``: ``(Aᵀ + U Cᵀ Uᵀ) X = B``).

        As `gfem._linalg.Handle.solve`; ``x0`` is an initial guess of ``X``
        and goes to the base solve of ``Y``.
        """
        base = self._base
        if base is None:
            raise RuntimeError("This solver handle was freed.")
        B = np.asarray(B)
        flat = B.ndim == 1
        columns = B[:, None] if flat else B
        guess = None if x0 is None else np.asarray(x0).reshape(columns.shape)
        k = columns.shape[1]
        U = self.vectors
        key = (transpose, tolerance)
        before = len(getattr(base, "iterations", ()))
        if key in self._z:
            Z = self._z[key]
            Y = base.solve(columns, transpose=transpose, tolerance=tolerance, x0=guess)
        else:
            both = np.concatenate([columns, U], axis=1)
            if guess is not None:
                guess = np.concatenate([guess, np.zeros(U.shape, guess.dtype)], axis=1)
            out = base.solve(both, transpose=transpose, tolerance=tolerance, x0=guess)
            Y, Z = out[:, :k], out[:, k:]
            self._z[key] = Z
        its = list(getattr(base, "iterations", ()))[before:]
        self._iterations.extend(its[:k])
        C = self.coefficients.T if transpose else self.coefficients
        small = np.eye(C.shape[0]) + C @ (U.T @ Z)
        X = Y - Z @ np.linalg.solve(small, C @ (U.T @ Y))
        return X[:, 0] if flat else X

    def apply(self, X: np.ndarray) -> np.ndarray:
        """``(A + U C Uᵀ) X`` with the base handle's full matrix."""
        if self._base is None:
            raise RuntimeError("This solver handle was freed.")
        X = np.asarray(X)
        return self._base.apply(X) + self.vectors @ (
            self.coefficients @ (self.vectors.T @ X)
        )

    def free(self) -> None:
        """Release the base handle and the kept columns."""
        if self._base is not None:
            self._base.free()
        self._base = None
        self._z = {}
