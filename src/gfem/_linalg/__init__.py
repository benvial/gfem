"""Every linear solve goes through one handle seam: prepare / solve / free.

``prepare(operator, symmetry=..., kind=...)`` assembles the operator's form
and sets up a backend (a factorization, a preconditioner) into a `Handle`.
``Handle.solve(B)`` solves for one or many right-hand sides, as often as
needed: terminal solves, port solves and adjoint solves reuse the same
factors. ``Handle.free()`` releases the backend's memory.

Vectors are NumPy arrays over all DOFs of the operator's space. ``solve``
reads only the free rows of ``B`` and returns zeros in the fixed (Dirichlet)
rows, so a Dirichlet lifting ``u_D`` is completed by
``u = u_D + handle.solve(-handle.apply(u_D))``.

Backends:

- ``"direct"``: NGSolve sparse Cholesky, always passed explicitly, never
  NGSolve's UMFPACK default. One factorization serves every right-hand side.
- ``"bddc_gmres"``: NGSolve BDDC with a sparse Cholesky coarse problem, and
  conjugate gradients (Hermitian operators) or GMRES (the others) to the
  relative ``tolerance``. GMRES takes an initial guess (``solve(B, x0=...)``,
  e.g. the previous frequency's solution) and then stops at ``tolerance``
  times the preconditioned right-hand side. BDDC reads the fixed DOFs from
  the space, so the operator's free DOFs must be the space's ``FreeDofs()``.
- ``"cudss"`` is not available in this version.

`factorize` gives a ``direct`` handle on a real symmetric SciPy matrix, for
auxiliary systems assembled outside NGSolve.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Protocol

import ngsolve as ngs
import numpy as np

__all__ = ["KINDS", "Handle", "Kind", "Operator", "Symmetry", "factorize", "prepare"]

Symmetry = Literal["complex_symmetric", "hermitian", "general"]
Kind = Literal["direct", "bddc_gmres", "cudss"]
KINDS: tuple[str, ...] = ("direct", "bddc_gmres")
#: Most Krylov iterations of an iterative solve (as Palace).
MAXITER = 400


@dataclasses.dataclass(frozen=True, eq=False)
class Operator:
    """A linear operator to prepare: an unassembled form and its free DOFs.

    The form is assembled by `prepare`, because some backends (BDDC) must
    see the element matrices during assembly.

    Attributes:
        form: The NGSolve bilinear form, not yet assembled. Symmetric
            symmetries need ``BilinearForm(..., symmetric=True)``.
        freedofs: The DOFs solved for; the others are fixed (Dirichlet).
    """

    form: ngs.BilinearForm
    freedofs: ngs.BitArray


class Handle(Protocol):
    """A prepared operator: solve many right-hand sides with one setup."""

    kind: str
    symmetry: Symmetry
    ndof: int

    @property
    def nbytes(self) -> int:
        """Host memory held by the matrix and the backend, in bytes."""
        ...

    def solve(
        self,
        B: np.ndarray,
        *,
        transpose: bool = False,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve ``A X = B`` on the free DOFs; ``B`` is ``(ndof,)`` or ``(ndof, k)``.

        ``x0`` (the shape of ``B``) is an initial guess for iterative
        backends; direct backends ignore it.
        """
        ...

    def apply(self, X: np.ndarray) -> np.ndarray:
        """``A X`` with the full assembled matrix (fixed rows included)."""
        ...

    def free(self) -> None:
        """Release the backend; later calls to `solve` raise."""
        ...


def prepare(
    operator: Operator,
    *,
    symmetry: Symmetry,
    kind: str,
    device: str = "cpu",
    tolerance: float = 1e-6,
    maxiter: int = MAXITER,
) -> Handle:
    """Assemble ``operator`` and set up the ``kind`` backend.

    Args:
        operator: The form and free DOFs.
        symmetry: ``"hermitian"`` (real symmetric or complex Hermitian),
            ``"complex_symmetric"`` (A^T = A) or ``"general"``.
        kind: The backend.
        device: ``"cpu"``; GPU backends are not available in this version.
        tolerance: Relative residual of iterative backends.
        maxiter: Most iterations of iterative backends; a solve that does
            not converge in them raises RuntimeError.

    Raises:
        NotImplementedError: The backend or device is not available.
        ValueError: The backend cannot handle this symmetry.
    """
    if device != "cpu":
        raise NotImplementedError(
            f"device={device!r}: GPU solves are not available in this version of "
            "gfem; use device='cpu'."
        )
    if kind == "direct":
        from gfem._linalg.direct import DirectHandle

        return DirectHandle.from_operator(operator, symmetry)
    if kind == "bddc_gmres":
        from gfem._linalg.bddc import BddcHandle

        return BddcHandle(operator, symmetry, tolerance=tolerance, maxiter=maxiter)
    raise NotImplementedError(
        f"linear_solver={kind!r} is not available in this version of gfem; use "
        f"one of {', '.join(repr(k) for k in KINDS)}."
    )


def factorize(matrix) -> Handle:
    """A ``direct`` handle on a real symmetric positive definite SciPy matrix.

    Every DOF is free. NGSolve sparse Cholesky, as for `prepare`.

    Raises:
        ValueError: ``matrix`` is not square.
    """
    from gfem._linalg.direct import DirectHandle

    return DirectHandle.from_scipy(matrix)
