"""Every linear solve goes through one handle seam: prepare / solve / free.

``prepare(operator, symmetry=..., kind=...)`` assembles the operator's form
and sets up a backend (a factorization, a preconditioner) into a `Handle`.
``Handle.solve(B)`` solves for one or many right-hand sides, as often as
needed: terminal solves, port solves and adjoint solves reuse the same
factors. ``Handle.free()`` releases the backend's memory.

A handle may also refactorize (`Refactorable`): ``handle.refactorize(operator)``
assembles another operator with the same sparsity pattern (the next
frequency of a sweep) and factorizes its values on the existing solver
plan, so a sweep plans once. ``"direct"`` and ``"cudss"`` handles can;
``"bddc_gmres"`` keeps its preconditioner for the new operator instead, caps
each warm solve on it, and sets it up again when reusing it costs more than a
setup (preconditioner reuse; ``prepare(..., reuse_preconditioner=False)`` sets
it up at every refactorization). `refactorize_or_prepare` refactorizes a
handle that can, and prepares a new one in place of any other.

An operator may also carry its piece sum (``Operator.piece_sum``, see
`gfem._linalg.pieces`): pieces assembled once and their weights, which the
``direct`` backend sums instead of assembling the form.

Vectors are NumPy arrays over all DOFs of the operator's space. ``solve``
reads only the free rows of ``B`` and returns zeros in the fixed (Dirichlet)
rows, so a Dirichlet lifting ``u_D`` is completed by
``u = u_D + handle.solve(-handle.apply(u_D))``.

Backends:

- ``"direct"``: an LDL^T factorization, always named explicitly, never
  NGSolve's default inverse: MKL PARDISO with METIS ordering when the ``mkl``
  extra is installed, NGSolve sparse Cholesky otherwise. One factorization
  serves every right-hand side.
- ``"bddc_gmres"``: NGSolve BDDC with a sparse Cholesky coarse problem, and
  conjugate gradients (Hermitian operators) or GMRES (the others) to the
  relative ``tolerance``. GMRES takes an initial guess (``solve(B, x0=...)``,
  e.g. the previous frequency's solution) and then stops at ``tolerance``
  times the preconditioned right-hand side. BDDC reads the fixed DOFs from
  the space, so the operator's free DOFs must be the space's ``FreeDofs()``.
- ``"cudss"``: the ``direct`` solver on a GPU (``device="cuda"`` or
  ``"cuda:N"``), NVIDIA cuDSS. The host holds all data; the factors stay on
  the GPU. Symmetric and Hermitian operators only. Needs the cuda extra; an
  unusable device raises `gfem.DeviceUnavailableError`, never falls back to
  the CPU.

`low_rank_update` wraps a handle of ``A`` into one of ``A + U C Uᵀ``
(Woodbury, the update never assembled): wave ports' modal corrections.

`factorize` gives a ``direct`` handle on a real symmetric SciPy matrix, for
auxiliary systems assembled outside NGSolve.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import ngsolve as ngs
import numpy as np

if TYPE_CHECKING:
    from gfem._linalg.pieces import PieceSum

__all__ = [
    "KINDS",
    "Handle",
    "Kind",
    "Operator",
    "Refactorable",
    "Symmetry",
    "factorize",
    "low_rank_update",
    "prepare",
    "refactorize_or_prepare",
]

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
        piece_sum: The same operator as a piece sum ``Σ θ_k A_k`` of pieces
            assembled once (`gfem._linalg.pieces`), or ``None``. A backend
            may sum it instead of assembling ``form`` (``"direct"`` does);
            the others assemble ``form``.
    """

    form: ngs.BilinearForm
    freedofs: ngs.BitArray
    piece_sum: PieceSum | None = None


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
        tolerance: float | None = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve ``A X = B`` on the free DOFs; ``B`` is ``(ndof,)`` or ``(ndof, k)``.

        ``tolerance`` overrides the prepared relative residual of iterative
        backends for this solve (adjoint solves); ``x0`` (the shape of ``B``)
        is an initial guess for iterative backends. Direct backends ignore
        both.
        """
        ...

    def apply(self, X: np.ndarray) -> np.ndarray:
        """``A X`` with the full assembled matrix (fixed rows included)."""
        ...

    def free(self) -> None:
        """Release the backend; later calls to `solve` raise."""
        ...


@runtime_checkable
class Refactorable(Handle, Protocol):
    """A handle that refactorizes: a new operator on the same solver plan.

    ``isinstance(handle, Refactorable)`` tells whether a handle can.
    """

    def refactorize(self, operator: Operator) -> None:
        """Assemble ``operator`` and factorize its values in place.

        A backend may sum the operator's piece sum (``operator.piece_sum``)
        instead of assembling its form. The operator has the handle's
        symmetry and free DOFs (the same space). A changed sparsity pattern
        is planned again, never refused. Later solves use the new operator.

        Raises:
            RuntimeError: The handle was freed.
        """
        ...


def refactorize_or_prepare(
    handle: Handle | None,
    operator: Operator,
    prepare_operator: Callable[[Operator], Handle],
) -> Handle:
    """A handle of ``operator`` that reuses ``handle``'s solver plan when it can.

    A `Refactorable` ``handle`` refactorizes ``operator`` and is returned;
    any other is freed, and ``prepare_operator(operator)`` is returned.

    Args:
        handle: The handle of the previous operator, or ``None``.
        operator: The new operator (the handle's symmetry and free DOFs).
        prepare_operator: Prepares a new handle (`prepare` with the
            handle's options).
    """
    if isinstance(handle, Refactorable):
        handle.refactorize(operator)
        return handle
    if handle is not None:
        handle.free()
    return prepare_operator(operator)


def prepare(
    operator: Operator,
    *,
    symmetry: Symmetry,
    kind: str,
    device: str = "cpu",
    tolerance: float = 1e-6,
    maxiter: int = MAXITER,
    width: int = 1,
    threads: int | None = None,
    reuse_preconditioner: bool = True,
) -> Handle:
    """Assemble ``operator`` and set up the ``kind`` backend.

    Args:
        operator: The form and free DOFs.
        symmetry: ``"hermitian"`` (real symmetric or complex Hermitian),
            ``"complex_symmetric"`` (A^T = A) or ``"general"``.
        kind: The backend.
        device: ``"cpu"``, or ``"cuda"`` / ``"cuda:N"`` for the ``direct``
            backend on a GPU (cuDSS). The GPU is checked before assembly.
        tolerance: Relative residual of iterative backends.
        maxiter: Most iterations of iterative backends; a solve that does
            not converge in them raises RuntimeError.
        width: Right-hand-side columns the GPU solver solves at once (the
            usual block width: terminals, ports); wider blocks go in chunks,
            narrower ones are zero-padded. CPU backends ignore it.
        threads: ``SolverOptions.threads``. ``1`` keeps the GPU solver's
            host-side planning on one thread; otherwise it is multithreaded
            when the cuDSS threading library is installed. CPU backends
            ignore it.
        reuse_preconditioner: Iterative backends: keep the preconditioner
            across `refactorize` (preconditioner reuse); False sets it up
            again at every refactorization. Direct backends ignore it.

    Raises:
        NotImplementedError: The backend is not available, or the GPU
            solver gets a general operator.
        ValueError: The backend cannot handle this symmetry, or a GPU is
            asked for an iterative backend.
        gfem.DeviceUnavailableError: The GPU cannot be used.
    """
    if device != "cpu":
        from gfem import _cuda

        if kind != "direct":
            raise ValueError(
                f"device={device!r} needs kind='direct': only the direct solver "
                f"runs on a GPU, not {kind!r}."
            )
        device_id = _cuda.require(device)
        _cuda.warn_if_jax_preallocates()
        from gfem._linalg.cudss import CudssHandle

        return CudssHandle.from_operator(
            operator, symmetry, device_id=device_id, width=width, threads=threads
        )
    if kind == "direct":
        from gfem._linalg.direct import DirectHandle

        return DirectHandle.from_operator(operator, symmetry)
    if kind == "bddc_gmres":
        from gfem._linalg.bddc import BddcHandle

        return BddcHandle(
            operator,
            symmetry,
            tolerance=tolerance,
            maxiter=maxiter,
            reuse=reuse_preconditioner,
        )
    raise NotImplementedError(
        f"linear_solver={kind!r} is not available in this version of gfem; use "
        f"one of {', '.join(repr(k) for k in KINDS)}."
    )


def factorize(matrix) -> Handle:
    """A ``direct`` handle on a real symmetric positive definite SciPy matrix.

    Every DOF is free. The ``direct`` backend's factorization, as for `prepare`.

    Raises:
        ValueError: ``matrix`` is not square.
    """
    from gfem._linalg.direct import DirectHandle

    return DirectHandle.from_scipy(matrix)


def low_rank_update(
    handle: Handle, vectors: np.ndarray, coefficients, *, borrow: bool = False
) -> Handle:
    """A handle of ``A + U C Uᵀ`` from a handle of ``A`` (Woodbury).

    See `gfem._linalg.lowrank`. The result owns ``handle`` unless it borrows
    it.

    Args:
        handle: The prepared base operator ``A``.
        vectors: ``U``, ``(ndof, p)``, zero on the fixed DOFs.
        coefficients: ``C``, ``(p, p)``.
        borrow: Leave ``handle`` to its owner: freeing the result releases
            only the update's own columns.
    """
    from gfem._linalg.lowrank import LowRankUpdate

    return LowRankUpdate(handle, vectors, coefficients, borrow=borrow)
