"""The system operator of a driven solve: ``A(ω) + W(ω)`` and every handle of it.

``A(ω) = Σ θ_k(ω) A_k`` is the driven operator with the wave ports' Robin
pieces, and ``W(ω) = U C Uᵀ`` the wave ports' rank-two modal corrections
(`gfem._engine.wave_ports`), never assembled: a handle of ``A + W`` is a
Woodbury update of a handle of ``A`` (`gfem._linalg.low_rank_update`), for
forward and adjoint solves alike (``A + W`` is complex symmetric). The
``direct`` and ``petsc`` linear solvers sum ``A(ω)`` from the pieces ``A_k``
assembled once (the piece sum, `gfem._linalg.pieces`), kept until `free`.

A `SystemOperator` answers two questions: a forward sweep's right-hand
sides at each ω, warm-started, through one base handle (`SystemSweep`), and
a backward pass's adjoint loads at frequency ``n`` (`solve_adjoint`), at the
adjoint tolerance. On the CPU an adjoint solve keeps one handle per
frequency for later backward passes; on a GPU one base handle refactorized
at each frequency, so device memory holds one factorization whatever the
number of frequencies.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from gfem._engine.wave_ports import WavePortCoupling
from gfem._linalg import (
    Handle,
    Operator,
    ResolvedLinearSolver,
    low_rank_update,
    refactorize_or_prepare,
)
from gfem._linalg.pieces import AssembledPieces, PieceSum

if TYPE_CHECKING:
    from gfem._engine.driven import DrivenOperator

__all__ = ["SystemOperator", "SystemSweep"]


class SystemOperator:
    """The system operator ``A(ω) + W(ω)`` of one driven solve, and every handle of it.

    Owns the wave ports' Robin pieces, the piece sum, the modal corrections
    ``W``, the adjoint handles (one per frequency on the CPU, one
    refactorized base on a GPU) and their memory. A forward sweep's base
    handle lives in its `SystemSweep`.

    Attributes:
        operator: The driven operator, wave-port Robin pieces included.
        waves: The wave ports' couplings.
    """

    def __init__(
        self,
        operator: DrivenOperator,
        waves: Sequence[WavePortCoupling],
        linear: ResolvedLinearSolver,
        *,
        width: int,
        adjoint_tolerance: float,
    ) -> None:
        """Add each wave port's Robin piece to ``operator``.

        Args:
            operator: The driven operator.
            waves: The wave ports' couplings.
            linear: The rule's linear solver.
            width: The right-hand sides per solve (the excited ports).
            adjoint_tolerance: The relative residual of adjoint solves.
        """
        if waves:
            operator = dataclasses.replace(
                operator, pieces=[*operator.pieces, *(c.piece for c in waves)]
            )
        self.operator = operator
        self.waves = list(waves)
        self._linear = linear
        self._width = width
        self._adjoint_tolerance = adjoint_tolerance
        # assembled at the first direct prepare (the piece sum); a backward
        # pass reuses them, `free` drops them
        self._pieces = AssembledPieces(
            operator.space, [piece.integrand for piece in operator.pieces]
        )
        self._handles: dict[int, Handle] = {}
        self._adjoint_sweep: SystemSweep | None = None

    def sweep(
        self, *, solves: int = 1, reuse_preconditioner: bool = True
    ) -> SystemSweep:
        """One base handle for a sweep's frequencies (see `SystemSweep`).

        ``reuse_preconditioner=False`` sets an iterative backend's
        preconditioner up at every frequency, and ``solves`` (a uniform
        sweep's frequencies) is the solve calls the base is expected to serve
        (`gfem._linalg.ResolvedLinearSolver.prepare`).
        """
        return self._sweep(solves=solves, reuse_preconditioner=reuse_preconditioner)

    def solve_adjoint(self, n: int, omega: float, loads: np.ndarray) -> np.ndarray:
        """Solve the adjoint loads ``(ndof, q)`` of frequency ``n`` at ``ω``.

        At the adjoint tolerance; ``A + W`` is complex symmetric, so the
        adjoint solve is a forward solve.

        Raises:
            gfem.LinearSolverConvergenceError: The linear solve did not
                converge.
        """
        handle = self._adjoint_handle(n, omega)
        return handle.solve(loads, tolerance=self._adjoint_tolerance)

    @property
    def nbytes(self) -> int:
        """Host memory held by the adjoint handles, the pieces and ``W``, in bytes."""
        held = sum(int(h.nbytes) for h in self._handles.values())
        if self._adjoint_sweep is not None:  # the GPU adjoint base
            base = self._adjoint_sweep._base  # noqa: SLF001
            held += 0 if base is None else int(base.nbytes)
        held += sum(int(c.vectors.nbytes) for c in self.waves)
        return held + self._pieces.nbytes

    def free(self) -> None:
        """Release the adjoint handles and the pieces (a prepare reassembles them)."""
        for handle in self._handles.values():
            handle.free()
        self._handles = {}
        if self._adjoint_sweep is not None:
            self._adjoint_sweep.free()
        self._adjoint_sweep = None
        self._pieces.free()

    def operator_at(self, omega: float) -> Operator:
        """``A(ω)``, the operator without the modal corrections, to prepare.

        Its form, and its piece sum for the ``direct`` and ``petsc`` linear
        solvers.
        """
        op = self.operator
        weights = tuple(complex(piece.theta(omega)) for piece in op.pieces)
        return Operator(
            op.form(omega),
            op.space.FreeDofs(),
            piece_sum=PieceSum(self._pieces, weights),
        )

    def corrections(self, omega: float) -> tuple[np.ndarray, np.ndarray]:
        """``W(ω) = U C Uᵀ``: ``U`` ``(ndof, 2 n_wave)``, ``C`` block diagonal.

        Two columns per wave port, the mode's n x H with and without ``grad
        E_n`` (see `WavePortCoupling.correction`).
        """
        columns, diagonal = [], []
        for c in self.waves:
            p, d = c.correction(omega)
            columns.append(c.vectors @ p)
            diagonal.append(d)
        return np.concatenate(columns, axis=1), np.diag(np.concatenate(diagonal))

    def _sweep(self, **options: Any) -> SystemSweep:
        """A sweep whose base is prepared with ``options`` (`linear.prepare`'s)."""
        return SystemSweep(
            self.operator_at,
            functools.partial(self._system_handle, borrow=True),
            functools.partial(
                self._linear.prepare,
                symmetry="complex_symmetric",
                width=self._width,
                **options,
            ),
        )

    def _system_handle(self, base: Handle, omega: float, *, borrow: bool) -> Handle:
        """A handle of ``A(ω) + W(ω)`` from a handle ``base`` of ``A(ω)``.

        ``base`` itself without wave ports; otherwise a Woodbury update that
        owns ``base`` or ``borrow``s it.
        """
        if not self.waves:
            return base
        return low_rank_update(base, *self.corrections(omega), borrow=borrow)

    def _adjoint_handle(self, n: int, omega: float) -> Handle:
        """The adjoint handle of ``A(ω) + W(ω)`` of frequency ``n``.

        On the CPU prepared once per ``n`` and kept; on a GPU the adjoint
        base refactorized at ``ω``, valid until the next call.
        """
        if self._linear.device != "cpu":
            if self._adjoint_sweep is None:
                self._adjoint_sweep = self._sweep(tolerance=self._adjoint_tolerance)
            return self._adjoint_sweep.at(omega)
        if n not in self._handles:
            handle = self._linear.prepare(
                self.operator_at(omega),
                symmetry="complex_symmetric",
                width=self._width,
                tolerance=self._adjoint_tolerance,
            )
            self._handles[n] = self._system_handle(handle, omega, borrow=False)
        return self._handles[n]


class SystemSweep:
    """One base handle of ``A(ω)`` for a whole sweep: one solver plan.

    `at` gives a handle of the system operator ``A(ω) + W(ω)`` at each
    frequency. The first call prepares the base handle; later ones
    refactorize it at the new ω when it can, or else free it and prepare a
    new one (`gfem._linalg.refactorize_or_prepare`). An iterative base keeps
    its preconditioner across frequencies (preconditioner reuse) unless
    prepared with ``reuse_preconditioner=False``, as the adaptive sweep's
    is. Wave ports wrap the base per frequency in a borrowing Woodbury
    update. A handle from `at` is valid until the next call. `free` (or
    leaving the ``with`` block) releases the base, also when the sweep
    raises.
    """

    def __init__(
        self,
        operator_at: Callable[[float], Operator],
        system_handle: Callable[[Handle, float], Handle],
        prepare: Callable[[Operator], Handle],
    ) -> None:
        """Hold ``ω -> A(ω)``, the wrap into ``A(ω) + W(ω)`` and the base's prepare.

        The wrap borrows the base handle.
        """
        self._operator_at = operator_at
        self._system_handle = system_handle
        self._prepare = prepare
        self._base: Handle | None = None

    def at(self, omega: float) -> Handle:
        """A handle of ``A(ω) + W(ω)``, borrowing the sweep's base handle."""
        operator = self._operator_at(omega)
        self._base = refactorize_or_prepare(self._base, operator, self._prepare)
        return self._system_handle(self._base, omega)

    def solve(
        self, omega: float, rhs: np.ndarray, *, x0: np.ndarray | None = None
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Solve at ``ω``: the solution and the Krylov iterations per column."""
        return self.at(omega).solve_counted(rhs, x0=x0)

    def free(self) -> None:
        """Release the base handle."""
        if self._base is not None:
            self._base.free()
        self._base = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.free()
