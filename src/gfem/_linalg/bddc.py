"""The ``bddc_gmres`` backend: NGSolve BDDC preconditioner and a Krylov solver.

BDDC splits each element's DOFs into a wirebasket (vertices and low-order
edges) and the rest, and solves a global coarse problem on the wirebasket.
That coarse problem is factorized with an explicit ``sparsecholesky``
inverse, never NGSolve's UMFPACK default. The Krylov method follows the
operator: conjugate gradients for a Hermitian (real symmetric positive
definite) operator, GMRES otherwise.
"""

from __future__ import annotations

import ngsolve as ngs
import numpy as np

from gfem._linalg import Operator, Symmetry
from gfem._linalg.base import NgsHandle, host_bytes
from gfem._linalg.direct import INVERSE

#: The coarse (wirebasket) problem's solver: always a direct factorization.
COARSE = {"coarsetype": "direct", "inverse": INVERSE}


class BddcHandle(NgsHandle):
    """One BDDC preconditioner, reused by a Krylov solve per right-hand side.

    BDDC reads the fixed DOFs from the space: the operator's free DOFs must
    be the space's ``FreeDofs()`` (build the space with ``dirichlet=``).

    Attributes:
        iterations: Krylov iterations of every column solved so far.
    """

    kind = "bddc_gmres"

    def __init__(
        self,
        operator: Operator,
        symmetry: Symmetry,
        *,
        tolerance: float,
        maxiter: int,
    ) -> None:
        """Register BDDC on the form, assemble, and set up the Krylov solver."""
        form = operator.form
        space = form.space
        free = np.array(list(operator.freedofs), dtype=bool)  # ty: ignore[invalid-argument-type]
        if not np.array_equal(free, np.array(list(space.FreeDofs()), dtype=bool)):  # ty: ignore[invalid-argument-type]
            raise ValueError(
                "BDDC fixes the Dirichlet DOFs of the space, so the operator's free "
                "DOFs must be the space's FreeDofs(); build the space with "
                "dirichlet= on the fixed boundaries."
            )
        preconditioner = ngs.Preconditioner(form, "bddc", **COARSE)
        form.Assemble()
        is_complex = bool(space.is_complex)
        super().__init__(form.mat, operator.freedofs, symmetry, is_complex=is_complex)
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.iterations: list[int] = []
        self._preconditioner = preconditioner
        if symmetry == "hermitian":
            self._solver = ngs.solvers.CGSolver(
                form.mat,
                preconditioner.mat,
                tol=tolerance,
                maxiter=maxiter,
                conjugate=is_complex,
            )
        else:
            self._solver = ngs.solvers.GMResSolver(
                form.mat, preconditioner.mat, tol=tolerance, maxiter=maxiter
            )

    def _check(self, solver) -> None:
        self.iterations.append(int(solver.iterations))
        residuals = solver.residuals
        # Krylov solvers stop early on an exact (lucky) breakdown too.
        if len(residuals) < self.maxiter or residuals[-1] <= (
            self.tolerance * residuals[0]
        ):
            return
        name = "CG" if self.symmetry == "hermitian" else "GMRES"
        raise RuntimeError(
            f"BDDC + {name} did not converge to the relative tolerance "
            f"{self.tolerance:g} in {self.maxiter} iterations (reached "
            f"{residuals[-1] / residuals[0]:.2g}). Use linear_solver='direct', or a "
            "looser tolerance."
        )

    def _inverse_map(self, B: np.ndarray, tolerance: float | None) -> np.ndarray:
        if tolerance is None or tolerance == self.tolerance:
            return self._map(self._solver, B, after=self._check)
        solver = self._solver
        if solver is None:
            raise RuntimeError("This solver handle was freed.")
        with self._lock:  # reentrant: _map takes it again
            prepared = self.tolerance
            # NGSolve's Krylov solvers read .tol on every Solve
            solver.tol = self.tolerance = tolerance  # ty: ignore[invalid-assignment]
            try:
                return self._map(solver, B, after=self._check)
            finally:
                solver.tol = self.tolerance = prepared  # ty: ignore[invalid-assignment]

    @property
    def nbytes(self) -> int:
        """Host memory of the matrix and the preconditioner, in bytes."""
        pre = self._preconditioner
        if self._solver is None or self._matrix is None or pre is None:
            return 0
        return host_bytes(self._matrix) + host_bytes(pre.mat)

    def free(self) -> None:
        """Drop the Krylov solver, the preconditioner and the matrix."""
        super().free()
        self._preconditioner = None
