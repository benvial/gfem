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

#: The coarse (wirebasket) problem's solver: always an explicit sparse Cholesky
#: factorization. (``coarsetype="direct"`` is not set: on the driven H(curl)
#: reference it made the setup 5x slower and GMRES take 68 iterations, not 49.)
COARSE = {"inverse": INVERSE}


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

    def _check(self, solver, reference: float | None = None) -> None:
        """Record the iterations; raise if the solve stopped unconverged.

        ``reference`` is the residual the tolerance is relative to; ``None``
        is the first residual of the iteration.
        """
        self.iterations.append(int(solver.iterations))
        residuals = solver.residuals
        if not residuals:
            return
        start = residuals[0] if reference is None else reference
        # Krylov solvers stop early on an exact (lucky) breakdown too.
        if len(residuals) < self.maxiter or residuals[-1] <= self.tolerance * start:
            return
        name = "CG" if self.symmetry == "hermitian" else "GMRES"
        raise RuntimeError(
            f"BDDC + {name} did not converge to the relative tolerance "
            f"{self.tolerance:g} in {self.maxiter} iterations (reached "
            f"{residuals[-1] / start:.2g}). Use linear_solver='direct', or a "
            "looser tolerance."
        )

    def _inverse_map(
        self, B: np.ndarray, tolerance: float | None, x0: np.ndarray | None = None
    ) -> np.ndarray:
        solver = self._solver
        if solver is None:
            raise RuntimeError("This solver handle was freed.")
        with self._lock:  # reentrant: _map and _warm take it again
            prepared = self.tolerance
            if tolerance is not None:
                # NGSolve's Krylov solvers read .tol on every Solve
                solver.tol = self.tolerance = tolerance  # ty: ignore[invalid-assignment]
            try:
                if x0 is not None and self.symmetry != "hermitian":
                    return self._warm(B, x0)
                return self._map(solver, B, after=self._check)
            finally:
                solver.tol = self.tolerance = prepared  # ty: ignore[invalid-assignment]

    def _warm(self, B: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """GMRES from the initial guess ``x0``, to ``tolerance * |P b|``.

        The stopping residual is relative to the preconditioned right-hand
        side, not to the first residual, so a good guess saves iterations.
        A guess whose residual exceeds ``|P b|`` (worse than zero) is
        dropped.
        """
        cols, flat = self._columns(B)
        guesses, _ = self._columns(x0)
        matrix = self._matrix
        pre = self._preconditioner
        if matrix is None or self._solver is None or pre is None:
            raise RuntimeError("This solver handle was freed.")
        out = np.zeros(cols.shape, dtype=np.complex128 if self._complex else float)
        b, x, r = (matrix.CreateColVector() for _ in range(3))
        with self._lock:
            for j in range(cols.shape[1]):
                b.FV().NumPy()[:] = cols[:, j]
                r.data = pre.mat * b
                reference = ngs.Norm(r)
                x.FV().NumPy()[:] = guesses[:, j]
                r.data = b - matrix * x
                b_res = r.CreateVector()
                b_res.data = pre.mat * r
                if ngs.Norm(b_res) >= reference:
                    x[:] = 0
                solver = ngs.solvers.GMResSolver(
                    matrix,
                    pre.mat,
                    atol=self.tolerance * reference,
                    maxiter=self.maxiter,
                )
                solver.Solve(rhs=b, sol=x, initialize=False)
                self._check(solver, reference)
                out[:, j] = x.FV().NumPy()
        return out[:, 0] if flat else out

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
