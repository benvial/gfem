"""The ``bddc_gmres`` backend: NGSolve BDDC preconditioner and a Krylov solver.

BDDC splits each element's DOFs into a wirebasket (vertices and low-order
edges) and the rest, and solves a global coarse problem on the wirebasket.
That coarse problem is factorized with an explicit ``sparsecholesky``
inverse, never NGSolve's UMFPACK default. The Krylov method follows the
operator: conjugate gradients for a Hermitian (real symmetric positive
definite) operator, GMRES otherwise.

Across a sweep the handle keeps its preconditioner (`BddcHandle.refactorize`)
once a solve on it measured a reference iteration count, caps every warm
solve on the reused preconditioner from that count, and sets it up again
when GMRES iterations grow past one setup's worth. A handle prepared without
reuse sets BDDC up at each `BddcHandle.refactorize`.
"""

from __future__ import annotations

import ngsolve as ngs
import numpy as np

from gfem._linalg import Operator, Symmetry
from gfem._linalg.base import NgsHandle, host_bytes
from gfem._linalg.direct import INVERSE

#: Krylov iterations a BDDC setup is worth, summed over the right-hand sides
#: of a solve: on the driven references (order 2) one setup took as long as
#: about 30 GMRES iterations of one right-hand side.
SETUP_ITERATIONS = 30


def _set_up(form: ngs.BilinearForm) -> ngs.Preconditioner:
    """Register BDDC on ``form`` and assemble it, which sets BDDC up.

    The coarse (wirebasket) problem's solver is always an explicit sparse
    Cholesky factorization. (``coarsetype="direct"`` is not set: on the
    driven H(curl) reference it made the setup 5x slower and GMRES take 68
    iterations, not 49.)
    """
    preconditioner = ngs.Preconditioner(form, "bddc", inverse=INVERSE)
    form.Assemble()
    return preconditioner


def _slack(columns: int) -> int:
    """Extra iterations per column a reused preconditioner may cost: ``Δ``."""
    return max(1, round(SETUP_ITERATIONS / columns))


class BddcHandle(NgsHandle):
    """One BDDC preconditioner, reused by a Krylov solve per right-hand side.

    BDDC reads the fixed DOFs from the space: the operator's free DOFs must
    be the space's ``FreeDofs()`` (build the space with ``dirichlet=``).

    `refactorize` swaps in another operator and keeps the preconditioner
    (preconditioner reuse across a sweep) when a solve on it measured a
    reference iteration count, setting it up again when reusing it costs
    more than a setup: a `gfem._linalg.Refactorable` handle. With
    ``reuse=False`` it sets BDDC up at every refactorization.

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
        reuse: bool = True,
    ) -> None:
        """Register BDDC on the form, assemble, and set up the Krylov solver.

        ``reuse=False`` sets BDDC up again at every `refactorize`.
        """
        form = operator.form
        space = form.space
        free = np.array(list(operator.freedofs), dtype=bool)  # ty: ignore[invalid-argument-type]
        if not np.array_equal(free, np.array(list(space.FreeDofs()), dtype=bool)):  # ty: ignore[invalid-argument-type]
            raise ValueError(
                "BDDC fixes the Dirichlet DOFs of the space, so the operator's free "
                "DOFs must be the space's FreeDofs(); build the space with "
                "dirichlet= on the fixed boundaries."
            )
        preconditioner = _set_up(form)
        is_complex = bool(space.is_complex)
        super().__init__(form.mat, operator.freedofs, symmetry, is_complex=is_complex)
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.reuse = reuse
        self.iterations: list[int] = []
        self._hold(form, preconditioner)

    def _hold(self, form: ngs.BilinearForm, preconditioner) -> None:
        """Solve ``form``'s matrix on a preconditioner just set up from it."""
        self._form = form
        self._preconditioner = preconditioner
        self._matrix = form.mat
        self._solver = self._krylov(form.mat, preconditioner)
        self._reused = False
        self._reference_iterations: int | None = None
        self._rebuild = False

    def _build(self, form: ngs.BilinearForm) -> None:
        """Set BDDC up again on ``form`` (assembled again), and hold it."""
        # release the old setup first
        self._solver = self._preconditioner = self._matrix = self._form = None
        self._hold(form, _set_up(form))

    def _held(self) -> tuple[ngs.BaseMatrix, ngs.Preconditioner, ngs.BilinearForm]:
        """The matrix, its preconditioner and its form.

        Raises:
            RuntimeError: The handle was freed.
        """
        matrix, pre, form = self._matrix, self._preconditioner, self._form
        if self._solver is None or matrix is None or pre is None or form is None:
            raise RuntimeError("This solver handle was freed.")
        return matrix, pre, form

    def _krylov(self, matrix: ngs.BaseMatrix, pre: ngs.Preconditioner):
        """The Krylov solver of ``matrix`` on the preconditioner ``pre``."""
        if self.symmetry == "hermitian":
            return ngs.solvers.CGSolver(
                matrix,
                pre.mat,
                tol=self.tolerance,
                maxiter=self.maxiter,
                conjugate=self._complex,
            )
        return self._gmres(matrix, pre, maxiter=self.maxiter)

    def _gmres(
        self,
        matrix: ngs.BaseMatrix,
        pre: ngs.Preconditioner,
        *,
        maxiter: int,
        reference: float | None = None,
    ):
        """GMRES of ``matrix`` on ``pre``, to ``tolerance`` times ``reference``.

        ``reference=None`` stops relative to the first residual instead.
        """
        if reference is None:
            return ngs.solvers.GMResSolver(
                matrix, pre.mat, tol=self.tolerance, maxiter=maxiter
            )
        return ngs.solvers.GMResSolver(
            matrix, pre.mat, atol=self.tolerance * reference, maxiter=maxiter
        )

    def refactorize(self, operator: Operator) -> None:
        """Assemble ``operator`` and solve it on the held preconditioner.

        Preconditioner reuse (see `gfem._linalg.Refactorable`): only the
        matrix is assembled, and the Krylov solver is rebuilt on it; the
        preconditioner is reused. ``n_ref`` is the most iterations per column
        of the first solve on a fresh preconditioner (cold or warm, at the
        prepared tolerance), or, after a capped column set BDDC up inside a
        solve, of the iterations spent after that setup; a solve on a reused
        preconditioner never changes it. BDDC is set up again here instead
        when the handle does not reuse, when ``n_ref`` is unset, or when a
        warm solve on the reused preconditioner, at the prepared tolerance,
        went past ``n_ref + Δ`` iterations per column, with ``Δ``
        `SETUP_ITERATIONS` shared by the solve's columns. So every warm
        solve on a reused preconditioner is capped (see `_warm`). A reused
        preconditioner serves warm GMRES solves only; a solve without an
        initial guess sets it up again first. The rule counts iterations,
        never times, so runs repeat exactly.
        """
        form = operator.form
        with self._lock:
            _, pre, _ = self._held()
            if self._rebuild or not self.reuse or self._reference_iterations is None:
                self._build(form)
                return
            form.Assemble()
            self._form = form
            self._matrix = form.mat
            self._solver = self._krylov(form.mat, pre)
            self._reused = True

    def _converged(self, solver, reference: float | None, maxiter: int) -> bool:
        """Whether ``solver`` stopped at the tolerance, not at ``maxiter``.

        ``reference`` is the residual the tolerance is relative to; ``None``
        is the first residual of the iteration.
        """
        residuals = solver.residuals
        if not residuals:
            return True
        start = residuals[0] if reference is None else reference
        # Krylov solvers stop early on an exact (lucky) breakdown too.
        return len(residuals) < maxiter or residuals[-1] <= self.tolerance * start

    def _check(
        self, solver, reference: float | None = None, iterations: int = 0
    ) -> None:
        """Record the iterations; raise if the solve stopped unconverged.

        ``reference`` is as for `_converged`; ``iterations`` were spent on
        the column before ``solver`` (a capped solve on a reused preconditioner).
        """
        self.iterations.append(iterations + int(solver.iterations))
        if self._converged(solver, reference, self.maxiter):
            return
        residuals = solver.residuals
        start = residuals[0] if reference is None else reference
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
        _, _, form = self._held()
        with self._lock:  # reentrant: _map and _warm take it again
            if self._reused and (x0 is None or self.symmetry == "hermitian"):
                # only a warm start pays for a reused preconditioner
                self._build(form)
            solver = self._solver
            prepared = self.tolerance
            # only a solve at the prepared tolerance measures n_ref
            measure = tolerance is None or tolerance == prepared
            if tolerance is not None:
                # NGSolve's Krylov solvers read .tol on every Solve
                solver.tol = self.tolerance = tolerance  # ty: ignore[invalid-assignment]
            try:
                if x0 is not None and self.symmetry != "hermitian":
                    return self._warm(B, x0, measure=measure)
                before = len(self.iterations)
                out = self._map(solver, B, after=self._check)
                # a cold solve runs on a fresh preconditioner
                self._set_reference(self.iterations[before:], measure=measure)
                return out
            finally:
                solver.tol = self.tolerance = prepared  # ty: ignore[invalid-assignment]

    def _set_reference(self, iterations: list[int], *, measure: bool) -> None:
        """Set ``n_ref`` to the most ``iterations`` of a column, when unset.

        ``iterations`` ran on a fresh preconditioner, one entry per column.
        Only a solve at the prepared tolerance (``measure``) sets ``n_ref``.
        """
        if measure and iterations and self._reference_iterations is None:
            self._reference_iterations = max(iterations)

    def _warm(self, B: np.ndarray, x0: np.ndarray, *, measure: bool) -> np.ndarray:
        """GMRES from the initial guess ``x0``, to ``tolerance * |P b|``.

        The stopping residual is relative to the preconditioned right-hand
        side, not to the first residual, so a good guess saves iterations.
        A guess whose residual exceeds ``|P b|`` (worse than zero) is
        dropped.

        On a reused preconditioner (see `refactorize`) each column stops at
        ``min(maxiter, n_ref + 2Δ)`` iterations; a column that stops there
        unconverged sets BDDC up again at this matrix and goes on from where
        it stopped. Only a solve with ``measure`` true (at the prepared
        tolerance) counts: the iterations spent on a fresh preconditioner
        set ``n_ref`` when it is unset, and those spent on the reused one
        decide the rebuild rule, when every column stayed on it.
        """
        cols, flat = self._columns(B)
        guesses, _ = self._columns(x0)
        matrix, _, _ = self._held()
        out = np.zeros(cols.shape, dtype=np.complex128 if self._complex else float)
        b, x, r = (matrix.CreateColVector() for _ in range(3))
        slack = _slack(cols.shape[1])
        stale: list[int] = []  # per column, on the reused preconditioner
        fresh: list[int] = []  # per column, after a setup at this matrix
        with self._lock:
            # refactorize reuses a preconditioner only with n_ref set
            n_ref = self._reference_iterations or 0
            for j in range(cols.shape[1]):
                matrix, pre, form = self._held()
                on_reused = self._reused
                cap = self.maxiter
                if on_reused:
                    cap = min(cap, n_ref + 2 * slack)
                b.FV().NumPy()[:] = cols[:, j]
                r.data = pre.mat * b
                reference = ngs.Norm(r)
                x.FV().NumPy()[:] = guesses[:, j]
                r.data = b - matrix * x
                b_res = r.CreateVector()
                b_res.data = pre.mat * r
                if ngs.Norm(b_res) >= reference:
                    x[:] = 0
                solver = self._gmres(matrix, pre, maxiter=cap, reference=reference)
                solver.Solve(rhs=b, sol=x, initialize=False)
                spent = 0
                if on_reused and not self._converged(solver, reference, cap):
                    # set up at this matrix, and go on from the partial iterate
                    spent = int(solver.iterations)
                    self._build(form)
                    matrix, pre, _ = self._held()
                    r.data = pre.mat * b
                    reference = ngs.Norm(r)
                    solver = self._gmres(
                        matrix, pre, maxiter=self.maxiter, reference=reference
                    )
                    solver.Solve(rhs=b, sol=x, initialize=False)
                    on_reused = False
                self._check(solver, reference, spent)
                (stale if on_reused else fresh).append(int(solver.iterations))
                out[:, j] = x.FV().NumPy()
            # n_ref only from a fresh preconditioner; the rebuild rule, by
            # iteration count, only when the whole solve stayed reused; both
            # only at the prepared tolerance
            self._set_reference(fresh, measure=measure)
            if measure and self._reused and max(stale) > n_ref + slack:
                self._rebuild = True
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
        self._form = None
