"""The ``bddc_gmres`` backend: NGSolve BDDC preconditioner and a Krylov solver.

BDDC splits each element's DOFs into a wirebasket (vertices and low-order
edges) and the rest, and solves a global coarse problem on the wirebasket.
That coarse problem is factorized by an explicitly named inverse, never
NGSolve's default: gfem's flagged MKL PARDISO with the ``mkl`` extra (see
`gfem._linalg.pardiso`), ``sparsecholesky`` without it. The Krylov method
follows the operator: NGSolve's conjugate gradients for a Hermitian (real
symmetric positive definite) operator, otherwise gfem's GMRES
(`gfem._linalg.gmres`), which orthogonalizes its Krylov basis by CGS2 on
gfem's threads.

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
from gfem._linalg.base import NgsHandle, host_bytes, same_free_dofs
from gfem._linalg.direct import INVERSE
from gfem._linalg.gmres import Gmres
from gfem._linalg.pardiso import pardiso_inverse
from gfem.errors import LinearSolverConvergenceError

#: Krylov iterations a BDDC setup is worth, summed over the right-hand sides
#: of a solve: on the driven references (order 2) one setup took as long as
#: about 30 GMRES iterations of one right-hand side.
SETUP_ITERATIONS = 30


def _set_up(form: ngs.BilinearForm, symmetry: Symmetry) -> ngs.Preconditioner:
    """Register BDDC on ``form`` and assemble it, which sets BDDC up.

    The coarse (wirebasket) problem's solver is an explicitly named direct
    factorization: gfem's PARDISO (`pardiso_inverse`) when MKL is installed,
    symmetric when the operator is ``A^T = A`` (complex symmetric, or
    ``"hermitian"`` on a real space; a complex Hermitian one is not), sparse
    Cholesky otherwise. (``coarsetype="direct"`` is not set: on the driven
    H(curl) reference it made the setup 5x slower and GMRES take 68
    iterations, not 49.)
    """
    is_complex = bool(form.space.is_complex)
    inverse = pardiso_inverse(symmetry, is_complex=is_complex) or INVERSE
    preconditioner = ngs.Preconditioner(form, "bddc", inverse=inverse)
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
        if not same_free_dofs(operator.freedofs, space.FreeDofs()):
            raise ValueError(
                "BDDC fixes the Dirichlet DOFs of the space, so the operator's free "
                "DOFs must be the space's FreeDofs(); build the space with "
                "dirichlet= on the fixed boundaries."
            )
        preconditioner = _set_up(form, symmetry)
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
        self._hold(form, _set_up(form, self.symmetry))

    def _held(self) -> tuple[ngs.BaseMatrix, ngs.Preconditioner, ngs.BilinearForm]:
        """The matrix, its preconditioner and its form.

        Raises:
            RuntimeError: The handle was freed.
        """
        matrix, pre, form = self._matrix, self._preconditioner, self._form
        if self._solver is None or matrix is None or pre is None or form is None:
            raise RuntimeError("This solver handle was freed.")
        return matrix, pre, form

    def _krylov(
        self,
        matrix: ngs.BaseMatrix,
        pre: ngs.Preconditioner,
        *,
        maxiter: int | None = None,
        reference: float | None = None,
    ):
        """The Krylov solver of ``matrix`` on the preconditioner ``pre``.

        ``maxiter`` defaults to the handle's; ``reference`` is as for
        `_gmres`.
        """
        maxiter = self.maxiter if maxiter is None else maxiter
        if self.symmetry != "hermitian":
            return self._gmres(matrix, pre, maxiter=maxiter, reference=reference)
        if reference is None:
            stop = {"tol": self.tolerance}
        else:
            stop = {"atol": self.tolerance * reference}
        return ngs.solvers.CGSolver(
            matrix, pre.mat, maxiter=maxiter, conjugate=self._complex, **stop
        )

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
            return Gmres(matrix, pre.mat, tol=self.tolerance, maxiter=maxiter)
        return Gmres(matrix, pre.mat, atol=self.tolerance * reference, maxiter=maxiter)

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

    def _converged(self, solver, reference: float | None) -> bool:
        """Whether ``solver``'s last residual estimate is at the tolerance.

        ``reference`` is the residual the tolerance is relative to; ``None``
        is the first residual of the iteration. A stop at ``maxiter`` or
        before it (a breakdown) above the tolerance is not converged.
        """
        residuals = solver.residuals
        if not residuals:
            return True
        start = residuals[0] if reference is None else reference
        return residuals[-1] <= self.tolerance * start

    def _finish(
        self,
        solver,
        b: ngs.BaseVector,
        x: ngs.BaseVector,
        reference: float | None = None,
        iterations: int = 0,
    ) -> None:
        """Finish a column's solve: restart it if needed, record, raise.

        ``solver`` solved ``b`` into ``x``; a restart updates ``x`` in place.
        A Krylov solve that stops before its ``maxiter`` above the tolerance
        (CG's silent exit, or any solver's) restarts once from ``x``, with
        the iterations left, to ``tolerance`` times the same start. (gfem's
        GMRES returns the right iterate on a lucky breakdown, so it meets the
        tolerance there.) ``reference`` is as for `_converged`;
        ``iterations`` were spent on the column before ``solver`` (a capped
        solve on a reused preconditioner).
        """
        spent = iterations + int(solver.iterations)
        residuals = solver.residuals
        if not residuals:
            self.iterations.append(spent)
            return
        start = residuals[0] if reference is None else reference
        early = int(solver.iterations) < solver.maxiter
        if early and not self._converged(solver, start):
            matrix, pre, _ = self._held()
            left = solver.maxiter - solver.iterations
            solver = self._krylov(matrix, pre, maxiter=left, reference=start)
            solver.Solve(rhs=b, sol=x, initialize=False)
            spent += int(solver.iterations)
            early = int(solver.iterations) < solver.maxiter
        self.iterations.append(spent)
        if self._converged(solver, start):
            return
        name = "CG" if self.symmetry == "hermitian" else "GMRES"
        reached = f"{solver.residuals[-1] / start:.3e}"
        if early:
            what = (
                f": it stopped short of the tolerance after {spent} iterations, "
                f"restart included (reached {reached})"
            )
        else:
            what = f" in {self.maxiter} iterations (reached {reached})"
        raise LinearSolverConvergenceError(
            f"BDDC + {name} did not converge to the relative tolerance "
            f"{self.tolerance:g}{what}. Use linear_solver='direct', or a looser "
            "tolerance."
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
                out = self._map(solver, B, after=self._finish)
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
        it stopped. A breakdown above the tolerance on the reused
        preconditioner (a stop before the cap) sets BDDC up again the same
        way: that is the warm path's recovery, and only a breakdown on the
        fresh preconditioner meets the restart rule of `_finish`. Only a
        solve with ``measure`` true (at the prepared tolerance) counts: the
        iterations spent on a fresh preconditioner set ``n_ref`` when it is
        unset, and those spent on the reused one decide the rebuild rule,
        when every column stayed on it.
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
                if on_reused and not self._converged(solver, reference):
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
                self._finish(solver, b, x, reference, spent)
                (stale if on_reused else fresh).append(self.iterations[-1] - spent)
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
