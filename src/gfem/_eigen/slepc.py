"""Shift-invert Krylov-Schur (SLEPc) on the eigenmode seed operator.

SLEPc's EPS runs on a petsc4py Python shell matrix that applies the shared
seed operator (`gfem._eigen.seed`): ``T(sigma)^-1`` through the eigen
operator's own linear solver (any ``linear_solver``), the divergence-free
projection on every output, and the companion form when the seed is damped.
The problem is a standard non-Hermitian one, its largest eigenvalues sought,
and the spectral transform is SLEPc's default ``shift`` at 0, so SLEPc sets
up no linear solver of its own. The start vector is the seed's fixed,
projected vector, so a solve is deterministic; no deflation space is set.

petsc4py and slepc4py come from `gfem._petsc` on the first solve; nothing is
imported at module scope.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import numpy as np

from gfem import _petsc
from gfem._eigen.operator import EigenOperator
from gfem._eigen.seed import seed_operator
from gfem.options import PetscOptions, PetscValue, split_petsc_options

__all__ = ["shift_invert"]


class _Shell:
    """The Python context of the shell matrix: ``y = apply(x)``.

    PETSc reports an exception raised in a callback as its own error; the
    exception is kept in `error`, so the solve can raise it as it was.
    """

    def __init__(self, apply: Callable[[np.ndarray], np.ndarray]) -> None:
        self._apply = apply
        self.error: BaseException | None = None

    def mult(self, mat, x, y) -> None:  # noqa: ARG002 - petsc4py passes the matrix
        try:
            y.array = self._apply(x.array_r.copy())
        except BaseException as err:
            self.error = err
            raise


def _ncv(value: PetscValue, nev: int, size: int) -> int:
    """The user's ``eps_ncv``, at least ``nev + 1`` and at most ``size``."""
    ncv = None
    if value is not None and not isinstance(value, bool):
        with contextlib.suppress(ValueError):
            ncv = int(value)
    if ncv is None:
        raise ValueError(
            f"SolverOptions.petsc_options['eps_ncv'] must be an integer, got {value!r}."
        )
    return min(size, max(ncv, nev + 1))


def shift_invert(
    op: EigenOperator,
    project,
    nev: int,
    *,
    tol: float,
    options: PetscOptions = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``nev`` eigenpairs ``(k, x)`` of ``T`` nearest ``op.shift``.

    Args:
        op: The operator with ``T(sigma)`` prepared.
        project: The divergence-free projection, ``x -> x``.
        nev: How many pairs to ask SLEPc for.
        tol: SLEPc's relative tolerance on the shift-inverted eigenvalues.
        options: ``SolverOptions.petsc_options``. Its ``eps_ncv`` sets the
            subspace size (at least ``nev + 1``); its other SLEPc options
            (`gfem.options.split_petsc_options`) reach the EPS.

    Returns:
        Wavenumbers ``(m,)`` (per file unit, ``m <= nev``; may include
        pairs with ``Re k <= 0``) and eigenvectors ``(n, m)``. If SLEPc
        stops early, the pairs it converged.

    Raises:
        gfem.SolverUnavailableError: slepc4py cannot be used.
        RuntimeError: SLEPc failed (PETSc's reason quoted). An exception
            raised by the shift solve itself is raised as it is.
    """
    SLEPc = _petsc.require_slepc()
    PETSc = _petsc.require_petsc("eigen_solver")
    seed = seed_operator(op, project)
    nev, ncv = seed.dimensions(nev)
    _, eigen = split_petsc_options(options)
    pairs = tuple(p for p in eigen if p[0] != "eps_ncv")
    if len(pairs) < len(eigen):
        ncv = _ncv(dict(eigen)["eps_ncv"], nev, seed.size)

    shell = _Shell(seed.apply)
    comm = PETSc.COMM_SELF
    with _petsc.LOCK:
        mat = PETSc.Mat().createPython([seed.size, seed.size], shell, comm=comm)
        start = PETSc.Vec().createWithArray(seed.start.copy(), comm=comm)
        vector = mat.createVecRight()
        eps = SLEPc.EPS().create(comm)
        failed = True
        try:
            mat.setUp()
            prefix = _petsc.next_prefix()
            eps.setOptionsPrefix(prefix)
            eps.setOperators(mat)
            eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
            eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
            eps.setDimensions(nev=nev, ncv=ncv)
            eps.setTolerances(tol=tol)
            with _petsc.scoped_options(prefix, pairs):
                eps.setFromOptions()
            eps.setInitialSpace([start])
            eps.solve()
            # a negative converged reason is not an error: the pairs it
            # converged, as ARPACK's early stop
            m = min(eps.getConverged(), nev)
            theta = np.empty(m, complex)
            vectors = np.empty((seed.size, m), complex)
            for i in range(m):
                theta[i] = eps.getEigenpair(i, vector)
                vectors[:, i] = vector.array_r
            failed = False
        except (PETSc.Error, SystemError) as err:
            if shell.error is not None:
                raise shell.error from None
            raise RuntimeError(
                "SLEPc's seed solve failed (SolverOptions.petsc_options = "
                f"{dict(options)}): {_petsc.error_text(err)}"
            ) from _petsc.unwrap(err)
        finally:
            for obj in (eps, vector, start, mat):
                try:
                    obj.destroy()
                except (PETSc.Error, SystemError):
                    # SLEPc cannot destroy an EPS whose solve a shell error
                    # interrupted (its DS still holds a dense submatrix):
                    # that EPS leaks, and the solve's own error is raised
                    if not failed:
                        raise
    return seed.pairs(theta, vectors)
