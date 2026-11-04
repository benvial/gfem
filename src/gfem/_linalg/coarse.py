"""BDDC's coarse problem across ranks on PETSc MUMPS (ADRs 0069, 0078).

On rank 0 the coarse factorization holds about a third of a mesh's DOFs in one
process and every rank waits for its apply. NGSolve lets a registered ``coarsetype``
take its place, so with the MPI PETSc stack `CoarseMumps` factors the wirebasket
across ranks, with no change to BDDC: iterations and S stay those of the rank 0 route.

PT-Scotch's fill jumps below about 100 k wirebasket DOF per rank (r10M: 556 M entries
at 104 k, 4520 M at 52 k), so ``PCTELESCOPE`` in front of MUMPS runs it on at most
one rank per `DOFS_PER_RANK`, without building a communicator by hand.
"""

from __future__ import annotations

import math
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem import _mpi, _petsc
from gfem._linalg.petsc import (
    MUMPS_RETRIES,
    MUMPS_WORKSPACE_ERROR,
    RANK_MUMPS_OPTIONS,
    FreeNumbering,
    distributed_aij,
)

#: Wirebasket DOFs per coarse rank: PT-Scotch's fill is fine at 66 k and 104 k and
#: jumps at 52 k and 33 k on r0.5 and r10M; "about 100 k" errs on the fine side.
DOFS_PER_RANK = 100_000

#: Failures a larger workspace may cure: behind telescoping every failure reads 0.
RETRIED = (MUMPS_WORKSPACE_ERROR, 0)

#: Per communicator, the factorizations NGSolve asked for since the last `finish`.
_made: dict[int | None, list[CoarseMumps]] = {}


def reduction(dofs: int, size: int) -> int:
    """``PCTELESCOPE``'s factor: ``size`` ranks to at most one per `DOFS_PER_RANK`.

    Args:
        dofs: The coarse problem's free DOFs.
        size: The ranks of the communicator.

    Returns:
        1 for no reduction. PETSc keeps every factor-th rank, ``ceil(size / factor)``
        of them.
    """
    return max(1, math.ceil(size / max(1, dofs // DOFS_PER_RANK)))


class CoarseMumps(ngs.BaseMatrix):
    """The coarse factorization: a ``ParallelMatrix`` factored by MUMPS across ranks.

    NGSolve calls the creator twice per setup, first with the form's whole matrix,
    which is not assembled yet, and last with the wirebasket, so nothing is factored
    until `set_up`. The PETSc objects are destroyed by `free`, which every rank calls
    together, never by a collector that runs at another time on each rank.
    """

    def __init__(self, matrix: Any, freedofs: Any, comm: Any) -> None:
        """Hold what NGSolve gives; nothing is read from it yet."""
        super().__init__()
        self._source = matrix
        self._freedofs = freedofs
        self._comm = comm
        self._ksp: Any = None
        self._aij: Any = None
        self._x: Any = None
        self._b: Any = None
        self._freed = False

    def set_up(self) -> None:
        """Factor the wirebasket, on a new KSP while MUMPS's workspace is too small.

        Raises:
            gfem._mpi.AgreedError: The factorization failed, on every rank.
        """
        PETSc = _petsc.require_petsc()
        comm = self._comm
        with _petsc.LOCK:
            self._work = self._source.CreateColVector()
            self._numbering = FreeNumbering(
                self._source.row_pardofs, self._freedofs, self._work, comm
            )
            self._aij, _ = distributed_aij(
                self._source.local_mat, self._numbering, comm
            )
            self._x, self._b = self._aij.createVecs()
            self._factor = reduction(self._numbering.total, comm.size)
            workspace = dict(RANK_MUMPS_OPTIONS)["mat_mumps_icntl_14"]
            for attempt in range(MUMPS_RETRIES + 1):
                self._ksp = PETSc.KSP().create(comm)
                prefix = _petsc.next_prefix()
                self._ksp.setOptionsPrefix(prefix)
                self._ksp.setOperators(self._aij)
                with _petsc.scoped_options(prefix, self._options(workspace)):
                    self._ksp.setFromOptions()
                    self._ksp.setUp()
                    if self._factor > 1:
                        # the inner factorization happens at the first apply
                        self._b.set(1.0)
                        self._ksp.solve(self._b, self._x)
                failure = self._agreed_failure()
                if failure is None:
                    return
                self._ksp.destroy()
                self._ksp = None
                if failure not in RETRIED or attempt == MUMPS_RETRIES:
                    self.free()
                    raise _mpi.AgreedError(
                        "The factorization of BDDC's coarse problem failed (MUMPS "
                        f"INFOG(1)={failure or 'unread behind PCTELESCOPE'}): the "
                        "operator is singular at this frequency, or MUMPS needs more "
                        f"workspace than ICNTL(14) {workspace} gives."
                    )
                workspace *= 2

    def _options(self, workspace: int) -> tuple[tuple[str, Any], ...]:
        """PETSc's options: an exact MUMPS LDL^T, behind telescoping if it reduces."""
        factorization = (
            ("pc_type", "cholesky"),
            ("pc_factor_mat_solver_type", "mumps"),
            *{**dict(RANK_MUMPS_OPTIONS), "mat_mumps_icntl_14": workspace}.items(),
        )
        if self._factor == 1:
            return (("ksp_type", "preonly"), *factorization)
        return (
            ("ksp_type", "preonly"),
            ("pc_type", "telescope"),
            ("pc_telescope_reduction_factor", self._factor),
            ("telescope_ksp_type", "preonly"),
            *((f"telescope_{name}", value) for name, value in factorization),
        )

    def _agreed_failure(self) -> int | None:
        """``None`` if the set-up succeeded, else INFOG(1) on every rank.

        Behind ``PCTELESCOPE`` it is 0: petsc4py 3.26 has no way to the inner factor,
        so what shows there is the first apply's failed preconditioner and a solution
        that is not finite.
        """
        PETSc = _petsc.require_petsc()
        pc = self._ksp.getPC()
        failed = pc.getFailedReason() != PETSc.PC.FailedReason.NOERROR
        if self._factor > 1:
            failed = failed or not np.isfinite(self._x.array).all()
        if not _mpi.agree_any(self._comm, flag=failed):
            return None
        if self._factor > 1:
            return 0
        return _mpi.agree_min(self._comm, int(pc.getFactorMatrix().getMumpsInfog(1)))

    @property
    def nbytes(self) -> int:
        """This rank's factors, by MUMPS's own count (INFO(22)).

        0 behind ``PCTELESCOPE``, whose inner factor cannot be read
        (`_agreed_failure`): the cache's count then leaves these factors out.
        """
        if self._ksp is None or self._factor > 1:
            return 0
        return 10**6 * int(self._ksp.getPC().getFactorMatrix().getMumpsInfo(22))

    def IsComplex(self) -> bool:  # noqa: N802 - NGSolve's name
        """Only complex operators take this route (`gfem._linalg.bddc`)."""
        return True

    def Height(self) -> int:  # noqa: N802 - NGSolve's name
        """The wirebasket's local DOFs."""
        return self._source.height

    def Width(self) -> int:  # noqa: N802 - NGSolve's name
        """The wirebasket's local DOFs."""
        return self._source.width

    def CreateRowVector(self) -> Any:  # ty: ignore[invalid-method-override]  # noqa: N802
        """A vector of the wirebasket."""
        return self._source.CreateRowVector()

    def CreateColVector(self) -> Any:  # ty: ignore[invalid-method-override]  # noqa: N802
        """A vector of the wirebasket."""
        return self._source.CreateColVector()

    def Mult(self, x: Any, y: Any) -> None:  # noqa: N802 - NGSolve's name
        """``y`` = the inverse on ``x``: exchange, owned, MUMPS, owned, exchange."""
        if self._freed:
            raise RuntimeError("This coarse factorization was freed.")
        with _petsc.LOCK:
            if self._ksp is None:
                self.set_up()
            work = self._work
            work.data = x
            work.Cumulate()
            self._b.array[:] = self._numbering.state_rows(work.FV().NumPy())
            self._ksp.solve(self._b, self._x)
            y.FV().NumPy()[:] = self._numbering.state(self._x.array)
            y.SetParallelStatus(ngs.la.PARALLEL_STATUS.CUMULATED)

    def free(self) -> None:
        """Destroy the PETSc objects; every rank calls this together."""
        with _petsc.LOCK:
            self._freed = True
            for obj in (self._ksp, self._aij, self._x, self._b):
                if obj is not None:
                    obj.destroy()
            self._ksp = self._aij = self._x = self._b = None


def coarse_type(comm: Any) -> str:
    """The ``coarsetype`` flag of BDDC on ``comm``, registered on first use."""
    key = _mpi.comm_key(comm)
    name = f"gfem_mumps_{key}"
    if key not in _made:
        _made[key] = []

        def create(matrix: Any, freedofs: Any, *_: Any) -> CoarseMumps:
            made = CoarseMumps(matrix, freedofs, comm)
            _made[key].append(made)
            return made

        ngs.comp.RegisterPreconditioner(name, create)
    return name


def finish(comm: Any) -> CoarseMumps | None:
    """After a setup: the factorization it uses, factored; ``None`` on the rank 0 route.

    BDDC keeps the last one NGSolve asked for, and an earlier one was never factored.
    """
    made = _made.get(_mpi.comm_key(comm)) if comm is not None else None
    if not made:
        return None
    last = made[-1]
    made.clear()
    last.set_up()
    return last
