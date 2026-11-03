"""The PETSc linear solver: a PETSc KSP / PC on the free block, MUMPS by default.

The handle keeps the full NGSolve matrix (`apply` and the piece sum use it)
and, beside it, the free-row, free-column block as a PETSc AIJ matrix built
over NumPy arrays. A refactorization gathers the new values into those
arrays in place and marks them (``Mat.assemble()``), so a same-pattern
refactorization is numeric only, on one solver plan.

By default it is exact: ``ksp_type preonly``, ``pc_type cholesky``,
``pc_factor_mat_solver_type mumps``, MUMPS LDL^T on a matrix flagged
symmetric. Those defaults, ``ksp_max_it`` and ``ksp_rtol`` are entries of
PETSc's options database under the handle's own prefix, so a user's
``petsc_options`` override them; they are in the database only while the
handle reads them (`gfem._petsc.scoped_options`). A user option PETSc did
not read raises.

With ``ksp_reuse_preconditioner`` a refactorization keeps the preconditioner
set up at an earlier operator. Where a Krylov solve on it does not converge,
the handle sets it up once at the current operator and solves that column
again from the partial iterate; only a solve on a preconditioner set up at
the current operator raises `gfem.LinearSolverConvergenceError`.

Across ranks `RankPetscHandle` holds the free block as an ``MPIAIJ`` filled through
COO (ADR 0062), and is also what ``kind="direct"`` means there: MUMPS LDL^T on
PT-Scotch, with retries on a new KSP while MUMPS runs out of workspace (ADR 0068).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import ngsolve as ngs
import numpy as np

from gfem import _mpi, _petsc
from gfem._linalg import MAXITER, Symmetry
from gfem._linalg.base import (
    PlannedHandle,
    equals_its_transpose,
    free_mask,
    host_bytes,
    local_matrix,
)
from gfem.errors import LinearSolverConvergenceError

if TYPE_CHECKING:
    from gfem._linalg.layout import RankLayout
    from gfem.options import PetscOptions

#: gfem's options, under the user's: an exact MUMPS LDL^T factorization.
DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("ksp_type", "preonly"),
    ("pc_type", "cholesky"),
    ("pc_factor_mat_solver_type", "mumps"),
)

#: Added to the error of a Krylov solve that met an indefinite operator.
CG_HINT = (
    "CG needs a Hermitian positive definite operator; gfem's driven and eigen "
    "operators are complex symmetric, so use ksp_type gmres."
)
#: The preconditioners that factorize: their factors count in `nbytes`.
FACTORIZATIONS = ("cholesky", "lu", "icc", "ilu")

#: gfem's MUMPS options under ranks, under the user's. PT-Scotch: METIS's analysis is
#: sequential. ICNTL(14) 60: the default gave INFOG(1)=-9 on tiny meshes at 4 and 8
#: ranks and in 2 of 6 runs at 224 ranks, and 60 in none of 35 (ADR 0068).
RANK_MUMPS_OPTIONS: tuple[tuple[str, Any], ...] = (
    ("mat_mumps_icntl_28", 2),
    ("mat_mumps_icntl_29", 1),
    ("mat_mumps_icntl_14", 60),
)
#: INFOG(1) when MUMPS's main workspace is too small.
MUMPS_WORKSPACE_ERROR = -9
#: New KSPs on a too small workspace, each with ICNTL(14) doubled.
MUMPS_RETRIES = 2


class _Block(NamedTuple):
    """A free block as a PETSc AIJ over its own arrays (`PetscHandle._block`)."""

    aij: Any
    rowptr: np.ndarray
    cols: np.ndarray
    #: The free block's entries in NGSolve's value vector.
    index: np.ndarray
    values: np.ndarray


class AgreedValueError(_mpi.AgreedError, ValueError):
    """Every rank found the same bad option: a `ValueError` the job can catch."""


class FreeNumbering:
    """A matrix's free DOFs across ranks, in PETSc's ownership ranges (ADR 0062).

    ``EnumerateGlobally`` numbers every free DOF, the same on each rank that holds it;
    a rank's owned numbers are contiguous and start at the sum of the earlier ranks'
    counts, which is observed and not documented, so it is checked here. Moving a
    column costs one neighbour exchange each way.

    Attributes:
        numbers: Per local DOF, its number, or a negative value on a fixed one.
        total: The free DOFs over all ranks.
        count: This rank's owned free DOFs.
    """

    def __init__(self, pardofs: Any, freedofs: Any, work: Any, comm: Any) -> None:
        """Number ``freedofs``; ``work`` is a column vector of the matrix, for scratch.

        Raises:
            gfem._mpi.AgreedError: The owned numbers are not PETSc's ranges.
        """
        numbers, total = pardofs.EnumerateGlobally(freedofs)
        self.numbers = np.asarray(numbers, dtype=np.int64)
        self.total = int(total)
        master = free_mask(pardofs.MasterDofs(), len(self.numbers))
        self._owned = master & (self.numbers >= 0)
        self.count = int(self._owned.sum())
        start = comm.exscan(self.count) or 0
        owned = self.numbers[self._owned]
        self._position = owned - start
        ranges = np.array_equal(self._position, np.arange(self.count))
        if not _mpi.agree_all(comm, flag=ranges):
            raise _mpi.AgreedError(
                "NGSolve's owned free DOF numbers are not contiguous on each rank, "
                "so they are not PETSc's ownership ranges."
            )
        self._work = work

    def load_rows(self, load: np.ndarray) -> np.ndarray:
        """A load column's owned rows: its summed values, one neighbour exchange."""
        fv = self._work.FV().NumPy()
        fv[:] = load
        self._work.SetParallelStatus(ngs.la.PARALLEL_STATUS.DISTRIBUTED)
        self._work.Cumulate()
        return self.state_rows(fv)

    def state_rows(self, state: np.ndarray) -> np.ndarray:
        """A state column's owned rows, each rank's in PETSc's order."""
        rows = np.zeros(self.count, dtype=complex)
        rows[self._position] = state[self._owned]
        return rows

    def state(self, rows: np.ndarray) -> np.ndarray:
        """The state column of owned rows: zero on fixed DOFs, one neighbour exchange."""
        fv = self._work.FV().NumPy()
        fv[:] = 0
        values = rows[self._position]
        fv[self._owned] = values if np.iscomplexobj(fv) else values.real
        self._work.SetParallelStatus(ngs.la.PARALLEL_STATUS.DISTRIBUTED)
        self._work.Cumulate()
        return np.array(fv)  # a copy: the next column writes here


def distributed_aij(
    local: Any, numbering: FreeNumbering, comm: Any
) -> tuple[Any, np.ndarray]:
    """A rank's unassembled ``local`` matrix as a symmetric ``MPIAIJ`` by COO.

    Returns the matrix and the positions of its entries in ``local``'s value vector,
    for `set_values`.
    """
    PETSc = _petsc.require_petsc()
    data, cols, rowptr = local.CSR()
    cols = np.asarray(cols).astype(np.int64)
    rowptr = np.asarray(rowptr).astype(np.int64)
    rows = np.repeat(np.arange(local.height), np.diff(rowptr))
    numbers = numbering.numbers
    index = np.flatnonzero((numbers[rows] >= 0) & (numbers[cols] >= 0))
    size = (numbering.count, numbering.total)
    aij = PETSc.Mat().createAIJ([size, size], comm=comm)
    try:
        aij.setPreallocationCOO(
            numbers[rows[index]].astype(PETSc.IntType),
            numbers[cols[index]].astype(PETSc.IntType),
        )
        set_values(aij, data, index)
        aij.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        aij.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
    except BaseException:
        aij.destroy()
        raise
    return aij, index


def set_values(aij: Any, data: Any, index: np.ndarray) -> None:
    """New values, on the same pattern, from a local matrix's value vector.

    ``INSERT_VALUES`` zeroes and sums the COO duplicates. ``ADD_VALUES`` would add to
    the previous frequency's values and give a wrong answer with no error.
    """
    PETSc = _petsc.require_petsc()
    values = np.asarray(data)[index].astype(PETSc.ScalarType)
    aij.setValuesCOO(values, addv=PETSc.InsertMode.INSERT_VALUES)


class PetscHandle(PlannedHandle):
    """A PETSc KSP / PC on the free block of the operator.

    Accepts ``A^T = A``: a complex symmetric operator, or a real symmetric
    one (``"hermitian"`` on a real space), solved in complex arithmetic.
    A column solved again on a preconditioner set up anew counts the
    iterations of both solves (none for ``ksp_type preonly``).
    """

    kind = "petsc"
    sums_pieces = True

    def __init__(
        self,
        matrix: ngs.BaseMatrix,
        freedofs: ngs.BitArray,
        symmetry: Symmetry,
        *,
        is_complex: bool,
        tolerance: float,
        maxiter: int = MAXITER,
        options: PetscOptions = (),
        reuse: bool = True,
    ) -> None:
        """Set the KSP / PC up on an assembled matrix (see `prepare`).

        ``options`` are the user's, read over gfem's defaults; ``reuse=False``
        sets the preconditioner up again at every `refactorize`. ``symmetry``
        is refused before (`from_operator`).
        """
        _refuse_storage(local_matrix(matrix, self.layout.comm))
        super().__init__(matrix, freedofs, symmetry, is_complex=is_complex)
        self._lock = _petsc.LOCK  # PETSc's database and registry are global
        #: The KSP and the AIJ (petsc4py objects); ``None`` once freed.
        self._ksp: Any = None
        self._mat: Any = None
        #: ``ksp_reuse_preconditioner`` holds (petsc4py has no getter for it):
        #: a refactorization keeps the preconditioner set up earlier
        self._reuses = False
        #: The preconditioner was set up at the current operator.
        self._pc_is_current = True
        _petsc.require_petsc("linear_solver")
        self._prefix = _petsc.next_prefix()
        # plain numbers: the database holds strings, and a NumPy scalar's is
        # its repr (``np.float64(1e-08)``), which PETSc refuses
        defaults = (
            *self._defaults(),
            ("ksp_max_it", int(maxiter)),
            ("ksp_rtol", float(tolerance)),
        )
        self._options = tuple({**dict(defaults), **dict(options)}.items())
        self._user = dict(options)
        self._reuse = reuse
        with self._lock:
            try:
                self._adopt(self._block(matrix))
                self._new_ksp()
                with _petsc.scoped_options(self._prefix, self._options):
                    self._configure(self._user, reuse=reuse)
                    self._set_up_now()
                    self._check_read([name for name, _ in options])
            except BaseException:
                self.free()
                raise

    def _defaults(self) -> tuple[tuple[str, Any], ...]:
        """gfem's options on this communicator, under the user's."""
        return DEFAULTS

    def _new_ksp(self) -> None:
        """A KSP on the held AIJ under this handle's prefix, not configured."""
        PETSc = _petsc.require_petsc()
        ksp = PETSc.KSP().create(self._petsc_comm())
        self._solver = self._ksp = ksp
        ksp.setOptionsPrefix(self._prefix)
        ksp.setOperators(self._mat)

    def _petsc_comm(self) -> Any:
        """The communicator of the handle's PETSc objects."""
        return _petsc.require_petsc().COMM_SELF

    def _configure(self, user: dict[str, Any], *, reuse: bool) -> None:
        """Read the options into the KSP, then set what gfem decides.

        A Krylov method measures the true (unpreconditioned) residual unless
        the user names ``ksp_norm_type``.
        """
        PETSc = _petsc.require_petsc()
        ksp = self._ksp
        try:
            ksp.setFromOptions()
        except PETSc.Error as err:
            raise self._bad_option(
                "SolverOptions.petsc_options was refused by PETSc: "
                f"{_petsc.error_text(err)}."
            ) from err
        self._direct = ksp.getType() == "preonly"
        name = "ksp_reuse_preconditioner"
        asked = PETSc.Options().getBool(f"{self._prefix}{name}", False)
        if self._direct and name in user and asked:
            raise self._bad_option(
                f"SolverOptions.petsc_options[{name!r}] with ksp_type preonly would "
                "solve the previous operator after each refactorization; use a "
                "Krylov ksp_type with it."
            )
        if not self._direct and "ksp_norm_type" not in user:
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        if not reuse:
            ksp.getPC().setReusePreconditioner(False)
        self._reuses = reuse and not self._direct and asked

    def _check_read(self, names: list[str]) -> None:
        """Raise if PETSc did not read some of the user's ``names``.

        Call after the first set-up, with the options still in the database.
        """
        unread = _petsc.unread(self._prefix, names)
        if _mpi.agree_all(self.layout.comm, flag=not unread):
            return
        keys = ", ".join(f"SolverOptions.petsc_options[{n!r}]" for n in unread)
        verb = "was" if len(unread) == 1 else "were"
        raise self._bad_option(
            f"{keys} {verb} not read by PETSc's linear solver ({self._describe()}): "
            "check the name, or remove options this configuration does not use."
        )

    def _bad_option(self, message: str) -> ValueError:
        """The error of options PETSc refused."""
        return ValueError(message)

    def _describe(self) -> str:
        """The configuration, e.g. ``KSP preonly, PC cholesky, MUMPS``."""
        pc = self._ksp.getPC()
        text = f"KSP {self._ksp.getType()}, PC {pc.getType()}"
        if pc.getType() in FACTORIZATIONS:
            text += f", {pc.getFactorSolverType().upper()}"
        return text

    @classmethod
    def _admit(cls, symmetry: Symmetry, *, is_complex: bool) -> None:
        _refuse(symmetry, is_complex=is_complex)

    def _new_values(self, write: Callable[[ngs.BaseMatrix], None]) -> None:
        """Write the held matrix and set the KSP up on it; under the lock.

        The free block's values are gathered into the AIJ's arrays and set
        up again on the same solver plan (a numeric factorization by
        default).
        """
        matrix = self._live()
        write(matrix)
        self._refresh(matrix)
        self._set_up()
        self._pc_is_current = not self._reuses  # a reused preconditioner was kept

    def _refresh(self, matrix: ngs.BaseMatrix) -> None:
        """Move the held matrix's new values to the AIJ; under the lock."""
        self._gather(matrix.CSR()[0], self._index, self._values)
        self._mat.assemble()  # marks the new values: PETSc sets up again

    def _new_plan(
        self,
        matrix: ngs.BaseMatrix,
        freedofs: ngs.BitArray | None,  # noqa: ARG002
    ) -> None:
        """A new AIJ of ``matrix`` under the same KSP, set up; under the lock."""
        # the new plan is built aside: a failure leaves the handle as it was
        block, old = self._block(matrix), self._mat
        try:
            self._ksp.setOperators(block.aij)
        except BaseException:
            block.aij.destroy()
            raise
        self._matrix = matrix
        self._adopt(block)
        try:
            self._set_up()  # symbolic and numeric, with the options read
        finally:
            old.destroy()  # the KSP let go of it at setOperators
        self._pc_is_current = not self._reuses

    def _set_up_anew(self) -> None:
        """Set a reused preconditioner up at the current operator; under the lock.

        The preconditioner set up here is the one reused afterwards.
        """
        pc = self._ksp.getPC()
        pc.setReusePreconditioner(False)
        try:
            self._set_up()
        finally:
            # only called while ``_reuses`` holds, so reuse was on before
            pc.setReusePreconditioner(True)
        self._pc_is_current = True

    def _set_up(self) -> None:
        """`_set_up_now` with the handle's options in PETSc's database."""
        with _petsc.scoped_options(self._prefix, self._options):
            self._set_up_now()

    def _set_up_now(self) -> None:
        """``KSPSetUp``; raise if it, or the preconditioner's set-up, failed."""
        PETSc = _petsc.require_petsc()
        try:
            self._ksp.setUp()
        except PETSc.Error as err:
            raise self._failed("KSPSetUp", err) from err
        if self._ksp.getPC().getFailedReason() != PETSc.PC.FailedReason.NOERROR:
            raise self._factorization_failed()

    def _failed(self, call: str, err: Exception) -> RuntimeError:
        return RuntimeError(
            f"PETSc failed in {call} ({self._describe()}): {_petsc.error_text(err)}."
        )

    def _factorization_failed(self) -> RuntimeError:
        """The error of a failed preconditioner set-up (a singular operator)."""
        PETSc = _petsc.require_petsc()
        pc = self._ksp.getPC()
        reason = _name(PETSc.PC.FailedReason, pc.getFailedReason())
        if reason == "NOERROR":  # failed at the solve, after a good set-up
            reason = _name(PETSc.KSP.ConvergedReason, self._ksp.getConvergedReason())
        if pc.getType() in FACTORIZATIONS and pc.getFactorSolverType() == "mumps":
            info = pc.getFactorMatrix().getMumpsInfog(1)
            reason += f", MUMPS INFOG(1)={info}"
        return RuntimeError(
            f"PETSc's factorization failed ({reason}): the operator is singular "
            "at this frequency or shift, or MUMPS needs more workspace "
            "(petsc_options mat_mumps_icntl_14)."
        )

    def _block(self, matrix) -> _Block:
        """The free block of ``matrix`` as a new AIJ over new arrays.

        The handle is left as it is (`_adopt` takes the block); a failed
        creation destroys what it created.
        """
        PETSc = _petsc.require_petsc()
        data, cols, rowptr = matrix.CSR()
        cols = np.asarray(cols).astype(np.int64)
        rowptr = np.asarray(rowptr).astype(np.int64)
        rows = np.repeat(np.arange(matrix.height), np.diff(rowptr))
        free = self._free
        index = np.flatnonzero(free[rows] & free[cols])
        number = np.cumsum(free) - 1  # a free DOF's row in the free block
        n = int(free.sum())
        counts = np.bincount(number[rows[index]], minlength=n)
        rowptr = np.concatenate([[0], np.cumsum(counts)]).astype(PETSc.IntType)
        block_cols = number[cols[index]].astype(PETSc.IntType)
        values = np.zeros(index.size, dtype=PETSc.ScalarType)
        self._gather(data, index, values)
        aij = PETSc.Mat().createAIJWithArrays(
            (n, n), (rowptr, block_cols, values), comm=PETSc.COMM_SELF
        )
        try:
            aij.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            aij.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
        except BaseException:
            aij.destroy()
            raise
        return _Block(aij, rowptr, block_cols, index, values)

    def _adopt(self, block: _Block) -> None:
        """Hold ``block``: its AIJ, and the arrays the AIJ reads."""
        self._mat = block.aij
        self._rowptr, self._cols = block.rowptr, block.cols
        self._index, self._values = block.index, block.values

    def _gather(self, data, index: np.ndarray, values: np.ndarray) -> None:
        """Write into ``values`` the free block's entries of NGSolve's ``data``."""
        entries = np.asarray(data)[index]
        if self._complex:
            values[:] = entries
        else:
            values.real[:] = entries

    def _inverse_map(
        self, B: np.ndarray, tolerance: float | None, x0: np.ndarray | None = None
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """``A_ff^{-1} B_f`` on the free rows, zero on the fixed ones.

        ``preonly`` solves every column at once and ignores ``tolerance``
        and ``x0``; a Krylov method solves column by column, from ``x0``
        when given, to ``tolerance`` (the prepared one when ``None``).
        Returns the iterations of each column too, ``()`` for ``preonly``.
        """
        cols, flat = self._columns(B)
        with self._lock:
            self._live()
            rhs = self._load_rows(cols)
            guesses = None if x0 is None else self._state_rows(self._columns(x0)[0])
            counts: tuple[int, ...] = ()
            if self._direct:
                solved = self._exact(rhs)
            else:
                solved, counts = self._krylov(rhs, tolerance, guesses)
            out = self._states(solved, cols.shape)
        result = out if self._complex else out.real
        return (result[:, 0] if flat else result), counts

    def _load_rows(self, cols: np.ndarray) -> np.ndarray:
        """Load columns as the AIJ's rows."""
        return np.ascontiguousarray(cols[self._free], dtype=complex)

    def _state_rows(self, cols: np.ndarray) -> np.ndarray:
        """State columns as the AIJ's rows."""
        return cols[self._free]

    def _states(self, rows: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """The AIJ's rows as state columns of ``shape``, zero on the fixed DOFs."""
        out = np.zeros(shape, dtype=complex)
        out[self._free] = rows
        return out

    def _dense_size(self, rows: int, columns: int) -> Any:
        """A dense block's size for ``matSolve``, ``rows`` of them on this process."""
        return (rows, columns)

    def _exact(self, rhs: np.ndarray) -> np.ndarray:
        """``preonly``: one solve, or one block solve (``KSPMatSolve``)."""
        PETSc = _petsc.require_petsc()
        ksp = self._ksp
        n, k = rhs.shape
        try:
            if k == 1:
                x, b = self._mat.createVecs()
                b.array[:] = rhs[:, 0]
                try:
                    ksp.solve(b, x)
                    X = x.array.copy()[:, None]
                finally:
                    x.destroy()
                    b.destroy()
            else:
                columns = np.asfortranarray(rhs)  # PETSc's dense storage
                comm = self._petsc_comm()
                size = self._dense_size(n, k)
                Bm = PETSc.Mat().createDense(size, array=columns, comm=comm)
                Xm = PETSc.Mat().createDense(size, comm=comm)
                try:
                    ksp.matSolve(Bm, Xm)
                    # an empty rank's array has no second axis to keep
                    X = Xm.getDenseArray().reshape(rhs.shape, order="F").copy()
                finally:
                    Bm.destroy()
                    Xm.destroy()
        except PETSc.Error as err:
            raise self._failed("KSPSolve", err) from err
        if self._diverged():
            raise self._factorization_failed()
        return X

    def _krylov(
        self, rhs: np.ndarray, tolerance: float | None, guesses: np.ndarray | None
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """A Krylov solve per column; the solutions and each column's iterations.

        A column that does not converge on a reused preconditioner set up at
        an earlier operator is solved again, from its partial iterate, once
        the preconditioner is set up at this one (`_set_up_anew`).
        """
        ksp = self._ksp
        prepared = ksp.getTolerances()
        _, atol, divtol, max_it = prepared
        if tolerance is not None:
            ksp.setTolerances(rtol=tolerance, atol=atol, divtol=divtol, max_it=max_it)
        X = np.zeros_like(rhs)
        counts: list[int] = []
        x, b = self._mat.createVecs()
        try:
            for j in range(rhs.shape[1]):
                b.array[:] = rhs[:, j]
                ksp.setInitialGuessNonzero(guesses is not None)
                if guesses is not None:
                    x.array[:] = guesses[:, j]
                spent = self._solve(b, x)
                if self._diverged() and not self._pc_is_current:
                    self._set_up_anew()
                    finite = bool(np.isfinite(x.array).all())
                    if not _mpi.agree_all(self.layout.comm, flag=finite):  # start again
                        ksp.setInitialGuessNonzero(guesses is not None)
                        x.array[:] = 0 if guesses is None else guesses[:, j]
                    else:
                        ksp.setInitialGuessNonzero(True)
                    spent += self._solve(b, x)
                if self._diverged():
                    raise self._not_converged(b.norm(), spent)
                counts.append(spent)
                X[:, j] = x.array
        finally:
            x.destroy()
            b.destroy()
            ksp.setTolerances(*prepared)
        return X, tuple(counts)

    def _diverged(self) -> bool:
        """Whether the last solve diverged, the same answer on every rank."""
        return _mpi.agree_any(
            self.layout.comm, flag=self._ksp.getConvergedReason() < 0
        )

    def _solve(self, b, x) -> int:
        """``KSPSolve`` of ``b`` into ``x``; its iterations."""
        PETSc = _petsc.require_petsc()
        try:
            self._ksp.solve(b, x)
        except PETSc.Error as err:
            raise self._failed("KSPSolve", err) from err
        return int(self._ksp.getIterationNumber())

    def _not_converged(self, norm: float, spent: int) -> LinearSolverConvergenceError:
        """The error of a column that did not converge in ``spent`` iterations.

        ``spent`` counts every solve of the column (stale and fresh
        preconditioner).
        """
        PETSc = _petsc.require_petsc()
        ksp = self._ksp
        reason = ksp.getConvergedReason()
        name = _name(PETSc.KSP.ConvergedReason, reason)
        rtol = ksp.getTolerances()[0]
        reached = ksp.getResidualNorm() / (norm or 1.0)
        message = (
            f"PETSc {ksp.getType()} + {ksp.getPC().getType()} did not converge to "
            f"the relative tolerance {rtol:g} in {spent} iterations ({name}, "
            f"reached {reached:.3e}). Use linear_solver='direct', a looser "
            "tolerance, or other petsc_options."
        )
        if reason == PETSc.KSP.ConvergedReason.DIVERGED_INDEFINITE_MAT:
            message += f" {CG_HINT}"
        return LinearSolverConvergenceError(message)

    @property
    def nbytes(self) -> int:
        """Host memory of the matrices and the factors, in bytes.

        The NGSolve matrix, the AIJ's arrays and gather index, and a
        factorization preconditioner's factors: MUMPS's own count of its
        memory (INFOG(22), in millions of bytes), or a PETSc-native
        factor's entries. A non-factorization preconditioner's own memory
        is not counted.
        """
        with self._lock:
            if self._ksp is None or self._matrix is None:
                return 0
            arrays = (self._values, self._rowptr, self._cols, self._index)
            matrix = local_matrix(self._matrix, self.layout.comm)
            held = host_bytes(matrix) + sum(a.nbytes for a in arrays)
            return held + self._factor_bytes()

    def _factor_bytes(self) -> int:
        """The factors' memory, in bytes (0 without factors); under the lock."""
        pc = self._ksp.getPC()
        if pc.getType() not in FACTORIZATIONS:
            return 0
        factor = pc.getFactorMatrix()
        if pc.getFactorSolverType() == "mumps":
            return 10**6 * self._mumps_megabytes(factor)
        PETSc = _petsc.require_petsc()
        entry = np.dtype(PETSc.ScalarType).itemsize + np.dtype(PETSc.IntType).itemsize
        return int(factor.getInfo()["nz_used"]) * entry

    def _mumps_megabytes(self, factor: Any) -> int:
        """MUMPS's own count of its memory, in millions of bytes."""
        return int(factor.getMumpsInfog(22))

    def _release(self) -> None:
        """Destroy the KSP and the AIJ; under the lock."""
        # each on its own: a failed set-up may have made the AIJ only
        for obj in (self._ksp, self._mat):
            if obj is not None:
                obj.destroy()
        self._ksp = self._mat = None


class RankPetscHandle(PetscHandle):
    """A PETSc KSP / PC on the free block across ranks: ``petsc``, and ``direct``.

    Columns are a rank's, and their roles name their status (ADR 0062): a load goes
    through an exchange to owned rows, and owned rows through another to a state.
    The AIJ holds its own values, so a refactorization sends new ones through COO.
    ``kind="direct"`` reads no user options; ``kind="petsc"`` reads them over gfem's
    (`RANK_MUMPS_OPTIONS`), and gets the same exact solve by default.

    A MUMPS that ran out of workspace (INFOG(1)=-9) is set up again on a new KSP with
    ICNTL(14) doubled, since PETSc skips the numeric factorization on one whose
    INFOG(1) is negative. Any other failure, and a workspace still too small after
    `MUMPS_RETRIES`, raises `gfem._mpi.AgreedError` on every rank instead of a solve
    that returns NaN. A ``PETSc.Error`` is one rank's own and ends the job.
    """

    def __init__(
        self,
        matrix: ngs.BaseMatrix,
        freedofs: ngs.BitArray,
        symmetry: Symmetry,
        *,
        is_complex: bool,
        tolerance: float,
        layout: RankLayout,
        kind: str = "petsc",
        maxiter: int = MAXITER,
        options: PetscOptions = (),
        reuse: bool = True,
    ) -> None:
        """Number the free DOFs, then as `PetscHandle`, on ``layout``'s communicator."""
        self.kind = kind
        self.layout = layout
        self._numbering = FreeNumbering(
            layout.pardofs, freedofs, matrix.CreateColVector(), layout.comm
        )
        super().__init__(
            matrix,
            freedofs,
            symmetry,
            is_complex=is_complex,
            tolerance=tolerance,
            maxiter=maxiter,
            options=options,
            reuse=reuse,
        )

    def _defaults(self) -> tuple[tuple[str, Any], ...]:
        return (*DEFAULTS, *RANK_MUMPS_OPTIONS)

    def _petsc_comm(self) -> Any:
        return self.layout.comm

    def _bad_option(self, message: str) -> ValueError:
        return AgreedValueError(message)

    def _block(self, matrix) -> _Block:
        """The free block of ``matrix`` as a new ``MPIAIJ``, which copies the values."""
        local = local_matrix(matrix, self.layout.comm)
        aij, index = distributed_aij(local, self._numbering, self.layout.comm)
        nothing = np.zeros(0, dtype=np.int64)
        return _Block(aij, nothing, nothing, index, nothing)

    def _refresh(self, matrix: ngs.BaseMatrix) -> None:
        local = local_matrix(matrix, self.layout.comm)
        set_values(self._mat, local.CSR()[0], self._index)
        self._ksp.setOperators(self._mat)

    def _set_up_now(self) -> None:
        """``KSPSetUp`` at the rank's BLAS threads, a new KSP while MUMPS needs one."""
        PETSc = _petsc.require_petsc()
        for attempt in range(MUMPS_RETRIES + 1):
            try:
                with self.layout.blas_threads(ngs.GetNumThreads()):
                    self._ksp.setUp()
            except PETSc.Error as err:
                raise self._failed("KSPSetUp", err) from err
            failure = self._agreed_failure()
            if failure is None:
                return
            if failure != MUMPS_WORKSPACE_ERROR or attempt == MUMPS_RETRIES:
                raise self._factorization_failed()
            self._retry_with_more_workspace()

    def _agreed_failure(self) -> int | None:
        """``None`` if the set-up succeeded, else INFOG(1), or 0 without MUMPS."""
        PETSc = _petsc.require_petsc()
        pc = self._ksp.getPC()
        comm = self.layout.comm
        failed = pc.getFailedReason() != PETSc.PC.FailedReason.NOERROR
        if not _mpi.agree_any(comm, flag=failed):
            return None
        if pc.getType() in FACTORIZATIONS and pc.getFactorSolverType() == "mumps":
            return _mpi.agree_min(comm, int(pc.getFactorMatrix().getMumpsInfog(1)))
        return 0

    def _retry_with_more_workspace(self) -> None:
        """A new KSP, inside the scope of the handle's options, with ICNTL(14) doubled."""
        PETSc = _petsc.require_petsc()
        name = "mat_mumps_icntl_14"
        options = dict(self._options)
        options[name] = 2 * max(int(options[name]), 20)
        self._options = tuple(options.items())
        # `scoped_options` deletes this name on exit whatever it holds
        PETSc.Options().setValue(f"-{self._prefix}{name}", str(options[name]))
        spent = self._ksp
        self._new_ksp()
        spent.destroy()
        self._configure(self._user, reuse=self._reuse)

    def _factorization_failed(self) -> RuntimeError:
        return _mpi.AgreedError(str(super()._factorization_failed()))

    def _load_rows(self, cols: np.ndarray) -> np.ndarray:
        rows = np.zeros((self._numbering.count, cols.shape[1]), dtype=complex)
        for j in range(cols.shape[1]):
            rows[:, j] = self._numbering.load_rows(cols[:, j])
        return rows

    def _state_rows(self, cols: np.ndarray) -> np.ndarray:
        rows = np.zeros((self._numbering.count, cols.shape[1]), dtype=complex)
        for j in range(cols.shape[1]):
            rows[:, j] = self._numbering.state_rows(cols[:, j])
        return rows

    def _states(self, rows: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        out = np.zeros(shape, dtype=complex)
        for j in range(shape[1]):
            out[:, j] = self._numbering.state(rows[:, j])
        return out

    def _dense_size(self, rows: int, columns: int) -> Any:
        PETSc = _petsc.require_petsc()
        return ((rows, self._numbering.total), (PETSc.DECIDE, columns))

    def _mumps_megabytes(self, factor: Any) -> int:
        """This rank's: INFO(22). INFOG(22) is the sum, and ADR 0058 sums per node."""
        return int(factor.getMumpsInfo(22))


def _name(enum, value: int) -> str:
    """The name of ``value`` in a petsc4py enum class (its number if none)."""
    names = {v: k for k, v in vars(enum).items() if k.isupper()}
    return names.get(int(value), str(value))


def _refuse(symmetry: Symmetry, *, is_complex: bool) -> None:
    """Raise unless the handle can take ``symmetry`` (before the assembly).

    It needs ``A^T = A`` (the matrix is flagged symmetric for MUMPS LDL^T).
    """
    if not equals_its_transpose(symmetry, is_complex=is_complex):
        raise ValueError(
            "linear_solver='petsc' needs A^T = A: a real symmetric or complex "
            f"symmetric operator, not {symmetry!r}; use linear_solver='direct' or "
            "'bddc_gmres'."
        )


def _refuse_storage(matrix: Any) -> None:
    """Raise unless ``matrix`` is NGSolve's full sparse matrix (before any PETSc call).

    Its free block becomes the AIJ as it is, never mirrored from a triangle.
    """
    if "SparseMatrixSymmetric" in type(matrix).__name__:
        raise ValueError(
            "linear_solver='petsc' needs the full sparse matrix; got NGSolve "
            "symmetric (triangular) storage."
        )
    if not isinstance(matrix, ngs.la.BaseSparseMatrix):
        # a form's storage, not the caller's argument type: as the storage above
        raise ValueError(  # noqa: TRY004
            "linear_solver='petsc' needs an assembled sparse matrix; got "
            f"{type(matrix).__name__}."
        )
