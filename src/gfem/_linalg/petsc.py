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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import ngsolve as ngs
import numpy as np

from gfem import _petsc
from gfem._linalg import MAXITER, Operator, Symmetry
from gfem._linalg.base import (
    NgsHandle,
    assemble_operator,
    equals_its_transpose,
    host_bytes,
    same_pattern,
)

if TYPE_CHECKING:
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


class _Block(NamedTuple):
    """A free block as a PETSc AIJ over its own arrays (`PetscHandle._block`)."""

    aij: Any
    rowptr: np.ndarray
    cols: np.ndarray
    #: The free block's entries in NGSolve's value vector.
    index: np.ndarray
    values: np.ndarray


class PetscHandle(NgsHandle):
    """A PETSc KSP / PC on the free block of the operator.

    Accepts ``A^T = A``: a complex symmetric operator, or a real symmetric
    one (``"hermitian"`` on a real space), solved in complex arithmetic.

    Attributes:
        iterations: Krylov iterations of every column solved so far (empty
            for ``ksp_type preonly``).
    """

    kind = "petsc"

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
        sets the preconditioner up again at every `refactorize`.
        """
        _refuse(symmetry, is_complex=is_complex, matrix=matrix)
        super().__init__(matrix, freedofs, symmetry, is_complex=is_complex)
        self._lock = _petsc.LOCK  # PETSc's database and registry are global
        self.iterations: list[int] = []
        self._pattern: ngs.BaseMatrix | None = None
        #: The KSP and the AIJ (petsc4py objects); ``None`` once freed.
        self._ksp: Any = None
        self._mat: Any = None
        PETSc = _petsc.require_petsc("linear_solver")
        self._prefix = _petsc.next_prefix()
        defaults = (*DEFAULTS, ("ksp_max_it", maxiter), ("ksp_rtol", tolerance))
        self._options = tuple({**dict(defaults), **dict(options)}.items())
        with self._lock:
            try:
                self._adopt(self._block(matrix))
                ksp = PETSc.KSP().create(PETSc.COMM_SELF)
                self._solver = self._ksp = ksp
                ksp.setOptionsPrefix(self._prefix)
                ksp.setOperators(self._mat)
                with _petsc.scoped_options(self._prefix, self._options):
                    self._configure(dict(options), reuse=reuse)
                    self._set_up_now()
                    self._check_read([name for name, _ in options])
            except BaseException:
                self.free()
                raise

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
            raise ValueError(
                "SolverOptions.petsc_options was refused by PETSc: "
                f"{_petsc.error_text(err)}."
            ) from err
        self._direct = ksp.getType() == "preonly"
        name = "ksp_reuse_preconditioner"
        if (
            self._direct
            and name in user
            and PETSc.Options().getBool(f"{self._prefix}{name}", False)
        ):
            raise ValueError(
                f"SolverOptions.petsc_options[{name!r}] with ksp_type preonly would "
                "solve the previous operator after each refactorization; use a "
                "Krylov ksp_type with it."
            )
        if not self._direct and "ksp_norm_type" not in user:
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        if not reuse:
            ksp.getPC().setReusePreconditioner(False)

    def _check_read(self, names: list[str]) -> None:
        """Raise if PETSc did not read some of the user's ``names``.

        Call after the first set-up, with the options still in the database.
        """
        unread = _petsc.unread(self._prefix, names)
        if not unread:
            return
        keys = ", ".join(f"SolverOptions.petsc_options[{n!r}]" for n in unread)
        verb = "was" if len(unread) == 1 else "were"
        raise ValueError(
            f"{keys} {verb} not read by PETSc's linear solver ({self._describe()}): "
            "check the name, or remove options this configuration does not use."
        )

    def _describe(self) -> str:
        """The configuration, e.g. ``KSP preonly, PC cholesky, MUMPS``."""
        pc = self._ksp.getPC()
        text = f"KSP {self._ksp.getType()}, PC {pc.getType()}"
        if pc.getType() in FACTORIZATIONS:
            text += f", {pc.getFactorSolverType().upper()}"
        return text

    @classmethod
    def from_operator(
        cls,
        operator: Operator,
        symmetry: Symmetry,
        *,
        tolerance: float,
        maxiter: int = MAXITER,
        options: PetscOptions = (),
        reuse: bool = True,
    ) -> PetscHandle:
        """Assemble (or sum the piece sum) and set up (see `gfem._linalg.prepare`)."""
        is_complex = bool(operator.form.space.is_complex)
        _refuse(symmetry, is_complex=is_complex)  # before paying for the assembly
        matrix, pattern = assemble_operator(operator)
        handle = cls(
            matrix,
            operator.freedofs,
            symmetry,
            is_complex=is_complex,
            tolerance=tolerance,
            maxiter=maxiter,
            options=options,
            reuse=reuse,
        )
        handle._pattern = pattern
        return handle

    def refactorize(self, operator: Operator) -> None:
        """Refactorize on ``operator``'s values (see `gfem._linalg.Refactorable`).

        An operator whose piece sum shares this handle's pieces is summed
        into the held matrix in place (`apply` reads it); any other
        operator's piece sum is summed, or its form assembled, into a new
        matrix whose values are copied in. The free block's values are then
        gathered into the AIJ's arrays and set up again on the same solver
        plan (a numeric factorization by default). A changed pattern is
        planned anew: a new AIJ under the same KSP.

        Raises:
            RuntimeError: The handle was freed, or PETSc failed (as `prepare`).
        """
        piece_sum = operator.piece_sum
        # read outside the lock: the first read assembles the pieces
        summed = None if piece_sum is None else piece_sum.pattern
        with self._lock:
            matrix = self._live()
            pattern = self._pattern
            if piece_sum is not None and pattern is not None and summed is pattern:
                piece_sum.sum_into(matrix)
                self._refactor_values(matrix)
                return
        new, pattern = assemble_operator(operator)  # assembly needs no lock
        with self._lock:
            matrix = self._live()
            if same_pattern(matrix, new):
                self._pattern = pattern
                matrix.AsVector().data = new.AsVector()
                self._refactor_values(matrix)
                return
            # the new plan is built aside: a failure leaves the handle as it was
            block, old = self._block(new), self._mat
            try:
                self._ksp.setOperators(block.aij)
            except BaseException:
                block.aij.destroy()
                raise
            self._matrix, self._pattern = new, pattern
            self._adopt(block)
            try:
                self._set_up()  # symbolic and numeric, with the options read
            finally:
                old.destroy()  # the KSP let go of it at setOperators

    def _live(self) -> ngs.BaseMatrix:
        """The held NGSolve matrix; call under the lock.

        Raises:
            RuntimeError: The handle was freed.
        """
        if self._ksp is None or self._matrix is None:
            raise RuntimeError("This solver handle was freed.")
        return self._matrix

    def _refactor_values(self, matrix) -> None:
        """Set the KSP up on ``matrix``'s new values; under the lock."""
        self._gather(matrix.CSR()[0], self._index, self._values)
        self._mat.assemble()  # marks the new values: PETSc sets up again
        self._set_up()

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
    ) -> np.ndarray:
        """``A_ff^{-1} B_f`` on the free rows, zero on the fixed ones.

        ``preonly`` solves every column at once and ignores ``tolerance``
        and ``x0``; a Krylov method solves column by column, from ``x0``
        when given, to ``tolerance`` (the prepared one when ``None``).
        """
        cols, flat = self._columns(B)
        free = self._free
        rhs = np.ascontiguousarray(cols[free], dtype=complex)
        guesses = None if x0 is None else self._columns(x0)[0][free]
        out = np.zeros(cols.shape, dtype=complex)
        with self._lock:
            if self._ksp is None:
                raise RuntimeError("This solver handle was freed.")
            if self._direct:
                out[free] = self._exact(rhs)
            else:
                out[free] = self._krylov(rhs, tolerance, guesses)
        result = out if self._complex else out.real
        return result[:, 0] if flat else result

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
                Bm = PETSc.Mat().createDense(
                    (n, k), array=columns, comm=PETSc.COMM_SELF
                )
                Xm = PETSc.Mat().createDense((n, k), comm=PETSc.COMM_SELF)
                try:
                    ksp.matSolve(Bm, Xm)
                    X = Xm.getDenseArray().copy()
                finally:
                    Bm.destroy()
                    Xm.destroy()
        except PETSc.Error as err:
            raise self._failed("KSPSolve", err) from err
        if ksp.getConvergedReason() < 0:
            raise self._factorization_failed()
        return X

    def _krylov(
        self, rhs: np.ndarray, tolerance: float | None, guesses: np.ndarray | None
    ) -> np.ndarray:
        """A Krylov solve per column, recording its iterations."""
        PETSc = _petsc.require_petsc()
        ksp = self._ksp
        prepared = ksp.getTolerances()
        _, atol, divtol, max_it = prepared
        if tolerance is not None:
            ksp.setTolerances(rtol=tolerance, atol=atol, divtol=divtol, max_it=max_it)
        X = np.zeros_like(rhs)
        x, b = self._mat.createVecs()
        try:
            for j in range(rhs.shape[1]):
                b.array[:] = rhs[:, j]
                ksp.setInitialGuessNonzero(guesses is not None)
                if guesses is not None:
                    x.array[:] = guesses[:, j]
                try:
                    ksp.solve(b, x)
                except PETSc.Error as err:
                    raise self._failed("KSPSolve", err) from err
                self.iterations.append(int(ksp.getIterationNumber()))
                if ksp.getConvergedReason() < 0:
                    raise self._not_converged(b.norm())
                X[:, j] = x.array
        finally:
            x.destroy()
            b.destroy()
            ksp.setTolerances(*prepared)
        return X

    def _not_converged(self, norm: float) -> RuntimeError:
        PETSc = _petsc.require_petsc()
        ksp = self._ksp
        reason = ksp.getConvergedReason()
        name = _name(PETSc.KSP.ConvergedReason, reason)
        rtol, _, _, max_it = ksp.getTolerances()
        reached = ksp.getResidualNorm() / (norm or 1.0)
        message = (
            f"PETSc {ksp.getType()} + {ksp.getPC().getType()} did not converge to "
            f"the relative tolerance {rtol:g} in {max_it} iterations ({name}, "
            f"reached {reached:.3e}). Use linear_solver='direct', a looser "
            "tolerance, or other petsc_options."
        )
        if reason == PETSc.KSP.ConvergedReason.DIVERGED_INDEFINITE_MAT:
            message += f" {CG_HINT}"
        return RuntimeError(message)

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
            held = host_bytes(self._matrix) + sum(a.nbytes for a in arrays)
            return held + self._factor_bytes()

    def _factor_bytes(self) -> int:
        """The factors' memory, in bytes (0 without factors); under the lock."""
        pc = self._ksp.getPC()
        if pc.getType() not in FACTORIZATIONS:
            return 0
        factor = pc.getFactorMatrix()
        if pc.getFactorSolverType() == "mumps":
            return 10**6 * int(factor.getMumpsInfog(22))
        PETSc = _petsc.require_petsc()
        entry = np.dtype(PETSc.ScalarType).itemsize + np.dtype(PETSc.IntType).itemsize
        return int(factor.getInfo()["nz_used"]) * entry

    def free(self) -> None:
        """Destroy the KSP and the AIJ, and drop the matrices.

        Waits for a running solve or refactorization.
        """
        with self._lock:
            # each on its own: a failed set-up may have made the AIJ only
            for obj in (self._ksp, self._mat):
                if obj is not None:
                    obj.destroy()
            super().free()
            self._ksp = self._mat = None
            self._pattern = None


def _name(enum, value: int) -> str:
    """The name of ``value`` in a petsc4py enum class (its number if none)."""
    names = {v: k for k, v in vars(enum).items() if k.isupper()}
    return names.get(int(value), str(value))


def _refuse(symmetry: Symmetry, *, is_complex: bool, matrix=None) -> None:
    """Raise unless the handle can take the operator (before any PETSc call).

    It needs ``A^T = A`` (the matrix is flagged symmetric for MUMPS LDL^T)
    and, when given, ``matrix`` as NGSolve's full sparse matrix: its free
    block becomes the AIJ as it is, never mirrored from a triangle.
    """
    if not equals_its_transpose(symmetry, is_complex=is_complex):
        raise ValueError(
            "linear_solver='petsc' needs A^T = A: a real symmetric or complex "
            f"symmetric operator, not {symmetry!r}."
        )
    if matrix is None:
        return
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
