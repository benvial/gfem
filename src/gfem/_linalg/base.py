"""What every NGSolve-backed handle shares: NumPy columns in, NumPy columns out."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, ClassVar, Self

import ngsolve as ngs
import numpy as np

from gfem._linalg import Operator, Symmetry


def equals_its_transpose(symmetry: Symmetry, *, is_complex: bool) -> bool:
    """Whether an operator of ``symmetry`` has ``A^T = A``.

    A complex symmetric operator, or a ``"hermitian"`` one on a real space
    (real symmetric); not a complex Hermitian or a general one.
    """
    return symmetry == "complex_symmetric" or (
        symmetry == "hermitian" and not is_complex
    )


def host_bytes(obj) -> int:
    """Host memory NGSolve reports for a matrix or an inverse, in bytes."""
    return sum(int(u.host) for u in obj.GetMemoryUsage())


def same_pattern(a, b) -> bool:
    """Whether two assembled sparse matrices have one sparsity pattern.

    Compares the sizes and nonzero counts, then the CSR row pointers and
    column indices (no copies: NGSolve's CSR arrays are views).
    """
    if a.height != b.height or a.nze != b.nze:
        return False
    _, cols_a, rowptr_a = a.CSR()
    _, cols_b, rowptr_b = b.CSR()
    return np.array_equal(np.asarray(rowptr_a), np.asarray(rowptr_b)) and (
        np.array_equal(np.asarray(cols_a), np.asarray(cols_b))
    )


def assemble_operator(
    operator: Operator,
) -> tuple[ngs.BaseMatrix, ngs.BaseMatrix | None]:
    """A new matrix of ``operator`` and the pattern of its pieces, if summed.

    The piece sum when the operator has one whose pieces share a pattern;
    otherwise the assembled form and ``None``.
    """
    piece_sum = operator.piece_sum
    if piece_sum is not None:
        matrix = piece_sum.matrix()
        if matrix is not None:
            return matrix, piece_sum.pattern
    operator.form.Assemble()
    return operator.form.mat, None


def free_mask(freedofs: ngs.BitArray | None, ndof: int) -> np.ndarray:
    """The free DOFs as a NumPy boolean mask; ``None`` means every DOF is free.

    No per-DOF Python loop: every bit set gives ``np.ones``; otherwise an
    NGSolve vector is set to 1 on the BitArray (in C++) and compared with 0.

    Raises:
        ValueError: ``freedofs`` does not have ``ndof`` bits.
    """
    if freedofs is not None and len(freedofs) != ndof:
        raise ValueError(
            f"The free-DOF set has {len(freedofs)} bits, the space {ndof} DOFs."
        )
    if freedofs is None or freedofs.NumSet() == ndof:
        return np.ones(ndof, dtype=bool)
    marks = ngs.la.CreateVVector(len(freedofs))
    marks[:] = 0.0
    marks[freedofs] = 1.0
    return marks.FV().NumPy() != 0.0


def same_free_dofs(a: ngs.BitArray, b: ngs.BitArray) -> bool:
    """Whether two free-DOF sets are equal, without building masks.

    The same object short-circuits; otherwise equal lengths, equal counts
    and no bit of ``a`` outside ``b`` (``a`` within ``b`` and as many bits
    means ``a == b``).
    """
    return a is b or (
        len(a) == len(b) and a.NumSet() == b.NumSet() and (a & ~b).NumSet() == 0
    )


class NgsHandle:
    """A handle on an assembled NGSolve matrix with free DOFs.

    Subclasses set ``self._solver`` (an NGSolve operator applying the
    inverse on the free DOFs), implement `nbytes`, and tear their backend
    down in `_release`. Calls are serialized by a lock: NGSolve inverses
    keep work buffers, and JAX may call a host callback from several
    threads. `free` takes it too, so it waits for a running solve or
    refactorization.
    """

    kind: str

    def __init__(
        self,
        matrix: ngs.BaseMatrix,
        freedofs: ngs.BitArray | None,
        symmetry: Symmetry,
        *,
        is_complex: bool,
    ) -> None:
        """Hold ``matrix``; ``freedofs=None`` means every DOF is free."""
        self.symmetry: Symmetry = symmetry
        self.ndof = int(matrix.height)
        self._matrix: ngs.BaseMatrix | None = matrix
        self._free = free_mask(freedofs, self.ndof)
        self._complex = is_complex
        self._solver: Any = None
        self._lock = threading.RLock()

    def _columns(self, B: np.ndarray) -> tuple[np.ndarray, bool]:
        B = np.asarray(B)
        if B.shape[0] != self.ndof or B.ndim not in (1, 2):
            raise ValueError(
                f"Expected an array of shape ({self.ndof},) or ({self.ndof}, k), "
                f"got {B.shape}."
            )
        return (B[:, None], True) if B.ndim == 1 else (B, False)

    def _map(self, op, B: np.ndarray, after=None) -> np.ndarray:
        """``op`` applied to each column of ``B``.

        ``after(op, x, y)`` checks each column's ``y = op * x``, and may
        change ``y`` in place.
        """
        cols, flat = self._columns(B)
        dtype = np.complex128 if self._complex else np.float64
        out = np.zeros(cols.shape, dtype=dtype)
        matrix = self._matrix
        if matrix is None:
            raise RuntimeError("This solver handle was freed.")
        x = matrix.CreateColVector()
        y = matrix.CreateColVector()
        with self._lock:
            for j in range(cols.shape[1]):
                x.FV().NumPy()[:] = cols[:, j]
                y.data = op * x
                if after is not None:
                    after(op, x, y)
                out[:, j] = y.FV().NumPy()
        return out[:, 0] if flat else out

    def _inverse_map(
        self,
        B: np.ndarray,
        tolerance: float | None,  # noqa: ARG002
        x0: np.ndarray | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Hook for subclasses: the inverse on the free rows of ``B``.

        ``x0`` (same shape as ``B``, zero on fixed rows) is an initial guess.
        Direct backends ignore ``tolerance`` and ``x0``. Returns the Krylov
        iterations of each column too, ``()`` for a direct solve.
        """
        return self._map(self._solver, B), ()

    def solve(
        self,
        B: np.ndarray,
        *,
        transpose: bool = False,
        tolerance: float | None = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """``X`` alone: ``solve_counted(B, ...)[0]``."""
        return self.solve_counted(B, transpose=transpose, tolerance=tolerance, x0=x0)[0]

    def solve_counted(
        self,
        B: np.ndarray,
        *,
        transpose: bool = False,
        tolerance: float | None = None,
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """``A_ff X_f = B_f`` on the free DOFs; ``X`` is zero on fixed DOFs.

        ``transpose`` solves ``A^T X = B``: the same solve for a real or
        complex symmetric operator, the conjugate one for a complex
        Hermitian operator. ``tolerance`` is the relative residual of this
        solve for iterative backends (an adjoint solve on the forward's
        preconditioner); ``None`` is the prepared tolerance. ``x0``, the
        shape of ``B``, is an initial guess for iterative backends; its fixed
        rows are not read. Direct backends ignore both.

        Returns ``X`` and the Krylov iterations of each column of ``B`` (see
        `gfem._linalg.Handle.solve_counted`); a real handle solves a complex
        ``B`` as two real solves, and a column counts both.

        Raises:
            RuntimeError: The handle was freed.
            NotImplementedError: ``transpose`` for a general operator.
        """
        if self._solver is None:
            raise RuntimeError("This solver handle was freed.")
        B = np.asarray(B)
        conjugate = transpose and self._complex and self.symmetry == "hermitian"
        if transpose and self.symmetry == "general":
            raise NotImplementedError(
                f"The {self.kind!r} solver has no transposed solve of a general "
                "operator."
            )
        if conjugate:
            # the conjugate system, solved in this call: its counts unchanged
            B = np.conj(B)
            x0 = None if x0 is None else np.conj(x0)
        free = self._free if B.ndim == 1 else self._free[:, None]
        B = np.where(free, B, 0)
        if x0 is not None:
            x0 = np.asarray(x0)
            if x0.shape != B.shape:
                raise ValueError(
                    f"The initial guess has shape {x0.shape}, the right-hand "
                    f"sides {B.shape}."
                )
            x0 = np.where(free, x0, 0)
        if not self._complex and np.iscomplexobj(B):
            re = None if x0 is None else x0.real
            im = None if x0 is None else x0.imag
            X_re, its_re = self._inverse_map(B.real, tolerance, re)
            X_im, its_im = self._inverse_map(B.imag, tolerance, im)
            its = tuple(a + b for a, b in zip(its_re, its_im, strict=True))
            return X_re + 1j * X_im, its
        if x0 is not None and not self._complex:
            x0 = x0.real
        X, its = self._inverse_map(B, tolerance, x0)
        return (np.conj(X), its) if conjugate else (X, its)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """``A X`` with the full assembled matrix.

        Raises:
            RuntimeError: The handle was freed.
        """
        if self._matrix is None:
            raise RuntimeError("This solver handle was freed.")
        return self._map(self._matrix, X)

    def _live(self) -> ngs.BaseMatrix:
        """The held matrix; call under the lock.

        Raises:
            RuntimeError: The handle was freed.
        """
        if self._solver is None or self._matrix is None:
            raise RuntimeError("This solver handle was freed.")
        return self._matrix

    def free(self) -> None:
        """Release the backend and drop the matrix.

        Waits for a running solve or refactorization.
        """
        with self._lock:
            self._release()
            self._solver = None
            self._matrix = None

    def _release(self) -> None:
        """Hook for subclasses: tear the backend down; under the lock."""


class PlannedHandle(NgsHandle):
    """A handle that factorizes on a solver plan, and refactorizes on it.

    The refactorization flow of the direct, PETSc and cuDSS backends: a
    `gfem._linalg.Refactorable` handle. Each backend is an adapter that
    supplies its refusal (`_admit`), "factorize new values on the current
    plan" (`_new_values`), "build a new plan" (`_new_plan`) and its
    teardown (`_release`); every step but `_admit` runs under the lock, on
    a live handle.

    A backend that ``sums_pieces`` sums an operator's piece sum
    (``Operator.piece_sum``) instead of assembling its form: into a new
    matrix at creation, into the held matrix at `refactorize` when the
    piece sum shares the held matrix's pieces. Otherwise it assembles the
    form, and never reads the piece sum.
    """

    #: The backend sums an operator's piece sum instead of assembling its form.
    sums_pieces: ClassVar[bool]
    #: The pattern of the pieces whose sums the held matrix holds, if summed.
    _pattern: ngs.BaseMatrix | None = None

    @classmethod
    def from_operator(
        cls, operator: Operator, symmetry: Symmetry, **options: Any
    ) -> Self:
        """Assemble (or sum the piece sum) and set up (see `gfem._linalg.prepare`).

        ``options`` are the backend's own keywords. The backend refuses
        ``symmetry`` before paying for the assembly.
        """
        is_complex = bool(operator.form.space.is_complex)
        cls._admit(symmetry, is_complex=is_complex)
        matrix, pattern = cls._assemble(operator)
        handle = cls(
            matrix, operator.freedofs, symmetry, is_complex=is_complex, **options
        )
        handle._pattern = pattern
        return handle

    @classmethod
    def _assemble(
        cls, operator: Operator
    ) -> tuple[ngs.BaseMatrix, ngs.BaseMatrix | None]:
        """A new matrix of ``operator`` and the pattern of its pieces, if summed."""
        if cls.sums_pieces:
            return assemble_operator(operator)
        operator.form.Assemble()
        return operator.form.mat, None

    def refactorize(
        self,
        operator: Operator,
        *,
        solves: int | None = None,  # noqa: ARG002
    ) -> None:
        """Refactorize on ``operator``'s values (see `gfem._linalg.Refactorable`).

        An operator whose piece sum shares the held matrix's pieces is summed
        into it in place (`apply` reads it); any other operator is assembled
        (`_assemble`) into a new matrix. On the held matrix's pattern its
        values are copied in, and the backend factorizes them on the same
        solver plan (`_new_values`); a changed pattern is planned anew
        (`_new_plan`).

        ``solves`` is ignored: only BDDC's coarse choice reads it.

        Raises:
            RuntimeError: The handle was freed.
        """
        piece_sum = operator.piece_sum if self.sums_pieces else None
        # read outside the lock: the first read assembles the pieces
        summed = None if piece_sum is None else piece_sum.pattern
        with self._lock:
            self._live()
            if piece_sum is not None and summed is not None and summed is self._pattern:
                self._new_values(piece_sum.sum_into)
                return
        new, pattern = self._assemble(operator)  # assembly needs no lock
        with self._lock:
            held = self._live()
            if same_pattern(held, new):
                self._pattern = pattern

                def write(matrix: ngs.BaseMatrix) -> None:
                    matrix.AsVector().data = new.AsVector()

                self._new_values(write)
                return
            # a failed plan leaves no pattern to sum into: the next
            # refactorization assembles
            self._pattern = None
            self._new_plan(new, operator.freedofs)
            self._pattern = pattern

    @classmethod
    def _admit(cls, symmetry: Symmetry, *, is_complex: bool) -> None:
        """Raise the backend's refusal of ``symmetry``, if it refuses it."""
        raise NotImplementedError

    def _new_values(self, write: Callable[[ngs.BaseMatrix], None]) -> None:
        """``write(held matrix)``, then factorize its values on the current plan."""
        raise NotImplementedError

    def _new_plan(self, matrix: ngs.BaseMatrix, freedofs: ngs.BitArray | None) -> None:
        """Plan and factorize ``matrix`` (same free DOFs), and hold it."""
        raise NotImplementedError
