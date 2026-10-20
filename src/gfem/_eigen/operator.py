"""The eigenmode operator ``T(k) = K + k D + k^2 M`` from the driven pieces.

The driven operator is a sum of pieces ``θ_p(ω) a_p`` (`gfem._engine.terms`).
With ``k`` the vacuum wavenumber per mesh file unit (``k = ω scale / c``),
every piece of a polynomial eigenmode problem has ``θ_p`` a polynomial of
degree at most 2 in ``k``: curl-curl and lumped L in ``K``; loss
conductance, first-order absorbing and lumped R in the damping ``D``
(``θ = i k``); permittivity and lumped C in ``M`` (``θ = -k^2``). Each
piece's class is read off its θ, not its name, so new polynomial terms need
no change here. A piece whose θ is not such a polynomial (a conductor's
surface impedance, the second-order absorbing boundary) makes the problem
nonlinear; `split` refuses it.

``T(k)`` is complex symmetric (``T^T = T``), so the two-sided Rayleigh
functional ``x^T T(k) x = 0`` gives the eigenvalue of an approximate
eigenvector ``x`` to second order in its error.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem._engine import terms
from gfem._linalg import Handle

__all__ = ["EigenOperator", "assemble", "split"]

#: Relative mismatch above which a θ is not a polynomial of degree <= 2.
_POLYNOMIAL_RTOL = 1e-9
#: Power iterations for the norm estimates of the backward error.
_NORM_ITERATIONS = 20


def split(
    pieces: list[terms.Piece], scale: float, k_ref: float
) -> dict[int, list[tuple[complex, terms.Piece]]]:
    """Each piece's coefficients of ``k^0``, ``k^1`` and ``k^2``.

    θ is fitted at ``k_ref`` (1, 2, 3) and checked at ``4.5 k_ref``.

    Returns:
        ``{power: [(coefficient, piece), ...]}``, powers 0, 1, 2.

    Raises:
        NotImplementedError: A piece's θ is not a polynomial of degree at
            most 2 in the frequency: the eigenproblem is nonlinear.
    """
    ks = k_ref * np.array([1.0, 2.0, 3.0, 4.5])
    out: dict[int, list[tuple[complex, terms.Piece]]] = {0: [], 1: [], 2: []}
    for piece in pieces:
        theta = np.array([complex(piece.theta(k * const.c / scale)) for k in ks])
        coefficients = np.linalg.solve(np.vander(ks[:3], 3, increasing=True), theta[:3])
        check = coefficients @ ks[3] ** np.arange(3)
        if abs(check - theta[3]) > _POLYNOMIAL_RTOL * np.abs(theta).max():
            raise NotImplementedError(
                f"The term {piece.name!r} depends on frequency in a way that makes "
                "the eigenmode problem nonlinear, which this version of gfem "
                "cannot solve."
            )
        largest = np.abs(coefficients * k_ref ** np.arange(3)).max()
        for power, c in enumerate(coefficients):
            if abs(c) * k_ref**power > 1e-12 * largest:
                out[power].append((complex(c), piece))
    return out


def assemble(
    space: ngs.FESpace,
    items: list[tuple[complex, terms.Piece]],
    *,
    matrix_free: bool = False,
):
    """``Σ c_p a_p`` as an NGSolve operator.

    Assembled, or applied element by element when ``matrix_free`` (no
    matrix stored: the damping is mostly boundary terms, which an assembled
    matrix would hold in a graph of the whole space).
    """
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True, nonassemble=matrix_free)
    for c, piece in items:
        form += c * piece.integrand(u, v)
    if not matrix_free:
        form.Assemble()
    return form.mat


def matvec(matrix, x: np.ndarray) -> np.ndarray:
    """An NGSolve matrix applied to a NumPy vector."""
    a = matrix.CreateColVector()
    a.FV().NumPy()[:] = x
    b = matrix.CreateColVector()
    b.data = matrix * a
    return b.FV().NumPy().copy()


@dataclasses.dataclass(eq=False)
class EigenOperator:
    """``T(k) = K + k D + k^2 M`` on the free DOFs, with ``T(sigma)`` prepared.

    Vectors are NumPy arrays over all DOFs of the space, zero on the fixed
    (`PEC`) DOFs. ``K`` is never stored: ``K x = T(sigma) x - sigma D x -
    sigma^2 M x``.

    Attributes:
        shift: ``sigma``, the wavenumber per file unit of the prepared
            operator: the target's, until `reshift`.
        prepare: ``k -> Handle`` of ``T(k)`` (``k`` may be complex).
        damping: ``D`` (a matrix-free NGSolve operator), or ``None`` when
            undamped.
        mass: ``M``, an NGSolve matrix.
        free: Free-DOF mask.
        scale: Metres per mesh file unit.
        shifts: How many times `reshift` refactorized.
    """

    shift: complex
    prepare: Callable[[complex], Handle]
    damping: object | None
    mass: object
    free: np.ndarray
    scale: float
    shifts: int = 0
    _norms: tuple[float, float, float] | None = None
    _handle: Handle | None = None

    @property
    def handle(self) -> Handle:
        """``T(shift)``, prepared on first use."""
        if self._handle is None:
            self._handle = self.prepare(self.shift)
        return self._handle

    def reshift(self, k: complex) -> None:
        """Refactorize at ``k``."""
        self.free_handle()
        self.shift = k
        self.shifts += 1

    def free_handle(self) -> None:
        """Release the prepared operator."""
        if self._handle is not None:
            self._handle.free()
            self._handle = None

    def solve(self, b: np.ndarray) -> np.ndarray:
        """``T(sigma)^-1 b`` on the free DOFs."""
        return self.handle.solve(np.where(self.free, b, 0))

    def parts(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(K x, D x, M x)`` on the free rows."""
        x = np.where(self.free, x, 0)
        mx = np.where(self.free, matvec(self.mass, x), 0)
        dx = (
            np.zeros_like(mx)
            if self.damping is None
            else np.where(self.free, matvec(self.damping, x), 0)
        )
        tx = np.where(self.free, self.handle.apply(x), 0)
        s = self.shift
        return tx - s * dx - s * s * mx, dx, mx

    def apply(self, k: complex, x: np.ndarray) -> np.ndarray:
        """``T(k) x``."""
        kx, dx, mx = self.parts(x)
        return kx + k * dx + k * k * mx

    def rayleigh(self, x: np.ndarray, k: complex) -> complex:
        """The root of ``x^T T(k) x = 0`` nearest ``k``."""
        kx, dx, mx = self.parts(x)
        a, b, c = x @ kx, x @ dx, x @ mx
        roots = np.roots([c, b, a]) if c != 0 else np.array([-a / b])
        return complex(roots[np.argmin(np.abs(roots - k))])

    @property
    def norms(self) -> tuple[float, float, float]:
        """2-norm estimates of ``K``, ``D`` and ``M`` (power iteration)."""
        if self._norms is None:
            rng = np.random.default_rng(0)
            start = np.where(self.free, rng.standard_normal(len(self.free)), 0)

            def k_op(x):
                return self.parts(x)[0]

            def m_op(x):
                return np.where(
                    self.free, matvec(self.mass, np.where(self.free, x, 0)), 0
                )

            def d_op(x):
                if self.damping is None:
                    return np.zeros_like(x)
                return np.where(
                    self.free, matvec(self.damping, np.where(self.free, x, 0)), 0
                )

            self._norms = (
                _norm2(k_op, start),
                _norm2(d_op, start),
                _norm2(m_op, start),
            )
        nk, nd, nm = self._norms
        return nk, nd, nm

    def backward_error(self, k: complex, x: np.ndarray) -> float:
        """``|T(k) x| / ((|K| + |k| |D| + |k|^2 |M|) |x|)``."""
        nk, nd, nm = self.norms
        r = np.linalg.norm(self.apply(k, x))
        den = (nk + abs(k) * nd + abs(k) ** 2 * nm) * np.linalg.norm(x)
        return float(r / den) if den > 0 else math.inf


def _norm2(op, start: np.ndarray) -> float:
    """``|A|_2`` of a complex symmetric operator: power iteration on ``A^H A``."""
    x = start / np.linalg.norm(start)
    value = 0.0
    for _ in range(_NORM_ITERATIONS):
        y = op(x)
        z = np.conj(op(np.conj(y)))  # A^H y, with A^T = A
        n = np.linalg.norm(z)
        if n == 0:
            return 0.0
        value = math.sqrt(n)
        x = z / n
    return value
