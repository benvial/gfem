"""The eigenmode operator ``T(k) = K + k D + k^2 M + A(k)`` from the driven pieces.

The driven operator is a sum of pieces ``θ_p(ω) a_p`` (`gfem._engine.terms`).
With ``k`` the vacuum wavenumber per mesh file unit (``k = ω scale / c``),
every polynomial piece has ``θ_p`` a polynomial of degree at most 2 in
``k``: curl-curl and lumped L in ``K``; loss conductance, first-order
absorbing and lumped R in the damping ``D`` (``θ = i k``); permittivity and
lumped C in ``M`` (``θ = -k^2``). Each polynomial piece's class is read off
its θ, not its name, so new polynomial terms need no change here. A piece
that declares itself nonpolynomial (a conductor's surface impedance, the
second-order absorbing boundary) makes the problem nonlinear: it stays a
separate term ``θ_p(k) A_p`` of ``A(k)``, applied with its exact θ and
dθ/dk.

The seed problem freezes each nonpolynomial factor at the shift ``sigma``
to first order, ``θ_p(k) ≈ θ_p(sigma) + θ_p'(sigma) (k - sigma)``, which
adds it to ``K`` and ``D``. The seed's ``T(sigma)`` is then the true
``T(sigma)``, so the one prepared operator serves both the shift-invert
Arnoldi of the seed and the refinement of its pairs. A seed can be far
from its root when a nonpolynomial term is large (18 % for a
second-order absorbing wall in the near field of an open line end): only
the refinement on the true ``T(k)`` gives the mode.

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

__all__ = ["EigenOperator", "NonPolynomial", "Split", "assemble", "split"]

#: Relative mismatch above which a θ is not a polynomial of degree <= 2.
_POLYNOMIAL_RTOL = 1e-9
#: Power iterations for the norm estimates of the backward error.
_NORM_ITERATIONS = 20
#: Newton steps of the nonlinear Rayleigh functional, and its relative step
#: at convergence.
_NEWTON_STEPS = 50
_NEWTON_RTOL = 1e-15


@dataclasses.dataclass(frozen=True)
class Split:
    """The driven pieces sorted by their frequency class.

    Attributes:
        powers: ``{power: [(coefficient, piece), ...]}`` for powers 0, 1
            and 2 of ``k``: the polynomial pieces.
        nonpolynomial: The pieces with a nonpolynomial θ.
    """

    powers: dict[int, list[tuple[complex, terms.Piece]]]
    nonpolynomial: list[terms.Piece]


def split(pieces: list[terms.Piece], scale: float, k_ref: float) -> Split:
    """Each polynomial piece's coefficients of ``k^0``, ``k^1`` and ``k^2``.

    θ is fitted at ``k_ref`` (1, 2, 3) and checked at ``4.5 k_ref``.
    Nonpolynomial pieces are passed through.

    Raises:
        NotImplementedError: A piece declared polynomial has a θ that is not
            a polynomial of degree at most 2 in the frequency.
    """
    ks = k_ref * np.array([1.0, 2.0, 3.0, 4.5])
    out: dict[int, list[tuple[complex, terms.Piece]]] = {0: [], 1: [], 2: []}
    nonpolynomial = []
    for piece in pieces:
        if piece.frequency_class == "nonpolynomial":
            nonpolynomial.append(piece)
            continue
        theta = np.array([complex(piece.theta(k * const.c / scale)) for k in ks])
        coefficients = np.linalg.solve(np.vander(ks[:3], 3, increasing=True), theta[:3])
        check = coefficients @ ks[3] ** np.arange(3)
        if abs(check - theta[3]) > _POLYNOMIAL_RTOL * np.abs(theta).max():
            raise NotImplementedError(
                f"The term {piece.name!r} depends on frequency in a way that makes "
                "the eigenmode problem nonlinear, but it does not supply the "
                "derivative the nonlinear eigensolver needs."
            )
        largest = np.abs(coefficients * k_ref ** np.arange(3)).max()
        for power, c in enumerate(coefficients):
            if abs(c) * k_ref**power > 1e-12 * largest:
                out[power].append((complex(c), piece))
    return Split(out, nonpolynomial)


@dataclasses.dataclass(frozen=True)
class NonPolynomial:
    """One nonpolynomial term ``θ(k) A`` of the eigenmode operator.

    Attributes:
        name: The piece's name.
        matrix: ``A``, an NGSolve operator (matrix-free).
        value: ``θ(k)`` for the wavenumber per file unit ``k`` (complex).
        derivative: ``dθ/dk``.
    """

    name: str
    matrix: object
    value: Callable[[complex], complex]
    derivative: Callable[[complex], complex]

    def linear(self, k: complex) -> tuple[complex, complex]:
        """``(c0, c1)`` with ``θ(z) ≈ c0 + c1 z`` to first order around ``k``."""
        slope = self.derivative(k)
        return self.value(k) - k * slope, slope

    @classmethod
    def from_piece(
        cls, piece: terms.Piece, matrix: object, scale: float
    ) -> NonPolynomial:
        """The term of ``piece`` (θ in ω) in the wavenumber per file unit."""
        per_k = const.c / scale  # omega per unit k
        d_theta = piece.d_theta
        if d_theta is None:  # pragma: no cover - Piece validates
            raise ValueError(f"{piece.name!r} has no d_theta.")
        return cls(
            piece.name,
            matrix,
            lambda k: complex(piece.theta(k * per_k)),
            lambda k: complex(d_theta(k * per_k)) * per_k,
        )


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
    """``T(k) = K + k D + k^2 M + Σ θ_p(k) A_p`` on the free DOFs.

    ``T(sigma)`` is prepared. Vectors are NumPy arrays over all DOFs of the
    space, zero on the fixed (`PEC`) DOFs. ``K`` is never stored: ``K x =
    T(sigma) x - sigma D x - sigma^2 M x - Σ θ_p(sigma) A_p x``.

    Attributes:
        shift: ``sigma``, the wavenumber per file unit of the prepared
            operator: the target's, until `reshift`.
        prepare: ``k -> Handle`` of ``T(k)`` (``k`` may be complex).
        damping: ``D`` (a matrix-free NGSolve operator), or ``None`` when
            undamped.
        mass: ``M``, an NGSolve matrix.
        free: Free-DOF mask.
        scale: Metres per mesh file unit.
        nonpolynomial: The nonpolynomial terms ``θ_p(k) A_p``.
        shifts: How many times `reshift` refactorized.
    """

    shift: complex
    prepare: Callable[[complex], Handle]
    damping: object | None
    mass: object
    free: np.ndarray
    scale: float
    nonpolynomial: tuple[NonPolynomial, ...] = ()
    shifts: int = 0
    _norms: tuple[float, ...] | None = None
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

    def _matvec(self, matrix, x: np.ndarray) -> np.ndarray:
        """A matrix on the free DOFs (zero on the fixed rows)."""
        return np.where(self.free, matvec(matrix, np.where(self.free, x, 0)), 0)

    def _parts(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """``(K x, D x, M x, [A_p x])`` on the free rows."""
        x = np.where(self.free, x, 0)
        mx = self._matvec(self.mass, x)
        dx = (
            np.zeros_like(mx) if self.damping is None else self._matvec(self.damping, x)
        )
        ax = [self._matvec(t.matrix, x) for t in self.nonpolynomial]
        s = self.shift
        kx = np.where(self.free, self.handle.apply(x), 0) - s * dx - s * s * mx
        for t, a in zip(self.nonpolynomial, ax, strict=True):
            kx = kx - t.value(s) * a
        return kx, dx, mx, ax

    def parts(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(K x, D x, M x)`` on the free rows."""
        kx, dx, mx, _ = self._parts(x)
        return kx, dx, mx

    def apply(self, k: complex, x: np.ndarray) -> np.ndarray:
        """``T(k) x``."""
        kx, dx, mx, ax = self._parts(x)
        out = kx + k * dx + k * k * mx
        for t, a in zip(self.nonpolynomial, ax, strict=True):
            out = out + t.value(k) * a
        return out

    def derivative(self, k: complex, x: np.ndarray) -> np.ndarray:
        """``T'(k) x = D x + 2 k M x + Σ θ_p'(k) A_p x``, ``'`` = ``d/dk``.

        The denominator of the eigenvalue derivative ``dk/dp = -x^T (dT/dp)
        x / x^T T'(k) x`` (and of the Rayleigh functional's Newton step).
        """
        x = np.where(self.free, x, 0)
        out = 2 * k * self._matvec(self.mass, x)
        if self.damping is not None:
            out = out + self._matvec(self.damping, x)
        for t in self.nonpolynomial:
            out = out + t.derivative(k) * self._matvec(t.matrix, x)
        return out

    # --- the frozen seed -----------------------------------------------------------

    @property
    def seed_damped(self) -> bool:
        """Whether the seed problem has a damping term (``D`` or a frozen one)."""
        return self.damping is not None or bool(self.nonpolynomial)

    def seed_damping(self, x: np.ndarray) -> np.ndarray:
        """The seed's damping on ``x``: ``D`` plus each ``θ_p'(sigma) A_p``.

        (The seed's ``K`` adds ``θ_p(sigma) - sigma θ_p'(sigma)`` per term,
        which only enters through ``T(sigma)``.)
        """
        s = self.shift
        d = np.zeros(len(self.free), dtype=complex)
        if self.damping is not None:
            d = self._matvec(self.damping, x)
        for t in self.nonpolynomial:
            d = d + t.derivative(s) * self._matvec(t.matrix, x)
        return d

    def seed_mass(self, x: np.ndarray) -> np.ndarray:
        """The seed's mass on ``x`` (``M``: the seed freezes no term into it)."""
        return self._matvec(self.mass, x)

    # --- the Rayleigh functional and the backward error -----------------------------

    def rayleigh(self, x: np.ndarray, k: complex) -> complex:
        """The root of ``x^T T(k) x = 0`` nearest ``k``.

        A quadratic's root in closed form. With nonpolynomial terms, a
        Newton iteration that keeps the quadratic exact: each step takes the
        root nearest the current ``k`` of the quadratic with every term
        linearized around ``k`` (`NonPolynomial.linear`), so it stays on the
        branch of the start and converges quadratically.
        """
        kx, dx, mx, ax = self._parts(x)
        a, b, c = x @ kx, x @ dx, x @ mx
        if not self.nonpolynomial:
            return _nearest_root((a, b, c), k)
        pairs = [(t, x @ y) for t, y in zip(self.nonpolynomial, ax, strict=True)]
        k = complex(k)
        for _ in range(_NEWTON_STEPS):
            linear = [a, b, c]
            for t, n in pairs:
                c0, c1 = t.linear(k)
                linear[0] += c0 * n
                linear[1] += c1 * n
            new = _nearest_root(linear, k)
            done = abs(new - k) <= _NEWTON_RTOL * abs(new)
            k = new
            if done:
                break
        return k

    @property
    def norms(self) -> tuple[float, ...]:
        """2-norm estimates of ``K``, ``D``, ``M`` and each ``A_p``."""
        if self._norms is None:
            rng = np.random.default_rng(0)
            start = np.where(self.free, rng.standard_normal(len(self.free)), 0)

            def k_op(x):
                return self.parts(x)[0]

            def matrix_op(matrix):
                if matrix is None:
                    return np.zeros_like
                return lambda x: self._matvec(matrix, x)

            self._norms = (
                _norm2(k_op, start),
                _norm2(matrix_op(self.damping), start),
                _norm2(matrix_op(self.mass), start),
                *(_norm2(matrix_op(t.matrix), start) for t in self.nonpolynomial),
            )
        return self._norms

    def backward_error(self, k: complex, x: np.ndarray) -> float:
        """``|T(k) x| / ((|K| + |k| |D| + |k|^2 |M| + Σ |θ_p(k)| |A_p|) |x|)``."""
        nk, nd, nm, *na = self.norms
        r = np.linalg.norm(self.apply(k, x))
        den = nk + abs(k) * nd + abs(k) ** 2 * nm
        den += sum(
            abs(t.value(k)) * a for t, a in zip(self.nonpolynomial, na, strict=True)
        )
        den *= np.linalg.norm(x)
        return float(r / den) if den > 0 else math.inf


def _nearest_root(coefficients, k: complex) -> complex:
    """The root of ``c0 + c1 z + c2 z^2`` nearest ``k``."""
    a, b, c = coefficients
    roots = np.roots([c, b, a]) if c != 0 else np.array([-a / b])
    return complex(roots[np.argmin(np.abs(roots - k))])


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
