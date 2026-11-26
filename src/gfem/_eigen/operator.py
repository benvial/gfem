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

A wave port adds a low-rank term ``U C(k) U^T`` (`LowRank`, its modal
correction): ``U`` is fixed by the port mode basis and only the small
``C(k)`` moves. The prepared ``T(sigma)`` holds it as a Woodbury update of
the factorized rest (`gfem._linalg.low_rank_update`); a reshift
refactorizes the rest on its plan and wraps it again.

The seed problem freezes each nonpolynomial factor at the shift ``sigma``
to first order, ``θ_p(k) ≈ θ_p(sigma) + θ_p'(sigma) (k - sigma)``, which
adds it to ``K`` and ``D`` (a term may ask for its value alone,
``θ_p(sigma)``, into ``K``: `NonPolynomial.seed_order`). The seed's
``T(sigma)`` is then the true
``T(sigma)``, so the one prepared operator serves both the shift-invert
Arnoldi of the seed and the refinement of its pairs. A seed can be far
from its root when a nonpolynomial term is large (18 % for a
second-order absorbing wall in the near field of an open line end): only
the refinement on the true ``T(k)`` gives the mode.

``T(k)`` is complex symmetric (``T^T = T``), so the two-sided Rayleigh
functional ``x^T T(k) x = 0`` gives the eigenvalue of an approximate
eigenvector ``x`` to second order in its error. Under a Bloch wave vector
``T(k)`` is general (`EigenOperator.symmetry`): its left eigenvector ``y``
(``T(k)^T y = 0``) is not ``x``, and the functional is ``y^T T(k) x = 0``
(`gfem._eigen.refine` carries ``y``); every matrix then has its transpose.

The matrices are NumPy linear maps (`LinearMap`): `gfem._engine.eigen`
assembles them from the NGSolve forms.

Its columns have the roles of `gfem._linalg.layout`: an eigenvector, a solve's
result and a map's input are states, a map's result (``T(k) x``, ``M x``) is a
load. Every whole-vector product goes through the operator's layout: a state
paired with a load (the Rayleigh functional, `EigenOperator.curl_ratio`), the
norm of a state or of a load, and a load fed back to a map is exchanged to a
state first (`_norm2`). On one process (`ONE_PROCESS`) each is today's NumPy
expression, bit for bit.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, Literal, Protocol

import numpy as np
import scipy.constants as const
import scipy.linalg

from gfem._engine import terms
from gfem._linalg import Handle, Operator, low_rank_update, refactorize_or_prepare
from gfem._linalg.layout import ONE_PROCESS, DofLayout

__all__ = [
    "RESHIFT_SOLVES",
    "EigenOperator",
    "EigenSymmetry",
    "LinearMap",
    "LowRank",
    "NonPolynomial",
    "Split",
    "split",
]


class LinearMap(Protocol):
    """A matrix of the H(curl) space: ``x -> A x``, or ``A^T x`` with ``transpose``.

    On NumPy vectors over all its DOFs (`gfem._engine.eigen` builds them from
    the NGSolve forms).
    """

    def __call__(self, x: np.ndarray, /, *, transpose: bool = False) -> np.ndarray:
        """``A x``, or ``A^T x``."""
        ...


#: Whether ``T(k)^T = T(k)`` (no Bloch phase) or not: the symmetry the shift
#: is prepared with, read from the driven operator.
EigenSymmetry = Literal["complex_symmetric", "general"]

#: Relative mismatch above which a θ is not a polynomial of degree <= 2.
_POLYNOMIAL_RTOL = 1e-9
#: Power iterations for the norm estimates of the backward error.
_NORM_ITERATIONS = 20
#: Newton steps of the nonlinear Rayleigh functional, and its relative step
#: at convergence.
_NEWTON_STEPS = 50
_NEWTON_RTOL = 1e-15
#: Solves one refinement reshift's factorization is expected to serve: its
#: mode's remaining correction steps and the next modes' (1-7 measured on the
#: unit cases, the guard mode included).
RESHIFT_SOLVES = 7


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
        matrix: ``A``, a linear map (applied matrix-free).
        value: ``θ(k)`` for the wavenumber per file unit ``k`` (complex).
        derivative: ``dθ/dk``.
        seed_order: How the seed freezes θ at the shift: ``1`` to first
            order, ``0`` at its value alone (no slope in the seed's damping).
    """

    name: str
    matrix: LinearMap
    value: Callable[[complex], complex]
    derivative: Callable[[complex], complex]
    seed_order: Literal[0, 1] = 1

    def linear(self, k: complex) -> tuple[complex, complex]:
        """``(c0, c1)`` with ``θ(z) ≈ c0 + c1 z`` to first order around ``k``."""
        slope = self.derivative(k)
        return self.value(k) - k * slope, slope

    @classmethod
    def from_piece(
        cls,
        piece: terms.Piece,
        matrix: LinearMap,
        scale: float,
        *,
        seed_order: Literal[0, 1] = 1,
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
            seed_order,
        )


class LowRankSource(Protocol):
    """What a `LowRank` term reads: ``U`` and ``C(k)``, ``C'(k)`` on it.

    `gfem._engine.port_term.PortTerm`: ``at(k)`` may change ``vectors``
    (an enrichment of the port mode basis), so ``vectors`` is read after it.
    """

    @property
    def vectors(self) -> np.ndarray:
        """``U``, ``(ndof, p)``."""
        ...

    def at(self, k: complex) -> Any:
        """The term at ``k``: its ``c`` is ``C(k)`` and its ``d_c`` ``C'(k)``."""
        ...


@dataclasses.dataclass(frozen=True)
class LowRank:
    """One low-rank term ``U C(k) U^T`` of the eigenmode operator.

    A wave port surface's modal correction. ``C`` is complex symmetric, so
    the term is its own transpose.

    Attributes:
        name: For messages.
        source: ``U`` and ``C(k)``.
        free: Free-DOF mask: ``U`` is taken on the free rows.
        seed_order: As `NonPolynomial.seed_order`.
    """

    name: str
    source: LowRankSource
    free: np.ndarray
    seed_order: Literal[0, 1] = 1

    def at(self, k: complex) -> tuple[np.ndarray, np.ndarray]:
        """``(U, C(k))``, ``U`` zero on the fixed rows."""
        c = self.source.at(k).c
        return np.where(self.free[:, None], self.source.vectors, 0), c

    def slope(self, k: complex) -> tuple[np.ndarray, np.ndarray]:
        """``(U, C'(k))``."""
        d_c = self.source.at(k).d_c
        return np.where(self.free[:, None], self.source.vectors, 0), d_c


class ShiftPreparer(Protocol):
    """Prepares a `gfem._linalg` handle of ``T(k)`` at one shift.

    ``solves`` is the solve calls the handle is expected to serve; it
    defaults to the seed solve's.
    """

    def __call__(self, operator: Operator, solves: int = ..., /) -> Handle: ...


@dataclasses.dataclass(eq=False)
class EigenOperator:
    """``T(k) = K + k D + k^2 M + Σ θ_p(k) A_p + Σ U_q C_q(k) U_q^T`` on the free DOFs.

    ``T(sigma)`` is prepared: the rest of ``T`` factorized, the low-rank
    terms held by a Woodbury update of it. Vectors are NumPy arrays over all DOFs of the
    space, zero on the fixed (`PEC`) DOFs. ``K`` is never stored: ``K x =
    A(sigma) x - sigma D x - sigma^2 M x - Σ θ_p(sigma) A_p x`` with
    ``A(sigma)`` the factorized rest, whose low-rank terms are not in it. A
    ``"general"`` operator applies and solves its transpose too
    (``transpose=True``: the transposed maps, the prepared handle's
    transposed apply and solve on the same factors).

    Attributes:
        shift: ``sigma``, the wavenumber per file unit of the prepared
            operator: the target's, until `reshift`.
        operator: ``k -> T(k)``, not yet assembled (``k`` may be complex).
        prepare: A `gfem._linalg` handle of an operator from ``operator``,
            for the seed solve unless it is told the solves it serves.
        damping: ``D`` (a linear map, applied matrix-free), or ``None``
            when undamped.
        mass: ``M``, a linear map.
        free: Free-DOF mask.
        scale: Metres per mesh file unit.
        nonpolynomial: The nonpolynomial terms ``θ_p(k) A_p``.
        low_rank: The low-rank terms ``U_q C_q(k) U_q^T`` (complex
            symmetric), on one process.
        symmetry: ``"complex_symmetric"``, or ``"general"`` under a Bloch
            wave vector (the driven operator's).
        law_constant: The part of ``K`` from polynomial permittivity law
            pieces (a lossless Drude law's constant θ = k_p²), or
            ``None``: `curl_ratio` leaves it out of the curl energy.
        layout: How the DOFs lie across ranks (`ONE_PROCESS` on one process):
            every whole-vector product's.
        numbers: Each DOF's one-process number, or ``None`` on one process:
            the start of the `norms` estimates and the sign of a mode
            (`gfem._sign.largest_value`) are then functions of them, the same
            on every rank count.
        numbering: Under ranks, the owned free DOFs as rows
            (`gfem._linalg.petsc.FreeNumbering`): the seed solver's vectors.
            ``None`` on one process.
        shifts: How many times `reshift` refactorized.
    """

    shift: complex
    operator: Callable[[complex], Operator]
    prepare: ShiftPreparer
    damping: LinearMap | None
    mass: LinearMap
    free: np.ndarray
    scale: float
    nonpolynomial: tuple[NonPolynomial, ...] = ()
    symmetry: EigenSymmetry = "complex_symmetric"
    law_constant: LinearMap | None = None
    layout: DofLayout = ONE_PROCESS
    numbers: np.ndarray | None = None
    numbering: Any = None
    shifts: int = 0
    low_rank: tuple[LowRank, ...] = ()
    _norms: tuple[float, ...] | None = None
    _handle: Handle | None = None
    _base: Handle | None = None
    #: θ_p of the shift the base was prepared at: K subtracts these, since an
    #: enrichment of a port basis since then moves θ_p(shift) off them
    _thetas: tuple[complex, ...] = ()
    _range: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def handle(self) -> Handle:
        """``T(shift)``, prepared on first use."""
        return self.factorize()

    def factorize(self) -> Handle:
        """Prepare ``T(shift)`` now unless it is prepared, and return it."""
        if self._handle is None:
            t = self._operator(self.shift)
            self._base = self.prepare(t)
            self._handle = self._wrap(self._base, self.shift)
        return self._handle

    def _operator(self, k: complex) -> Operator:
        """``operator(k)``, holding the θ_p it was assembled with (`_thetas`)."""
        t = self.operator(k)
        # the port term keeps its value at k: these are the ones in the form
        self._thetas = tuple(p.value(k) for p in self.nonpolynomial)
        return t

    def _wrap(
        self,
        base: Handle,
        k: complex,
        stacked: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Handle:
        """``base`` with the low-rank terms at ``k`` (``base`` itself without).

        The update borrows ``base``: a reshift refactorizes it and wraps it
        again. ``stacked``: `_stacked` at ``k``, when already evaluated.
        """
        if not self.low_rank:
            return base
        u, c = stacked if stacked is not None else self._stacked(k)
        return low_rank_update(base, u, c, borrow=True)

    def _stacked(
        self,
        k: complex,
        *,
        slope: bool = False,
        which: tuple[LowRank, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The low-rank terms' ``U`` side by side, ``C(k)`` (``C'(k)``) block-diagonal.

        ``which``: those, instead of every one.
        """
        which = self.low_rank if which is None else which
        parts = [t.slope(k) if slope else t.at(k) for t in which]
        return (
            np.concatenate([u for u, _ in parts], axis=1),
            scipy.linalg.block_diag(*(c for _, c in parts)),
        )

    def _low_rank(
        self,
        k: complex,
        x: np.ndarray,
        *,
        slope: bool = False,
        transpose: bool = False,
        which: tuple[LowRank, ...] | None = None,
    ) -> np.ndarray:
        """``Σ U C(k) U^T x`` (``slope``: with ``C'(k)``; ``transpose``: ``C^T``)."""
        u, c = self._stacked(k, slope=slope, which=which)
        if transpose:
            c = c.T
        return u @ (c @ self.layout.pair(u, np.where(self.free, x, 0)))

    def reshift(self, k: complex, *, solves: int = RESHIFT_SOLVES) -> None:
        """Refactorize at ``k``.

        A prepared handle factorizes ``T(k)`` on its solver plan when it can,
        or is replaced by a new one
        (`gfem._linalg.refactorize_or_prepare`); either way it is told it
        serves ``solves`` solves. An operator not prepared yet is prepared on
        first use.

        ``solves``: the shift-invert solves the new factorization is expected
        to serve: `RESHIFT_SOLVES` for a refinement reshift; the rule passes
        the seed's count when it moves back to the target for a seed solve.
        """
        if self._handle is not None:
            # everything that can refuse (the port term at ``k``) before the
            # handle is freed: a refusal leaves the operator at its shift
            stacked = self._stacked(k) if self.low_rank else None
            t = self._operator(k)
            if self._handle is not self._base:
                self._handle.free()  # the update's own columns
            self._base = refactorize_or_prepare(
                self._base,
                t,
                lambda t: self.prepare(t, solves),
                solves=solves,
            )
            self._handle = self._wrap(self._base, k, stacked)
        self.shift = k
        self.shifts += 1

    @property
    def general(self) -> bool:
        """Whether ``T^T != T``: the left eigenvector is not the right one."""
        return self.symmetry == "general"

    def free_handle(self) -> None:
        """Release the prepared operator."""
        if self._handle is not None and self._handle is not self._base:
            self._handle.free()
        if self._base is not None:
            self._base.free()
        self._handle = self._base = None

    @property
    def nbytes(self) -> int:
        """Host bytes held: the prepared handle's and the assembled maps'.

        A map tells its bytes by an ``nbytes`` attribute (0 when applied
        matrix-free); one without counts nothing.
        """
        held = 0 if self._handle is None else int(self._handle.nbytes)
        maps = [self.mass, self.damping, *(t.matrix for t in self.nonpolynomial)]
        return held + sum(int(getattr(m, "nbytes", 0)) for m in maps if m is not None)

    def solve(self, b: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """``T(sigma)^-1 b`` on the free DOFs (``transpose``: ``T(sigma)^-T b``)."""
        b = np.where(self.free, b, 0)
        if not transpose or not self.general:
            return self.handle.solve(b)
        return self.handle.solve(b, transpose=True)

    def _matvec(
        self, matrix: LinearMap, x: np.ndarray, *, transpose: bool = False
    ) -> np.ndarray:
        """A matrix (or its transpose) on the free DOFs (zero on the fixed rows)."""
        x = np.where(self.free, x, 0)
        y = matrix(x, transpose=True) if transpose and self.general else matrix(x)
        return np.where(self.free, y, 0)

    def _parts(
        self, x: np.ndarray, *, transpose: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """``(K x, D x, M x, [A_p x])`` on the free rows (or the transposes)."""
        transpose = transpose and self.general
        x = np.where(self.free, x, 0)
        mx = self._matvec(self.mass, x, transpose=transpose)
        dx = (
            np.zeros_like(mx)
            if self.damping is None
            else self._matvec(self.damping, x, transpose=transpose)
        )
        ax = [
            self._matvec(t.matrix, x, transpose=transpose) for t in self.nonpolynomial
        ]
        s = self.shift
        self.factorize()
        base = self._base
        if base is None:  # pragma: no cover - factorize sets it
            raise RuntimeError("The shift is not prepared.")
        # the factorized rest: the low-rank terms the update holds are not in it
        tx = base.apply(x, transpose=True) if transpose else base.apply(x)
        kx = np.where(self.free, tx, 0) - s * dx - s * s * mx
        for theta, a in zip(self._thetas, ax, strict=True):
            kx = kx - theta * a
        return kx, dx, mx, ax

    def parts(
        self, x: np.ndarray, *, transpose: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(K x, D x, M x)`` on the free rows (``transpose``: of ``K^T``, ...)."""
        kx, dx, mx, _ = self._parts(x, transpose=transpose)
        return kx, dx, mx

    def apply(
        self, k: complex, x: np.ndarray, *, transpose: bool = False
    ) -> np.ndarray:
        """``T(k) x`` (``transpose``: ``T(k)^T x``)."""
        kx, dx, mx, ax = self._parts(x, transpose=transpose)
        out = kx + k * dx + k * k * mx
        for t, a in zip(self.nonpolynomial, ax, strict=True):
            out = out + t.value(k) * a
        if self.low_rank:
            out = out + self._low_rank(k, x, transpose=transpose and self.general)
        return out

    def derivative(self, k: complex, x: np.ndarray) -> np.ndarray:
        """``T'(k) x = D x + 2 k M x + Σ θ_p'(k) A_p x + Σ U C'(k) U^T x``.

        ``'`` is ``d/dk``.

        The denominator of the eigenvalue derivative ``dk/dp = -y^T (dT/dp)
        x / y^T T'(k) x`` (``y`` the left eigenvector, ``x`` when complex
        symmetric; and of the Rayleigh functional's Newton step).
        """
        x = np.where(self.free, x, 0)
        out = 2 * k * self._matvec(self.mass, x)
        if self.damping is not None:
            out = out + self._matvec(self.damping, x)
        for t in self.nonpolynomial:
            out = out + t.derivative(k) * self._matvec(t.matrix, x)
        if self.low_rank:
            out = out + self._low_rank(k, x, slope=True)
        return out

    def curl_ratio(self, k: complex, x: np.ndarray) -> float:
        """``|x^H K_c x| / (|k|^2 |x^H M x|)``: curl energy over electric energy.

        ``K_c`` is ``K`` without `law_constant` (the curl-curl and the
        lumped inductances). A gradient field has none: at a zero of a
        law's eps(omega) it is an eigenvector of ``T`` (a longitudinal
        mode), with a ratio at round-off (1e-15 / (k h)^2); a transverse
        mode's is of order one (0.55 to 15 measured).
        """
        kx, _, mx = self.parts(x)
        if self.law_constant is not None:
            kx = kx - self._matvec(self.law_constant, x)
        # x^H A x: the load A x paired with the state conj(x)
        pair, xh = self.layout.pair, np.conj(x)
        electric = abs(k) ** 2 * abs(pair(mx, xh))
        return float(abs(pair(kx, xh)) / electric) if electric > 0 else math.inf

    # --- the frozen seed -----------------------------------------------------------

    @property
    def seed_damped(self) -> bool:
        """Whether the seed problem has a damping term (``D`` or a frozen one)."""
        return (
            self.damping is not None or bool(self.nonpolynomial) or bool(self.low_rank)
        )

    def seed_damping(self, x: np.ndarray) -> np.ndarray:
        """The seed's damping on ``x``: ``D`` plus each ``θ_p'(sigma) A_p``.

        And each low-rank ``U C'(sigma) U^T``; a term with ``seed_order``
        0 adds nothing. (The seed's ``K`` adds ``θ_p(sigma) - sigma
        θ_p'(sigma)`` per term, or ``θ_p(sigma)`` at order 0, which only
        enters through ``T(sigma)``.)
        """
        s = self.shift
        d = np.zeros(len(self.free), dtype=complex)
        if self.damping is not None:
            d = self._matvec(self.damping, x)
        for t in self.nonpolynomial:
            if t.seed_order:
                d = d + t.derivative(s) * self._matvec(t.matrix, x)
        sloped = tuple(t for t in self.low_rank if t.seed_order)
        if sloped:
            d = d + self._low_rank(s, x, slope=True, which=sloped)
        return d

    def seed_mass(self, x: np.ndarray) -> np.ndarray:
        """The seed's mass on ``x`` (``M``: the seed freezes no term into it)."""
        return self._matvec(self.mass, x)

    # --- the Rayleigh functional and the backward error -----------------------------

    def rayleigh(
        self, x: np.ndarray, k: complex, *, left: np.ndarray | None = None
    ) -> complex:
        """The root of ``y^T T(k) x = 0`` nearest ``k``, ``y`` = ``left`` or ``x``.

        A quadratic's root in closed form. With nonpolynomial terms, a
        Newton iteration that keeps the quadratic exact: each step takes the
        root nearest the current ``k`` of the quadratic with every term
        linearized around ``k`` (`NonPolynomial.linear`), so it stays on the
        branch of the start and converges quadratically. A low-rank term
        enters as the scalar ``y^T U C(k) U^T x``, linearized the same way.
        """
        y = x if left is None else np.where(self.free, left, 0)
        kx, dx, mx, ax = self._parts(x)

        def pair(load: np.ndarray) -> complex:
            # a scalar: a rank layout's sum is a 0-d array, which += below
            # would change in place
            return complex(self.layout.pair(load, y))

        a, b, c = pair(kx), pair(dx), pair(mx)
        if not self.nonpolynomial and not self.low_rank:
            return _nearest_root((a, b, c), k)
        pairs = [(t, pair(v)) for t, v in zip(self.nonpolynomial, ax, strict=True)]
        k = complex(k)
        for _ in range(_NEWTON_STEPS):
            linear = [a, b, c]
            for t, n in pairs:
                c0, c1 = t.linear(k)
                linear[0] += c0 * n
                linear[1] += c1 * n
            if self.low_rank:
                w = pair(self._low_rank(k, x))
                dw = pair(self._low_rank(k, x, slope=True))
                linear[0] += w - k * dw
                linear[1] += dw
            new = _nearest_root(linear, k)
            done = abs(new - k) <= _NEWTON_RTOL * abs(new)
            k = new
            if done:
                break
        return k

    @property
    def norms(self) -> tuple[float, ...]:
        """2-norm estimates of ``K``, ``D``, ``M`` and each ``A_p``.

        Those of their transposes too: ``|A^T|_2 = |A|_2``.
        """
        if self._norms is None:
            start = np.where(self.free, self._norm_start(), 0)

            def k_op(x, *, transpose=False):
                return self.parts(x, transpose=transpose)[0]

            def matrix_op(matrix):
                if matrix is None:
                    return lambda x, **_: np.zeros_like(x)
                return lambda x, *, transpose=False: self._matvec(
                    matrix, x, transpose=transpose
                )

            def norm(op):
                return _norm2(op, start, self.layout, general=self.general)

            self._norms = (
                norm(k_op),
                norm(matrix_op(self.damping)),
                norm(matrix_op(self.mass)),
                *(norm(matrix_op(t.matrix)) for t in self.nonpolynomial),
            )
        return self._norms

    def _norm_start(self) -> np.ndarray:
        """The power iterations' start: a state of the one-process numbers.

        ``default_rng(0)`` over the column on one process (`numbers` ``None``);
        else `_hashed` of each DOF's one-process number, the same state
        whatever the ranks hold.
        """
        if self.numbers is None:
            return np.random.default_rng(0).standard_normal(len(self.free))
        return _hashed(self.numbers)

    def _low_rank_norm(self, k: complex) -> float:
        """``|Σ U C(k) U^T|_2 = |R C(k) R^T|_2`` with ``U = Q R``.

        ``R`` is kept until ``U`` changes (an enrichment of its basis).
        """
        u, c = self._stacked(k)
        if self._range is None or not np.array_equal(self._range[0], u):
            self._range = (u, np.linalg.qr(u, mode="r"))
        r = self._range[1]
        return float(np.linalg.norm(r @ c @ r.T, 2))

    def backward_error(
        self, k: complex, x: np.ndarray, *, transpose: bool = False
    ) -> float:
        """``|T(k) x| / ((|K| + |k| |D| + |k|^2 |M| + Σ |θ_p(k)| |A_p| + |W|) |x|)``.

        ``W = Σ U_q C_q(k) U_q^T`` (`_low_rank_norm`). ``transpose``: of
        ``T(k)^T x``, a left eigenvector's.
        """
        nk, nd, nm, *na = self.norms
        layout = self.layout
        r = layout.norm(self.apply(k, x, transpose=transpose))
        den = nk + abs(k) * nd + abs(k) ** 2 * nm
        den += sum(
            abs(t.value(k)) * a for t, a in zip(self.nonpolynomial, na, strict=True)
        )
        if self.low_rank:
            den += self._low_rank_norm(k)
        den *= layout.state_norm(x)
        return float(r / den) if den > 0 else math.inf


def _nearest_root(coefficients, k: complex) -> complex:
    """The root of ``c0 + c1 z + c2 z^2`` nearest ``k``."""
    a, b, c = coefficients
    roots = np.roots([c, b, a]) if c != 0 else np.array([-a / b])
    return complex(roots[np.argmin(np.abs(roots - k))])


def _norm2(
    op, start: np.ndarray, layout: DofLayout = ONE_PROCESS, *, general: bool = False
) -> float:
    """``|A|_2``: power iteration on ``A^H A``.

    ``A^H y = conj(A^T conj(y))``, with ``A^T = A`` for a complex symmetric
    operator, and ``op(x, transpose=True)`` for a ``general`` one. ``op`` takes
    a state and gives a load, so each load is exchanged to a state before it is
    fed back (on one process the exchange is the identity).
    """
    x = start / layout.state_norm(start)
    value = 0.0
    for _ in range(_NORM_ITERATIONS):
        y = layout.exchange(op(x))
        z = layout.exchange(np.conj(op(np.conj(y), transpose=general)))  # A^H y
        n = layout.state_norm(z)
        if n == 0:
            return 0.0
        value = math.sqrt(n)
        x = z / n
    return value


def _hashed(numbers: np.ndarray) -> np.ndarray:
    """A value in ``[-1, 1)`` per integer, a function of that integer alone.

    SplitMix64's finalizer of each number, its top 53 bits as a fraction: a
    start vector without structure, as a random one, that every rank computes
    alike from the one-process DOF numbers.
    """
    z = np.asarray(numbers, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * 2.0**-52 - 1.0
