"""The seed operator: what a shift-invert eigen solver iterates on.

Undamped (``D = 0``): the problem is linear in ``mu = k^2``,
``K x = mu (-M) x``, and the seed solve runs on ``(K + sigma^2 M)^-1 (-M) =
T(sigma)^-1 (-M)``, whose largest eigenvalues ``1 / (mu - sigma^2)`` are
nearest the shift.

Damped: the quadratic problem ``(K + k D + k^2 M) x = 0`` in its first
companion linearization, ``v = k x``,

    [[0, I], [-K, -D]] [x; v] = k [[I, 0], [0, M]] [x; v],

whose shift-invert step needs one solve with ``T(sigma)``: for
``(A - sigma B)^-1 B [y1; y2]``, ``a = -T(sigma)^-1 (M y2 + (D + sigma M)
y1)`` and ``b = y1 + sigma a``. The 2n system is never factorized.

``K``, ``D`` and ``M`` are the seed's: a nonlinear problem's nonpolynomial
terms frozen at the shift (`EigenOperator.seed_damping`, `seed_mass`).

``P`` is the divergence-free projection. The undamped output is projected.
A companion output is ``[P a; y1 + sigma P a]``, one projection per apply:
it equals the projected step ``[P a; P b]`` wherever ``P y1 = y1``, which
every output satisfies (ADR 0035). The start vector is a fixed, projected
vector, so a seed solve is deterministic. The eigen solvers
(`gfem._eigen.arpack`, `gfem._eigen.slepc`) only find the largest
eigenvalues ``theta`` of `SeedOperator.apply`; `SeedOperator.pairs` maps them
back to wavenumbers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from gfem._eigen.operator import EigenOperator

__all__ = ["MIN_SUBSPACE", "SeedOperator", "seed_operator"]

#: The smallest subspace (ncv) a seed solve builds: it applies the seed
#: operator, one ``T(sigma)`` solve, at least this many times per shift.
MIN_SUBSPACE = 20


@dataclasses.dataclass(frozen=True)
class SeedOperator:
    """The shift-inverted seed operator of an `EigenOperator`.

    Attributes:
        size: Its dimension: ``n``, or ``2 n`` for the companion form.
        n: The eigen operator's free DOFs.
        apply: ``y -> A y`` on complex ``(size,)`` vectors.
        start: The projected start vector, ``(size,)``.
        shift: The wavenumber ``sigma`` it is shift-inverted at.
        damped: Whether it is the companion form.
    """

    size: int
    n: int
    apply: Callable[[np.ndarray], np.ndarray]
    start: np.ndarray
    shift: complex
    damped: bool

    def dimensions(self, nev: int) -> tuple[int, int]:
        """``(nev, ncv)``: ``nev`` clamped to the operator, and the subspace size."""
        nev = max(1, min(nev, self.size - 2))
        return nev, min(self.size, max(2 * nev + 1, MIN_SUBSPACE))

    def pairs(
        self, theta: np.ndarray, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Wavenumbers ``(m,)`` and eigenvectors ``(n, m)`` of eigenpairs of `apply`.

        Args:
            theta: Eigenvalues of `apply`, ``(m',)``; zeros are dropped.
            vectors: Their eigenvectors, ``(size, m')``.
        """
        keep = np.abs(theta) > 0
        theta, vectors = theta[keep], vectors[: self.n, keep]
        s = self.shift
        k = s + 1 / theta if self.damped else np.sqrt(s * s + 1 / theta)
        return k, vectors


def seed_operator(op: EigenOperator, project) -> SeedOperator:
    """The seed operator of ``op`` (``T(sigma)`` prepared) and the projection."""
    free = op.free
    n = len(free)
    s = op.shift

    mass = op.seed_mass
    damped = op.seed_damped
    if not damped:
        size = n

        def apply(y):
            return project(op.solve(-mass(y)))

        start = project(free.astype(complex))
    else:
        size = 2 * n

        def apply(y):
            # [P a; y1 + s P a]: equal to [P a; P(y1 + s a)] where P y1 = y1,
            # which holds for every Krylov vector from the projected start,
            # and both map into that set (ADR 0035)
            y1, y2 = y[:n], y[n:]
            a = -op.solve(mass(y2 + s * y1) + op.seed_damping(y1))
            pa = project(a)
            return np.concatenate([pa, y1 + s * pa])

        x0 = project(free.astype(complex))
        start = np.concatenate([x0, s * x0])
    return SeedOperator(size, n, apply, start, s, damped)
