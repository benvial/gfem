"""Shift-invert Arnoldi (SciPy ARPACK) on the eigenmode operator.

Undamped (``D = 0``): the problem is linear in ``mu = k^2``,
``K x = mu (-M) x``, and ARPACK runs on ``(K + sigma^2 M)^-1 (-M) = T(sigma)^-1
(-M)``, whose largest eigenvalues ``1 / (mu - sigma^2)`` are nearest the
shift.

Damped: the quadratic problem ``(K + k D + k^2 M) x = 0`` in its first
companion linearization, ``v = k x``,

    [[0, I], [-K, -D]] [x; v] = k [[I, 0], [0, M]] [x; v],

whose shift-invert step needs one solve with ``T(sigma)``: for
``(A - sigma B)^-1 B [y1; y2]``, ``a = -T(sigma)^-1 (M y2 + (D + sigma M)
y1)`` and ``b = y1 + sigma a``. The 2n system is never factorized.

``K``, ``D`` and ``M`` are the seed's: a nonlinear problem's nonpolynomial
terms frozen at the shift (`EigenOperator.seed_damping`, `seed_mass`).

Every Arnoldi vector (each half of a companion vector) goes through the
divergence-free projection, and the start vector is a fixed, projected
vector, so a solve is deterministic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse.linalg as spla

from gfem._eigen.operator import EigenOperator

__all__ = ["shift_invert"]


class _Operator(spla.LinearOperator):
    """A complex ``size x size`` operator from its matrix-vector product."""

    def __init__(self, size: int, apply: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(dtype=np.dtype(complex), shape=(size, size))
        self._apply = apply

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._apply(np.asarray(x).reshape(-1))


def shift_invert(
    op: EigenOperator,
    project,
    nev: int,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``nev`` eigenpairs ``(k, x)`` of ``T`` nearest ``op.shift``.

    Args:
        op: The operator with ``T(sigma)`` prepared.
        project: The divergence-free projection, ``x -> x``.
        nev: How many pairs to ask ARPACK for.
        tol: ARPACK's relative tolerance on the shift-inverted eigenvalues.

    Returns:
        Wavenumbers ``(m,)`` (per file unit, ``m <= nev``; may include
        pairs with ``Re k <= 0``) and eigenvectors ``(n, m)``. If ARPACK
        stops early, the pairs it converged.
    """
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
            y1, y2 = y[:n], y[n:]
            d1 = op.seed_damping(y1)
            a = -op.solve(mass(y2) + d1 + s * mass(y1))
            return np.concatenate([project(a), project(y1 + s * a)])

        x0 = project(free.astype(complex))
        start = np.concatenate([x0, s * x0])
    linear = _Operator(size, apply)
    nev = max(1, min(nev, size - 2))
    ncv = min(size, max(2 * nev + 1, 20))
    try:
        theta, vectors = spla.eigs(
            linear, k=nev, which="LM", tol=tol, ncv=ncv, v0=start
        )
    except spla.ArpackNoConvergence as err:
        theta, vectors = err.eigenvalues, err.eigenvectors
    keep = np.abs(theta) > 0
    theta, vectors = theta[keep], vectors[:n, keep]
    k = s + 1 / theta if damped else np.sqrt(s * s + 1 / theta)
    return k, vectors
