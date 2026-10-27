"""Shift-invert Arnoldi (SciPy ARPACK) on the eigenmode seed operator.

ARPACK's implicitly restarted Arnoldi finds the largest eigenvalues of the
shared seed operator (`gfem._eigen.seed`, which derives the shift-invert
and companion forms): ``T(sigma)^-1`` through the eigen operator's linear
solver, the divergence-free projection on every vector, and a fixed,
projected start vector, so a solve is deterministic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse.linalg as spla

from gfem._eigen.operator import EigenOperator
from gfem._eigen.seed import seed_operator

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
    seed = seed_operator(op, project)
    nev, ncv = seed.dimensions(nev)
    try:
        theta, vectors = spla.eigs(
            _Operator(seed.size, seed.apply),
            k=nev,
            which="LM",
            tol=tol,
            ncv=ncv,
            v0=seed.start,
        )
    except spla.ArpackNoConvergence as err:
        theta, vectors = err.eigenvalues, err.eigenvectors
    return seed.pairs(theta, vectors)
