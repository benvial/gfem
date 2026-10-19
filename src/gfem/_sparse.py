"""Constant sparse matrices and SPD solves applied with JAX.

Auxiliary linear maps assembled once in NumPy (the shape extension, the
density filter) run under `jax.grad`, `jax.jit` and `jax.vmap` through these
two pieces: `Coo.matvec` is a JAX-traceable (so transposable) product, and
`spd_solve` solves with a host factorization behind
`jax.lax.custom_linear_solve`, whose adjoint is the same solve.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


class Coo:
    """A constant sparse matrix applied with JAX (linear in its argument)."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape):
        self.rows, self.cols, self.vals = rows, cols, vals
        self.n, self.m = shape

    def matvec(self, v: jax.Array) -> jax.Array:
        """The product with a vector ``(m,)``."""
        return jax.ops.segment_sum(
            jnp.asarray(self.vals) * v[self.cols], self.rows, num_segments=self.n
        )


def coo(a: sp.sparray | sp.spmatrix) -> Coo:
    """A `Coo` of a SciPy sparse matrix."""
    a = sp.coo_array(a)
    return Coo(a.row.astype(np.int64), a.col.astype(np.int64), a.data, a.shape)


def spd_solve(matrix: Coo, factor: Any, b: jax.Array) -> jax.Array:
    """``matrix^-1 b`` for a symmetric positive definite matrix.

    Args:
        matrix: The matrix, for JAX's derivative rules.
        factor: Its factorization: ``factor.solve(b)`` for a NumPy ``(n,)``.
        b: The right-hand side ``(n,)``, real.
    """

    def solve(_matvec: Callable, rhs: jax.Array) -> jax.Array:
        return jax.pure_callback(
            factor.solve,
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            rhs,
            vmap_method="sequential",
        )

    return jax.lax.custom_linear_solve(matrix.matvec, b, solve=solve, symmetric=True)
