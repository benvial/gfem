"""Residual inverse iteration (Neumaier 1985) on the shift factorization.

Each step takes ``k`` from the two-sided Rayleigh functional
``x^T T(k) x = 0`` (``T`` is complex symmetric, so the left eigenvector is
``x`` unconjugated) and corrects ``x <- x - T(sigma)^-1 T(k) x``, reusing
the ``T(sigma)`` factorization of the shift-invert Arnoldi. It stops when
the backward error reaches the eigen tolerance.

The error contracts by about ``|k - sigma| / |k' - sigma|`` per step (``k'``
the next eigenvalue). When it contracts by less than 2x over two steps, the
operator is refactorized at the current ``k`` (complex: a low-Q mode is far
from any real shift), at most `MAX_SHIFTS` times per mode; the iteration
then converges like Rayleigh quotient iteration.

For a polynomial ``T`` an Arnoldi pair usually meets the tolerance at once
(step 0 is only the Rayleigh functional, which sharpens ``k`` to second
order in the eigenvector's error); the iteration repairs the pairs that the
divergence-free projection (exact only without boundary damping) or an
early ARPACK stop left short.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from gfem._eigen.operator import EigenOperator

__all__ = ["MAX_SHIFTS", "Refined", "refine"]

#: Refactorizations allowed per refined mode.
MAX_SHIFTS = 3


@dataclasses.dataclass(frozen=True)
class Refined:
    """A refined eigenpair.

    Attributes:
        k: Wavenumber per file unit.
        x: Eigenvector (unit 2-norm).
        error: Its backward error.
        iterations: Correction steps taken.
        converged: Whether ``error`` met the tolerance.
    """

    k: complex
    x: np.ndarray
    error: float
    iterations: int
    converged: bool


def refine(
    op: EigenOperator,
    k: complex,
    x: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    reshift: bool = False,
) -> Refined:
    """Refine the pair ``(k, x)`` to backward error ``tolerance``.

    ``reshift``: refactorize at ``k`` before the first correction step (the
    current shift is nearer another mode than this one).

    Returns the best pair seen (``converged`` False if it misses the
    tolerance after ``max_iterations`` steps).
    """
    x = x / np.linalg.norm(x)
    best: Refined | None = None
    history: list[float] = []
    shifts = 0
    for step in range(max_iterations + 1):
        k = op.rayleigh(x, k)
        error = op.backward_error(k, x)
        history.append(error)
        if best is None or error < best.error:
            best = Refined(k, x, error, step, error <= tolerance)
        if error <= tolerance or step == max_iterations:
            break
        stalled = len(history) >= 3 and history[-1] > 0.5 * history[-3]
        if (reshift and step == 0) or (stalled and shifts < MAX_SHIFTS):
            op.reshift(k)
            shifts += 1
            history.clear()
        x = x - op.solve(op.apply(k, x))
        x = x / np.linalg.norm(x)
    if best is None:  # pragma: no cover - the loop runs at least once
        raise RuntimeError("No refinement step ran.")
    return best
