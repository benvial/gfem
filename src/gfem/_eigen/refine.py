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

A general ``T`` (a Bloch wave vector, `EigenOperator.general`) is refined
two-sided: a left vector ``y`` (``T(k)^T y = 0``), started at ``conj(x)``
(exact for a lossless, undamped cell), takes the same steps on the transposed
system, ``y <- y - T(sigma)^-T T(k)^T y`` on the same factorization; ``k``
comes from ``y^T T(k) x = 0``, second order in both errors, and a pair
converges when both backward errors meet the tolerance.

The vectors are state columns: their norms are the layout's
(`gfem._linalg.layout.DofLayout.state_norm`), over owned DOFs under ranks.

For a polynomial ``T`` an Arnoldi pair usually meets the tolerance at once
(step 0 is only the Rayleigh functional, which sharpens ``k`` to second
order in the eigenvector's error); the iteration repairs the pairs that the
divergence-free projection (exact only without boundary damping) or an
early stop of the seed solve left short.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from gfem._eigen.operator import EigenOperator
from gfem._engine.port_term import PortTermRefusalError

__all__ = ["MAX_SHIFTS", "Refined", "refine"]

#: Refactorizations allowed per refined mode.
MAX_SHIFTS = 3
#: A seed of a frozen (nonpolynomial) operator this close to an eigenpair
#: (backward error) has a Ritz value accurate enough to refactorize at; a
#: rougher one is refined on the current shift, which pulls it to the mode
#: nearest that shift. A polynomial operator's Ritz values are its own, so
#: its seeds are always refactorized at.
TRUSTED = 1e-6


@dataclasses.dataclass(frozen=True)
class Refined:
    """A refined eigenpair.

    Attributes:
        k: Wavenumber per file unit.
        x: Eigenvector (unit 2-norm).
        error: Its backward error (the larger of ``x``'s and ``y``'s).
        iterations: Correction steps taken.
        converged: Whether ``error`` met the tolerance.
        y: The left eigenvector (unit 2-norm) of a general operator,
            ``None`` when complex symmetric (the left vector is ``x``).
        refusal: The wave port term's refusal that stopped the refinement
            (the seed is lost: ``converged`` is False and the pair is the
            best seen before it), ``None`` otherwise.
    """

    k: complex
    x: np.ndarray
    error: float
    iterations: int
    converged: bool
    y: np.ndarray | None = None
    refusal: PortTermRefusalError | None = None


def refine(
    op: EigenOperator,
    k: complex,
    x: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    reshift: bool = True,
    reshift_first: bool = False,
) -> Refined:
    """Refine the pair ``(k, x)`` to backward error ``tolerance``.

    ``reshift_first``: refactorize at ``k`` before the first correction
    step (the current shift is nearer another mode than this one), unless
    the operator is nonpolynomial and the seed's backward error is above
    `TRUSTED`.

    Returns the best pair seen (``converged`` False if it misses the
    tolerance after ``max_iterations`` steps, or if the wave port term
    refuses an iterate: a lost seed, ``refusal`` set). With ``reshift`` False the
    operator is never refactorized.
    """
    # the eigenvector lives on the free DOFs: weight on the fixed ones would
    # hide the residual from the backward error (corrections never touch it)
    x = np.where(op.free, x, 0)
    norm = op.layout.state_norm
    x = x / norm(x)
    y = np.conj(x) if op.general else None
    best: Refined | None = None
    history: list[float] = []
    shifts = 0
    try:
        for step in range(max_iterations + 1):
            k = op.rayleigh(x, k, left=y)
            error = op.backward_error(k, x)
            if y is not None:
                error = max(error, op.backward_error(k, y, transpose=True))
            history.append(error)
            if best is None or error < best.error:
                best = Refined(k, x, error, step, error <= tolerance, y)
            if error <= tolerance or step == max_iterations:
                break
            stalled = len(history) >= 3 and history[-1] > 0.5 * history[-3]
            rough = bool(op.nonpolynomial) and error > TRUSTED
            trusted = reshift_first and step == 0 and not rough
            if reshift and (trusted or (stalled and shifts < MAX_SHIFTS)):
                op.reshift(k)
                shifts += 1
                history.clear()
            x = x - op.solve(op.apply(k, x))
            x = x / norm(x)
            if y is not None:
                y = y - op.solve(op.apply(k, y, transpose=True), transpose=True)
                y = y / norm(y)
    except PortTermRefusalError as refusal:
        # an iterate left the range where the port term follows its mode: the
        # seed is lost, with the best pair seen (or the seed itself)
        if best is None:
            best = Refined(k, x, math.inf, 0, False, y)
        return dataclasses.replace(best, converged=False, refusal=refusal)
    if best is None:  # pragma: no cover - the loop runs at least once
        raise RuntimeError("No refinement step ran.")
    return best
