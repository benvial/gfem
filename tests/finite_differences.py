"""Finite-difference checks of gradients through gfem solves.

One place for every gradient test's differences, so they share one error
budget. A solve's objective carries round-off noise ``delta`` of about
1e-15 relative: threaded assembly sums in a different order on every run,
and an iterative solve stops at its tolerance. A difference with step ``h``
then has two errors:

- truncation: ``h**2 J'''/6`` for the second-order central difference,
  ``h**4 J'''''/30`` for the fourth-order one used here;
- round-off: about ``delta/h`` (``1.5 delta/h`` for fourth order).

Both are measured against the directional derivative ``grad . d``, which
for a random ``d`` can sit far below ``|grad|`` (a direction nearly
orthogonal to the gradient). With second-order differences, a step small
enough to keep the truncation below a 1e-6 relative gate puts the
round-off at the gate too, so such a check fails at random under load.
Fourth-order differences let the step grow by 10-30x: both errors then sit
1-2 orders of magnitude below the gate.

Choosing ``step``: the largest step whose fourth-order truncation stays
below ~1e-8 relative. The scale is set by how strongly the objective
responds to the input, not by the input's magnitude: e.g. 1e-2 for a
relative permittivity of a few, 5e-2 for a conductivity of ~1 S/m in a
low-loss line. ``directions`` are unit vectors, so ``step`` is the norm of
the perturbation.

Usage::

    from finite_differences import assert_gradient_matches, unit_directions

    directions = unit_directions(rng, p.shape, 3)
    assert_gradient_matches(objective, p, directions, step=1e-2)
"""

from collections.abc import Callable, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "RTOL",
    "assert_gradient_matches",
    "assert_taylor_second_order",
    "derivative",
    "unit_directions",
]

RTOL = 1e-6
"""The gate: relative error of ``grad . d`` against the difference."""

Objective = Callable[[jax.Array], jax.Array]


def unit_directions(rng: np.random.Generator, shape, n: int) -> list[np.ndarray]:
    """``n`` random real directions of unit norm."""
    out = []
    for _ in range(n):
        d = rng.standard_normal(shape)
        out.append(d / np.linalg.norm(d))
    return out


def _samples(objective: Objective, p, d, h: float) -> list[float]:
    return [float(jnp.real(objective(p + k * h * d))) for k in (-2, -1, 1, 2)]


def _fourth_order(f: list[float], h: float) -> float:
    return (8 * (f[2] - f[1]) - (f[3] - f[0])) / (12 * h)


def derivative(objective: Objective, p, d, h: float) -> float:
    """Fourth-order central difference of ``objective`` at ``p`` along ``d``."""
    return _fourth_order(_samples(objective, p, d, h), h)


def _along(grad, d) -> float:
    # JAX's gradient over a complex leaf is dJ/dx - i dJ/dy, so the
    # derivative along d = a + i b is grad_x . a + grad_y . b = Re(grad . d).
    return float(jnp.real(jnp.sum(jnp.asarray(grad) * d)))


def assert_gradient_matches(
    objective: Objective,
    p,
    directions: Iterable,
    *,
    step: float,
    rel: float = RTOL,
    reference: Objective | None = None,
    grad=None,
) -> None:
    """``grad . d`` equals the fourth-order difference along every ``d``.

    Args:
        objective: Real scalar function of ``p``.
        p: Input array (real or complex).
        directions: Unit directions (real; multiply by ``1j`` to check the
            imaginary part of a complex input).
        step: Difference step (see the module docstring).
        rel: Relative tolerance on each directional derivative.
        reference: Function differenced instead of ``objective`` (e.g. the
            same objective on a direct solve when the solver under test
            leaves a residual that would show in the differences).
        grad: Gradient to check; ``jax.grad(objective)(p)`` when omitted.
    """
    if grad is None:
        grad = jax.grad(objective)(p)
    reference = reference or objective
    for i, d in enumerate(directions):
        got = _along(grad, d)
        f = _samples(reference, p, d, step)
        fd = _fourth_order(f, step)
        if abs(got - fd) > rel * abs(fd):
            fd2 = (f[2] - f[1]) / (2 * step)
            msg = (
                f"direction {i}: gradient {got:.12e} vs finite difference "
                f"{fd:.12e} (relative error {abs(got - fd) / abs(fd):.2e} > {rel:.0e}; "
                f"step {step:g}, second-order difference {fd2:.12e})"
            )
            raise AssertionError(msg)


def assert_taylor_second_order(
    objective: Objective,
    p,
    d,
    steps: Sequence[float],
    *,
    grad=None,
    tolerance: float = 0.1,
) -> None:
    """``|J(p + h d) - J(p) - h grad . d|`` falls as ``h**2`` over ``steps``.

    Steps must be large enough that the remainder sits well above the
    objective's round-off (``h**2 J''`` >> 1e-15 ``|J|``).
    """
    j0 = jnp.real(objective(p))
    if grad is None:
        grad = jax.grad(objective)(p)
    slope = _along(grad, d)
    remainders = [
        abs(float(jnp.real(objective(p + h * d)) - j0) - h * slope) for h in steps
    ]
    slopes = np.diff(np.log(remainders)) / np.diff(np.log(steps))
    if not np.allclose(slopes, 2.0, rtol=0, atol=tolerance):
        msg = f"Taylor remainder slopes {slopes} (expected 2), remainders {remainders}"
        raise AssertionError(msg)
