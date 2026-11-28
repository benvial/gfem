"""Shared checks for static physics values (host floats, never traced)."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from gfem.errors import NonDifferentiableError


def is_traced(value: Any) -> bool:
    """Whether ``value``, or any leaf of it as a JAX pytree, is a JAX tracer."""
    return any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(value))


def is_staging() -> bool:
    """Whether JAX is staging operations here (inside `jax.jit`, a scan body).

    A `jax.numpy` operation on constants returns a tracer only while JAX
    stages; under `jax.grad` or `jax.vmap` with concrete inputs it does not.
    """
    return isinstance(jnp.zeros(()) + 1.0, Tracer)


def same_lattice(a: Any, b: Any, rel: float = 1e-9) -> bool:
    r"""Whether two lists of translations are the same set, up to order and sign.

    A periodic pair and its reverse translate by :math:`\pm\mathbf a`, and
    a mesh read again may differ by round-off, also in a component that is
    zero: each translation matches one of the other list's within ``rel`` of
    the longest translation (not of its own component).
    """
    a = np.asarray(a, float).reshape(-1, 3)
    b = np.asarray(b, float).reshape(-1, 3)
    if len(a) != len(b):
        return False
    scale = float(np.linalg.norm(np.concatenate([a, b]), axis=1).max(initial=0.0))
    free = list(range(len(b)))
    for t in a:
        for j in free:
            if min(np.abs(t - b[j]).max(), np.abs(t + b[j]).max()) <= rel * scale:
                free.remove(j)
                break
        else:
            return False
    return True


def check_fixed(value: Any, what: str) -> None:
    """Raise if ``value``, a fixed physics value, is traced by JAX.

    Args:
        value: The user's value.
        what: The requirement, e.g. ``"Driven.frequency must be fixed
            numbers"``.

    Raises:
        NonDifferentiableError: ``value`` is traced by JAX.
    """
    if is_traced(value):
        raise NonDifferentiableError(
            f"{what}, not values computed inside jax.grad, jax.jit or jax.vmap: "
            "gfem does not differentiate with respect to it. Build it from "
            "concrete values, outside the transformed function."
        )


def static_float(value: Any, what: str, remedy: str) -> float:
    """A finite host float, or a physics error.

    Args:
        value: The user's value.
        what: What it is, e.g. ``"Conductor.conductivity"``.
        remedy: How to differentiate with respect to it, if traced.

    Raises:
        NonDifferentiableError: ``value`` is traced by JAX.
        ValueError: ``value`` is not a finite real number.
    """
    if is_traced(value):
        raise NonDifferentiableError(
            f"{what} is a fixed value and gfem does not differentiate with "
            f"respect to it. {remedy}"
        )
    try:
        out = float(np.asarray(value))
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a real number, got {value!r}.") from None
    if not math.isfinite(out):
        raise ValueError(f"{what} must be finite, got {out}.")
    return out


def optional_positive(value: Any, what: str, unit: str, remedy: str) -> float | None:
    """``None``, or a finite float > 0 (a physical size or circuit value)."""
    if value is None:
        return None
    out = static_float(value, what, remedy)
    if out <= 0:
        raise ValueError(f"{what} must be > 0 {unit}, got {out}.")
    return out


#: Relative slack on range limits: lengths given as fractions of another (a
#: rule as 1.25 radii, sigma as 4 pitches) land on the limit only to round-off.
SLACK = 1e-9


def metres_hint(length: float) -> str:
    """A reminder of the unit when a length looks like micrometres or more."""
    return " The lengths are in metres." if abs(length) >= 1e-3 else ""


def positive_length(value: Any, name: str, where: str, remedy: str) -> float:
    """A length as a finite host float > 0 (m), or raise.

    Args:
        value: The user's value.
        name: The argument's name, e.g. ``"min_width"``.
        where: The caller, e.g. ``"LayerPattern.check"``.
        remedy: How to differentiate with respect to it, if traced.

    Raises:
        TypeError: ``value`` is not a real number (``bool`` included).
        NonDifferentiableError: ``value`` is traced by JAX.
        ValueError: ``value`` not finite or not > 0.
    """
    if not is_traced(value) and (
        isinstance(value, bool | np.bool_)
        or np.ndim(value) != 0
        or np.asarray(value).dtype.kind not in "iuf"
    ):
        raise TypeError(
            f"{where}: {name} is a length in metres, a real number, got {value!r}."
        )
    out = static_float(value, f"{where} {name}", remedy)
    if not out > 0:
        raise ValueError(
            f"{where}: {name} must be a finite length > 0 m, got {out:g} m."
        )
    return out
