"""Shared checks for static physics values (host floats, never traced)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from jax.core import Tracer

from gfem.errors import NonDifferentiableError


def is_traced(value: Any) -> bool:
    """Whether ``value`` (or anything inside a tuple / list) is a JAX tracer."""
    if isinstance(value, Tracer):
        return True
    if isinstance(value, (tuple, list)):
        return any(is_traced(v) for v in value)
    return False


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
            f"{what}, not traced by JAX: gfem does not differentiate with respect "
            "to it. Build it from concrete values, outside the function JAX "
            "transforms."
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
