"""`gfem.solve`: the one verb that turns a model and a problem into a result."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, overload

from gfem._adjoint import solve_vjp
from gfem._checks import is_traced
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.shape import _check_quality

if TYPE_CHECKING:
    from gfem.problems import Driven, Eigenmode, Electrostatic, PortMode
    from gfem.results import (
        DrivenResult,
        EigenmodeResult,
        ElectrostaticResult,
        PortModeResult,
    )

__all__ = ["solve"]


@overload
def solve(
    model: Model, problem: Electrostatic, *, options: SolverOptions | None = None
) -> ElectrostaticResult: ...
@overload
def solve(
    model: Model, problem: Driven, *, options: SolverOptions | None = None
) -> DrivenResult: ...
@overload
def solve(
    model: Model, problem: Eigenmode, *, options: SolverOptions | None = None
) -> EigenmodeResult: ...
@overload
def solve(
    model: Model, problem: PortMode, *, options: SolverOptions | None = None
) -> PortModeResult: ...
def solve(model: Model, problem: Any, *, options: SolverOptions | None = None) -> Any:
    """Solve one problem type on a model.

    A pure function of its inputs: the model and problem are not changed,
    and the same inputs give the same result.

    It is also a JAX function of the model's array leaves: `jax.grad`,
    `jax.jit` and `jax.vmap` (sequential solves) apply. Each input then
    has a gradient (for an electrostatic capacitance: the `MaterialField`
    ``eps_r`` values), an exact zero gradient if the problem never reads it
    (``conductivity`` in electrostatics), or raises
    `gfem.NonDifferentiableError` while JAX traces, if the problem reads it
    but gfem cannot differentiate it (for example a problem value such as
    a frequency, or a field-derived output). `jax.lax.stop_gradient` on an
    input or an output is the opt-out. Only reverse mode is supported.

    Args:
        model: The device.
        problem: What to solve, e.g. `gfem.Electrostatic`.
        options: The numerics; ``None`` is ``SolverOptions()``.

    Returns:
        The problem type's result, e.g. `gfem.ElectrostaticResult` for an
        `Electrostatic` problem.

    Raises:
        TypeError: An argument has the wrong type.
        RegionNotFoundError: The problem names a region the mesh lacks.
        ValueError: The problem does not fit the model; the message gives
            the remedy.
        NonDifferentiableError: Under `jax.grad`, an input or output gfem
            does not differentiate is perturbed or used.
    """
    if not isinstance(model, Model):
        raise TypeError(f"solve needs a gfem.Model, got {type(model).__name__}.")
    if options is None:
        options = SolverOptions()
    elif not isinstance(options, SolverOptions):
        raise TypeError(
            f"options must be a gfem.SolverOptions, got {type(options).__name__}."
        )
    rule = _rule_class(problem)(model, problem, options)
    if is_traced((model, problem)):
        return rule.result(solve_vjp(rule, model, problem))
    # A moved mesh must still be valid on the solve domain.
    _check_quality(model.mesh, model._domain.active, stacklevel=3)  # noqa: SLF001
    outputs, _ = rule.forward(model, keep=False)
    return rule.result(outputs)


def _rule_class(problem: Any) -> type:
    """The engine rule of a problem type, imported on first use.

    Each problem class names its rule as ``_rule = "module:Class"``, so a new
    problem type needs no change here, and importing gfem does not import
    the engine.
    """
    path = getattr(type(problem), "_rule", None)
    if not isinstance(path, str):
        raise TypeError(
            "solve needs a problem type such as gfem.Electrostatic(...), got "
            f"{type(problem).__name__}."
        )
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)
