"""The `solve` entry point: one verb for every problem type."""

from __future__ import annotations

from typing import Any

from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import PortMode
from gfem.results import PortModeResult

__all__ = ["solve"]


def solve(
    model: Model, problem: PortMode, *, options: SolverOptions | None = None
) -> PortModeResult:
    """Solve one problem on a model.

    A pure function of its inputs: the model holds the physics, the problem
    says what to compute and the options hold every numerical knob.

    Args:
        model: The model.
        problem: What to compute: `PortMode`.
        options: Numerical options; ``None`` uses `SolverOptions()`.

    A `PortMode` problem reads only ``options.order``: port cross-sections
    are small, so they always use a sparse LU factorization with ARPACK on
    the CPU.

    Returns:
        The problem type's result: `PortModeResult` for `PortMode`.

    Raises:
        TypeError: ``model``, ``problem`` or ``options`` has the wrong type.
    """
    if not isinstance(model, Model):
        raise TypeError(f"solve needs a gfem.Model, got {type(model).__name__}.")
    options = SolverOptions() if options is None else options
    if not isinstance(options, SolverOptions):
        raise TypeError(
            f"options must be a gfem.SolverOptions, got {type(options).__name__}."
        )
    return _dispatch(model, problem, options)


def _dispatch(model: Model, problem: Any, options: SolverOptions) -> Any:
    if isinstance(problem, PortMode):
        from gfem._sweep.port_mode import solve_port_mode

        return solve_port_mode(model, problem, options)
    raise TypeError(
        f"solve got an unknown problem {problem!r}; give a problem type such as "
        "gfem.PortMode(...)."
    )
