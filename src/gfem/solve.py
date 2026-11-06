"""`gfem.solve`: the one verb that turns a model and a problem into a result."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, overload

from gfem import _mpi
from gfem._adjoint import solve_vjp
from gfem._checks import is_traced
from gfem._periodic import check_nodes
from gfem.errors import LinearSolverConvergenceError
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
    model: Model,
    problem: Electrostatic,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> ElectrostaticResult: ...
@overload
def solve(
    model: Model,
    problem: Driven,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> DrivenResult: ...
@overload
def solve(
    model: Model,
    problem: Eigenmode,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> EigenmodeResult: ...
@overload
def solve(
    model: Model,
    problem: PortMode,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> PortModeResult: ...
def solve(
    model: Model,
    problem: Any,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> Any:
    """Solve one problem type on a model.

    A pure function of its inputs: the model and problem are not changed,
    and the same inputs give the same result.

    It is also a JAX function of the model's arrays (the mesh node
    coordinates and every `MaterialField` array): `jax.grad`, `jax.jit`
    and `jax.vmap` (sequential solves) apply. Each input then
    has a gradient (for an electrostatic capacitance: the `MaterialField`
    ``eps_r`` values), an exact zero gradient if the problem never reads it
    (``conductivity`` in electrostatics), or raises
    `gfem.NonDifferentiableError` when `jax.jit` or `jax.grad` first runs
    the function, if the problem reads it
    but gfem cannot differentiate it (for example a problem value such as
    a frequency, or a field-derived output). `jax.lax.stop_gradient` on an
    input or an output is the opt-out. Only reverse mode is supported.

    Example:
        S-parameters of a model over a band::

            result = gfem.solve(model, gfem.Driven(frequency=[4e9, 5e9, 6e9]))
            s21 = result.s["p2", "p1"]  # S21 over frequency

    Args:
        model: The device.
        problem: What to solve, e.g. `gfem.Electrostatic`.
        options: The numerics; ``None`` is ``SolverOptions()``.
        comm: An mpi4py communicator. ``None`` lets the launcher decide: one
            process without one, the world communicator under ``mpiexec`` or
            ``srun``, where every rank makes the same call. A communicator of
            size 1, such as ``MPI.COMM_SELF``, runs one independent copy.

    Returns:
        The problem type's result, e.g. `gfem.ElectrostaticResult` for an
        `Electrostatic` problem.

    Raises:
        TypeError: An argument has the wrong type.
        DistributedUnsupportedError: On several ranks, this solve has no
            distributed path; ``feature`` names what, the message the remedy.
        SolverUnavailableError: Launched on several ranks without a usable
            mpi4py and Open MPI.
        RuntimeError: On several ranks, the ranks hold different inputs;
            the message names one.
        RegionNotFoundError: The problem names a region the mesh lacks.
        ValueError: The problem does not fit the model; the message gives
            the remedy.
        NonDifferentiableError: Under `jax.grad`, an input or output gfem
            does not differentiate is perturbed or used.
        LinearSolverConvergenceError: An iterative linear solve
            (``linear_solver="bddc_gmres"``, or ``"petsc"`` with a Krylov
            ``ksp_type``) did not converge; the message gives the remedy. A
            uniform `gfem.Driven` sweep stops there, and the error's
            ``result`` holds the frequencies it solved, ``unsolved`` the
            rest.
    """
    if not isinstance(model, Model):
        raise TypeError(f"solve needs a gfem.Model, got {type(model).__name__}.")
    if options is None:
        options = SolverOptions()
    elif not isinstance(options, SolverOptions):
        raise TypeError(
            f"options must be a gfem.SolverOptions, got {type(options).__name__}."
        )
    resolved = _mpi.resolve_comm(comm)
    if resolved is not None:
        _mpi.refuse(model, problem, options)
    rule_class = _rule_class(problem)
    if resolved is None:
        rule = rule_class(model, problem, options)
    else:
        _mpi.entry_check(resolved, "solve", model, problem, options)
        try:
            rule = rule_class(model, problem, options, comm=resolved)
        except (TypeError, ValueError) as err:
            # the rule reads inputs the entry check has just found equal
            _mpi.agreed(err)
            raise
    if is_traced((model, problem)):
        return rule.result(solve_vjp(rule, model, problem))
    # A moved mesh must keep its periodic pairs and stay valid on the solve
    # domain.
    check_nodes(model.mesh.nodes, model._periodic)  # noqa: SLF001
    _check_quality(
        model.mesh,
        model._domain.active,  # noqa: SLF001
        stacklevel=3,
        comm=resolved,
    )
    # a uniform sweep's partial result, built here: never in a JAX callback
    partial_result = getattr(rule, "partial_result", None)
    if partial_result is not None:
        rule.keeps_partial = True
    try:
        with _mpi.section(resolved, agreed=_mpi.FORWARD_ERRORS):
            outputs, _ = rule.forward(model, keep=False)
    except LinearSolverConvergenceError as err:
        if partial_result is not None:
            err.result = partial_result(err)
        raise
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
