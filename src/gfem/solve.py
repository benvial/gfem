"""`gfem.solve`: the one verb that turns a model and a problem into a result."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from typing import TYPE_CHECKING, Any, overload

from gfem import _mpi, _timing
from gfem._adjoint import solve_vjp
from gfem._checks import is_traced
from gfem._dim2 import refuse_solve
from gfem._domain import physics_of
from gfem._engine.solve_request import SolveRequest
from gfem._materials import MaterialField
from gfem._periodic import check_nodes
from gfem._pml import check_layer_nodes
from gfem.errors import LinearSolverConvergenceError
from gfem.model import _PHYSICS_NAMES, Model
from gfem.options import SolverOptions
from gfem.problems import check_problem
from gfem.shape import _check_quality

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gfem._eigen.rule import DualRequest
    from gfem._engine.driven import AdjointLoads
    from gfem._solve_cache import SolveCache
    from gfem.problems import (
        DCConduction,
        Driven,
        Eigenmode,
        Electrostatic,
        MechanicalEigenmode,
        Piezoelectric,
        PortMode,
        SteadyThermal,
        TransientThermal,
    )
    from gfem.results import (
        DCConductionResult,
        DrivenResult,
        EigenmodeResult,
        ElectrostaticResult,
        MechanicalEigenmodeResult,
        PiezoelectricResult,
        PortModeResult,
        SteadyThermalResult,
        TransientThermalResult,
    )

__all__ = ["release_memory", "solve"]


@overload
def solve(
    model: Model,
    problem: Electrostatic,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> ElectrostaticResult: ...
@overload
def solve(
    model: Model,
    problem: DCConduction,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> DCConductionResult: ...
@overload
def solve(
    model: Model,
    problem: SteadyThermal,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> SteadyThermalResult: ...
@overload
def solve(
    model: Model,
    problem: TransientThermal,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> TransientThermalResult: ...
@overload
def solve(
    model: Model,
    problem: Driven,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> DrivenResult: ...
@overload
def solve(
    model: Model,
    problem: Eigenmode,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> EigenmodeResult: ...
@overload
def solve(
    model: Model,
    problem: PortMode,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> PortModeResult: ...
@overload
def solve(
    model: Model,
    problem: Piezoelectric,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> PiezoelectricResult: ...
@overload
def solve(
    model: Model,
    problem: MechanicalEigenmode,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> MechanicalEigenmodeResult: ...
def solve(
    model: Model,
    problem: Any,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
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
    `gfem.NonDifferentiableError` when `jax.grad`, `jax.jit` or `jax.vmap`
    traces the function, if the problem reads it but gfem cannot
    differentiate it (for example a problem value such as
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
        cache: A directory for the solve cache. The result is stored there
            under its solve key, a hash of everything it depends on (the
            mesh, materials, problem, the solver options except ``threads``,
            ``frequency_workers``, ``retain_memory`` and ``device``, gfem's
            version and code, and the versions of NGSolve and, when the
            options select them, PETSc or SLEPc), and the same call again reads it
            back instead of solving: the same result class and values, its
            fields on ``model.mesh``, and the solve's warnings shown again at
            your line. A path that does not exist, or an empty directory,
            becomes a cache; delete the directory to clear it. A result read
            back may have been solved on another machine, device, thread
            count or rank count: the same answer to the precision gfem
            documents, not bit-for-bit. A ``device="cuda"`` call that reads
            its result needs no GPU; one that solves does. A solve under
            `jax.grad`, `jax.jit` or `jax.vmap`, one that raises, and one
            with an input the cache cannot record (a subclass of a gfem
            class, with a `gfem.SolveCacheWarning`) are solved and store
            nothing, as is one that keeps fields on more than one rank
            (with the same warning). The logger ``"gfem.solve_cache"``
            reports each read and store at INFO. ``None`` (the default)
            solves without a cache. The save and export recipe, "Solve once,
            read it back", shows it at work.

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
            the remedy. Or ``cache`` names a non-empty directory that is not
            a solve cache.
        NonDifferentiableError: Under `jax.grad`, an input or output gfem
            does not differentiate is perturbed or used.
        LinearSolverConvergenceError: An iterative linear solve
            (``linear_solver="bddc_gmres"`` or ``"pmg_gmres"``, or ``"petsc"``
            with a Krylov
            ``ksp_type``) did not converge; the message gives the remedy. A
            uniform `gfem.Driven` sweep stops there, and the error's
            ``result`` holds the frequencies it solved, ``unsolved`` the
            rest.
        NotImplementedError: On a 2D mesh, an input with no 2D path (the
            problem type, ``wave_vector``, the adaptive sweep, ``comm``, an
            iterative solver, a GPU, moved nodes); the message names the
            remedy.
    """
    use = _solve_cache(cache)
    return _solve(model, problem, options=options, comm=comm, cache=use)[0]


def _solve_cache(cache: str | os.PathLike | None) -> SolveCache | None:
    """The `SolveCache` of a ``cache=`` argument; None without one.

    Raises:
        TypeError: ``cache`` is not a path.
    """
    if cache is None:
        return None
    # imported here: the cache's module imports gfem.job, which imports this
    from gfem._solve_cache import SolveCache

    return SolveCache(cache)


def _solve(
    model: Model,
    problem: Any,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    adjoint_loads: AdjointLoads | Sequence[DualRequest] | None = None,
    cache: SolveCache | None = None,
) -> tuple[Any, dict[Any, Any] | None]:
    """`solve`, and the adjoints of the per-frequency ``adjoint_loads`` with it.

    ``gfem.refine_mesh`` asks for an estimate's solved duals:
    extra right-hand sides on the forward's
    factorization, formed per frequency from that frequency's solution
    (`gfem._engine.driven.AdjointLoads`); for an eigenmode problem, the
    participation duals of its `gfem._eigen.rule.DualRequest`s, on the
    forward's live factors. Returns
    the result, and those adjoints per (frequency, key), or ``None``
    without ``adjoint_loads``. Warnings point at the caller of `solve`,
    one frame above this one.

    With ``cache``, the call is read from or stored in the solve cache: the
    lookup runs after every check, so a warm cache raises the same errors.
    ``cache`` is left holding what happened (`SolveCache.report`), for
    `gfem.Job.run`'s run record.
    """
    if not isinstance(model, Model):
        raise TypeError(f"solve needs a gfem.Model, got {type(model).__name__}.")
    if options is None:
        options = SolverOptions()
    elif not isinstance(options, SolverOptions):
        raise TypeError(
            f"options must be a gfem.SolverOptions, got {type(options).__name__}."
        )
    refuse_solve(model, problem, options, comm)
    resolved = _mpi.resolve_comm(comm)
    session = (
        contextlib.nullcontext()
        if cache is None
        else cache.session(
            model, problem, comm=resolved, adjoint=adjoint_loads is not None
        )
    )
    with session:
        if resolved is not None:
            _mpi.refuse(model, problem, options)
        rule_class = _rule_class(problem, model.mesh.dim)
        check_problem(problem, model)
        _check_domain(model, problem)
        _check_blends(model, problem)
        if resolved is None:
            rule = rule_class(model, problem, options)
        else:
            # differing caches would split the ranks at the lookup's broadcast
            directory = "" if cache is None else os.fspath(cache.directory)
            _mpi.entry_check(
                resolved,
                "solve",
                model,
                problem,
                options,
                arrays={"the solve cache": directory},
            )
            try:
                rule = rule_class(model, problem, options, comm=resolved)
            except (TypeError, ValueError) as err:
                # the rule reads inputs the entry check has just found equal
                _mpi.agreed(err)
                raise
        if is_traced((model, problem)):
            # a rule may differentiate a problem through its own leaves (a thermal
            # problem's summed heat density)
            inputs = getattr(rule, "vjp_problem", None)
            leaves = problem if inputs is None else inputs(problem)
            traced = SolveRequest(adjoint_loads=adjoint_loads)
            return rule.result(
                _end_of_solve(
                    options, resolved, solve_vjp(rule, model, leaves, request=traced)
                )
            ), None
        # A moved mesh must keep its periodic pairs, keep its physical
        # elements off the layer's stretch and stay valid on the solve domain.
        check_nodes(model.mesh.nodes, model._periodic)  # noqa: SLF001
        check_layer_nodes(model, problem)
        physics = physics_of(problem)
        _check_quality(
            model.mesh,
            model._domain_of(physics).active,  # noqa: SLF001
            stacklevel=4,
            comm=resolved,
        )
        # a uniform sweep's partial result, built here: never in a JAX callback
        partial_result = getattr(rule, "partial_result", None)
        request = SolveRequest(
            adjoint_loads=adjoint_loads, keeps_partial=partial_result is not None
        )
        if cache is not None:
            stored = cache.lookup(model, problem, options)
            if stored is not None:
                return stored, None
        try:
            with (
                _mpi.section(resolved, agreed=_mpi.FORWARD_ERRORS),
                _timing.scope(_timing.problem_name(problem), resolved),
            ):
                outputs, _ = rule.forward(model, keep=False, request=request)
        except BaseException as err:
            _end_of_failed_solve(err, options, resolved, partial_result)
            raise
        adjoints = request.adjoints if adjoint_loads is not None else None
        result = rule.result(_end_of_solve(options, resolved, outputs))
        if cache is not None:
            cache.store(model, result, options)
        return result, adjoints


def _end_of_failed_solve(
    error: BaseException, options: SolverOptions, comm: Any, partial_result: Any
) -> None:
    """Before ``error`` leaves a forward: its partial result, and `_end_of_solve`.

    Emptying the pool is collective, so it runs on an agreed error only: every rank
    raises that one together.
    """
    if isinstance(error, LinearSolverConvergenceError) and partial_result is not None:
        error.result = partial_result(error)
    if comm is None or _mpi.is_agreed(error, _mpi.FORWARD_ERRORS):
        _end_of_solve(options, comm, None)


def _end_of_solve[T](options: SolverOptions, comm: Any, outputs: T) -> T:
    """``outputs``, the solve's pooled coarse factorizations freed if it keeps none.

    That is ``retain_memory=0``.

    Inside the call they served its later set-ups (a sweep's rebuilds, an eigen
    reshift); none outlives it, nor a solve that raised an agreed error. Every rank
    of ``comm`` gets here together.
    """
    coarse = sys.modules.get("gfem._linalg.coarse")
    if options.retain_memory == 0 and coarse is not None:
        coarse.empty(comm)
    return outputs


def _check_domain(model: Model, problem: Any) -> None:
    """Raise when the problem's physics has nothing left to solve on.

    A model needs some volume in some physics' domain; an empty
    domain for one physics raises when a problem of that physics is solved.
    A problem names its physics as ``_physics`` (`physics_of`).

    Raises:
        ValueError: Every volume is excluded from the problem's physics.
    """
    physics = physics_of(problem)
    if model._domain_of(physics).active.any():  # noqa: SLF001
        return
    name = _PHYSICS_NAMES[physics]
    raise ValueError(
        f"Every volume is excluded from {name} problems, so the "
        f"{type(problem).__name__} problem has nothing to solve on. Exclude "
        f"fewer volumes for {name} problems (excluded=)."
    )


def _check_blends(model: Model, problem: Any) -> None:
    """Raise for a density blend with a law end off the driven uniform sweep.

    Such a blend is solved by `gfem.Driven` on the uniform sweep
    only (`gfem._engine.dispersion.refuse_blends`).

    Raises:
        ValueError: An electrostatic problem and a blend with a `Drude` end.
        NotImplementedError: Any other problem than a uniform driven sweep.
    """
    if not any(
        isinstance(m, MaterialField) and m.blend_ends is not None
        for m in model.materials.values()
    ):
        return
    kind = type(problem).__name__
    if kind == "Driven" and problem.sweep == "uniform":
        return
    # imported here: importing gfem does not import the engine
    from gfem._engine.dispersion import refuse_blends

    where = (
        'the adaptive sweep (solve sweep="uniform")'
        if kind == "Driven"
        else f"the {kind} problem"
    )
    refuse_blends(model, where, static=kind == "Electrostatic")


def _rule_class(problem: Any, dim: int = 3) -> type:
    """The engine rule of a problem type on a mesh of ``dim``, imported on first use.

    Each problem class names its rule as ``_rule = "module:Class"``, and its
    rule on a 2D mesh as ``_rule_2d`` where that differs, so a new problem
    type needs no change here, and importing gfem does not import the engine.
    """
    path = getattr(type(problem), "_rule", None)
    if dim == 2:
        path = getattr(type(problem), "_rule_2d", path)
    if not isinstance(path, str) and hasattr(type(problem), "_rule_2d"):
        raise NotImplementedError(
            f"A {type(problem).__name__} problem solves a 2D mesh "
            "(gfem.Mesh.from_gmsh on a triangle-only file); a tetrahedral mesh "
            "or a slab cross-section has no "
            f"{type(problem).__name__.lower()} solve in this version."
        )
    if not isinstance(path, str):
        raise TypeError(
            "solve needs a problem type such as gfem.Electrostatic(...), got "
            f"{type(problem).__name__}."
        )
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def release_memory() -> None:
    """Free everything gfem keeps between solves in this process.

    That is what ``SolverOptions.retain_memory`` bounds, and the meshes and
    spaces it does not: what differentiated solves keep for their gradients,
    the solver plans, factorizations, assembled parts and wave port couplings
    kept for later designs, and the last discretizations of each mesh.

    Results, fields, meshes and models you hold are not touched, and no later
    result changes. A gradient whose solve was released is recomputed from
    the design when it runs, and the next solve builds its mesh, spaces and
    solver plan again: this costs time, never accuracy. Call it between two
    studies in one process, for example before a check at a higher order
    after an optimization loop.

    ``retain_memory=0`` keeps nothing from then on; this call drops what is
    kept now. It does not touch a solve cache on disk (``cache=``). JAX's
    own compilation caches are not touched: see ``jax.clear_caches()``.
    Under ``mpiexec`` or ``srun`` every rank must call it at the same point.

    Example:
        After an optimization loop, before another study::

            gfem.release_memory()
    """
    # nothing is kept before the engine's cache was first loaded
    cache = sys.modules.get("gfem._engine.cache")
    if cache is not None:
        cache.release_memory()
    # after the entries, whose handles give their coarse factorizations back
    coarse = sys.modules.get("gfem._linalg.coarse")
    if coarse is not None:
        coarse.release()
