"""`solve_vjp`: the one JAX ``custom_vjp`` around every host solve.

A rule (``gfem._engine.<problem>.<Problem>Rule``) runs the solve on the
host; this module owns everything JAX sees:

- **Inputs** are the model's and the problem's array leaves. Each leaf gets
  a kind from its pytree path: ``("nodes",)`` for the mesh node coordinates,
  ``("material", region, property)`` for a `MaterialField` value (per
  element, or per element and principal axis for a field with `axes`),
  ``("boundary", index, ...)``, ``("port", index, ...)`` and
  ``("problem", path)`` for the rest.
- **Static rules, at trace time** (``symbolic_zeros=True`` tells the forward
  which inputs are perturbed and the backward which outputs carry a
  cotangent): a perturbed input that the rule does not read gets an exact
  zero; a perturbed input it reads but the sensitivity kernel cannot
  differentiate, or a cotangent on an output the rule cannot differentiate
  while some perturbed input is read, raises `gfem.NonDifferentiableError`
  (when every perturbed input is unread, every output's zero is exact).
  ``jax.lax.stop_gradient`` is the opt-out: it makes the input unperturbed
  or the cotangent a symbolic zero.
- **Residuals** are an integer cache key and the design state (the input
  leaves). The forward stores its heavy objects (solver handle, fields) in
  ``gfem._engine.cache.ENTRIES`` (bounded by
  ``SolverOptions.retain_memory``, least recently used out first, never
  freed by a read, so ``jax.jacrev`` and a pullback called twice reuse it).
  An entry a backward has read is consumed, and the next differentiated
  forward frees it before it assembles, its base handle returning
  to the plan pool for that forward, so a pullback called after
  another differentiated solve runs its forward again.
  The backward rebuilds the model and the problem from the residual leaves,
  never from engine state, and re-runs the forward when the entry was
  evicted or never kept, so eviction costs time, never correctness.
- **Backward**: the rule folds the cotangents into field pairs
  (``rule.backward``), and `gfem._engine.sensitivity.sensitivity` contracts
  them per element. The cotangent convention is JAX's: for a real input,
  ``ct_p = Re(sum ct * dy/dp)``.

Host calls go through ``jax.pure_callback(..., vmap_method="sequential")``,
so ``vmap`` runs one solve per batch member. Errors raised on the host
during the backward reach the user as ``jax.errors.JaxRuntimeError`` with
the message kept, or as ``ValueError`` once a jitted function has succeeded.

Under ranks every rank runs this whole function and its callbacks on
replicated values; only host calls communicate. The integer key is a rank's own
number, and the backward agrees on whether every rank still holds its entry. The
kernel writes each rank's elements into global zeros, and each wanted leaf is summed
over ranks once, where the backward returns. A missing sum raises nothing.

A rule provides: ``form`` (its operator family, a key of
`gfem._engine.sensitivity.PROPERTIES`), ``differentiable_outputs`` (or, with
``input_aware``, a ``nondifferentiable_output(name, kinds)`` that also sees
the wanted input kinds and returns ``None`` where it can differentiate),
``output_specs(model)`` (name -> ``jax.ShapeDtypeStruct``), ``role(kind)``
(``"read"`` or ``"unread"``), ``forward(model, *, keep, request)`` (a
`gfem._engine.solve_request.SolveRequest` carrying the wanted input kinds,
so a forward keeps only what the backward reads, and whether it reruns for
a backward), ``backward(model, entry, cotangents, *, wanted)`` (the wanted
input kinds again), ``options`` and ``problem``,
optionally ``pieces`` (the outputs that are a rank's own piece under ranks,
whose cotangents the ranks compare by shape; none by default),
``explicit_problem()`` (problem-leaf cotangents the backward computed itself),
``explicit_nodes()`` (a node cotangent the backward computed itself, this
rank's part, or ``None``),
``unsupported_input(kind)`` (a reason, or ``None``) for read
inputs the rule itself cannot differentiate, and ``terms(model, *, nodes)``
(its operator as the kernel sees it, a list of
`gfem._engine.sensitivity.Term`: one per piece of a driven or eigenmode
operator, or the electrostatic form; ``nodes`` when the node coordinates
are wanted), ``lumped_terms(model)`` (the pieces that move with a
lumped value) and ``tensors(model, wanted)`` (a rule whose material leaves
are tensors in a rotated frame: `gfem._engine.sensitivity.Tensors`). Host
calls get the model and a shallow copy of the rule whose ``problem`` holds
the call's concrete values, so a rule reads problem values from
``self.problem`` and never sees a tracer there.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import sys
from collections.abc import Collection, Generator, Hashable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from gfem import _mpi, _timing
from gfem._engine.solve_request import SolveRequest
from gfem.errors import NonDifferentiableError

__all__ = ["adjoint_tolerance", "input_kind", "solve_vjp"]

_KEY = jax.ShapeDtypeStruct((), jnp.int64)


def adjoint_tolerance(options: Any) -> float:
    """Relative residual of adjoint solves of a `gfem.SolverOptions`.

    ``options.adjoint_tolerance``, or the forward ``options.tolerance`` when
    that is ``None``. Resolved here, not stored in the options, so that
    replacing the forward tolerance of options built with the default moves
    the adjoint one too.
    """
    if options.adjoint_tolerance is None:
        return options.tolerance
    return options.adjoint_tolerance


def _name(key: Any) -> Any:
    for attr in ("name", "key", "idx"):
        if hasattr(key, attr):
            return getattr(key, attr)
    return str(key)


def input_kind(path: Sequence[Any], owner: str) -> tuple:
    """The kind of a model (``owner="model"``) or problem leaf from its path."""
    names = tuple(_name(k) for k in path)
    if owner == "problem":
        return ("problem", jax.tree_util.keystr(tuple(path)))
    if names[0] == "mesh":
        return ("nodes",)
    if names[0] == "materials":
        # a permittivity law's leaf (eps_r.eps_inf, or a term's parameter
        # eps_r.terms[i].<name>) is a "law" kind: region, then "eps_inf", or
        # the term's index and the parameter's name
        law = names[4:] if names[3:4] == ("terms",) else names[3:]
        return ("law", names[1], *law) if law else ("material", names[1], names[2])
    if names[0] == "boundaries":
        return ("boundary", *names[1:])
    if names[0] == "ports":
        return ("port", *names[1:])
    return ("model", *names)


def _describe(kind: tuple) -> str:
    if kind[0] == "nodes":
        return "the mesh node coordinates (model.mesh.nodes)"
    if kind[0] in ("material", "law"):
        rest = "".join(f"[{k}]" if isinstance(k, int) else f".{k}" for k in kind[2:])
        law = ".eps_r" + (".terms" if len(kind) > 3 else "") if kind[0] == "law" else ""
        return f"model.materials[{kind[1]!r}]{law}{rest}"
    if kind[0] == "problem":
        return f"problem{kind[1]}"
    if kind[0] == "boundary":
        rest = "".join(f".{k}" for k in kind[2:])
        return f"model.boundaries[{kind[1]}]{rest}"
    if kind[0] == "port":
        rest = "".join(f".{k}" for k in kind[2:])
        return f"model.ports[{kind[1]}]{rest}"
    return "model." + ".".join(str(k) for k in kind)


def solve_vjp(  # noqa: PLR0915
    rule: Any, model: Any, problem: Any, *, request: SolveRequest | None = None
) -> dict[str, jax.Array]:
    """Run ``rule``'s solve as a differentiable JAX function of the leaves.

    Args:
        rule: The problem's rule, constructed (static checks done).
        model: The model; its leaves may be JAX tracers.
        problem: The problem; its leaves may be JAX tracers.
        request: What every host forward is asked beyond the wanted input
            kinds and the rerun flag, which this function sets.

    Returns:
        The rule's outputs by name, as JAX arrays.
    """
    m_paths, m_def = jax.tree_util.tree_flatten_with_path(model)
    p_paths, p_def = jax.tree_util.tree_flatten_with_path(problem)
    n_model = len(m_paths)
    kinds = [input_kind(path, "model") for path, _ in m_paths] + [
        input_kind(path, "problem") for path, _ in p_paths
    ]
    request = SolveRequest() if request is None else request
    specs = rule.output_specs(model)
    names = tuple(specs)
    out_specs = tuple(specs[n] for n in names)
    state: dict[str, Any] = {}
    rebuild = functools.partial(
        _rebuild, rule=rule, m_def=m_def, p_def=p_def, n_model=n_model
    )

    def host_forward(keep: bool, wanted: frozenset, *values: Any) -> tuple:
        comm = getattr(rule, "comm", None)
        host_rule, host_model = rebuild(values)
        with (
            _mpi.section(comm, agreed=_mpi.FORWARD_ERRORS),
            _coarse_pool_bounded(rule.options, comm),
        ):
            if comm is not None:
                _mpi.entry_check(
                    comm,
                    f"the forward of a {type(rule).__name__}",
                    host_model,
                    host_rule.problem,
                    rule.options,
                )
            if keep:
                _free_consumed(comm, rule.options)
            # what the backward will read: a rule may keep less
            outputs, entry = _forward(
                host_rule,
                host_model,
                keep=keep,
                request=dataclasses.replace(request, wanted=wanted),
            )
            key = -1 if entry is None else _keep(entry, rule.options, comm)
        arrays = tuple(
            np.asarray(outputs[n], dtype=s.dtype).reshape(s.shape)
            for n, s in zip(names, out_specs, strict=True)
        )
        return (*arrays, np.int64(key))

    def call_forward(keep: bool, values: Sequence[Any], wanted: frozenset) -> tuple:
        *outputs, key = jax.pure_callback(
            functools.partial(host_forward, keep, wanted),
            (*out_specs, _KEY),
            *values,
            vmap_method="sequential",
        )
        return tuple(outputs), key

    @jax.custom_vjp
    def primal(m_leaves: list, p_leaves: list) -> tuple:
        outputs, _ = call_forward(False, [*m_leaves, *p_leaves], frozenset())
        return outputs

    def fwd(m_leaves: list, p_leaves: list) -> tuple:
        inputs = [*m_leaves, *p_leaves]
        state["wanted"] = _wanted(rule, kinds, [x.perturbed for x in inputs])
        values = [x.value for x in inputs]
        # nothing to keep when every perturbed input is unread
        outputs, key = call_forward(
            bool(state["wanted"]),
            values,
            frozenset(kinds[i] for i in state["wanted"]),
        )
        return outputs, (key, values)

    def bwd(residuals: tuple, cotangents: tuple) -> tuple:
        key, values = residuals
        wanted = state["wanted"]
        wanted_kinds = frozenset(kinds[i] for i in wanted)
        live = []
        for name, ct in zip(names, cotangents, strict=True):
            if isinstance(ct, SymbolicZero):
                continue
            # with every perturbed input unread, every output's zero is exact
            if wanted:
                reason = _refusal(rule, name, wanted_kinds)
                if reason is not None:
                    raise NonDifferentiableError(reason)
            live.append((name, ct))
        grads: list[Any] = [jnp.zeros_like(v) for v in values]
        if live and wanted:
            computed = jax.pure_callback(
                functools.partial(
                    _host_backward,
                    rebuild=rebuild,
                    request=request,
                    kinds=kinds,
                    wanted=wanted,
                    live_names=tuple(n for n, _ in live),
                ),
                # a lumped value given as a Python number has no shape
                tuple(
                    jax.ShapeDtypeStruct(
                        jnp.shape(values[i]), jnp.result_type(values[i])
                    )
                    for i in wanted
                ),
                key,
                *values,
                *(ct for _, ct in live),
                vmap_method="sequential",
            )
            for i, g in zip(wanted, computed, strict=True):
                grads[i] = g
        return grads[:n_model], grads[n_model:]

    primal.defvjp(fwd, bwd, symbolic_zeros=True)
    m_leaves = [leaf for _, leaf in m_paths]
    p_leaves = [leaf for _, leaf in p_paths]
    outputs = primal(m_leaves, p_leaves)
    return dict(zip(names, outputs, strict=True))


def _refusal(rule: Any, name: str, kinds: frozenset[tuple]) -> str | None:
    """Why a cotangent on output ``name`` cannot be pulled back, or ``None``.

    A rule with ``input_aware`` answers from the output and the wanted input
    kinds (``rule.nondifferentiable_output(name, kinds)``, ``None`` when it
    can); any other from its ``differentiable_outputs``.
    """
    if getattr(rule, "input_aware", False):
        return rule.nondifferentiable_output(name, kinds)
    if name in rule.differentiable_outputs:
        return None
    return rule.nondifferentiable_output(name)


def _rebuild(
    values: Sequence[Any], *, rule: Any, m_def: Any, p_def: Any, n_model: int
) -> tuple[Any, Any]:
    """The host rule and model of one design state (residual values).

    The model comes from the values alone; the rule is a shallow copy whose
    ``problem`` holds the problem's values, unless the problem has none.
    """
    arrays = [np.asarray(v) for v in values]
    host_model = jax.tree.unflatten(m_def, arrays[:n_model])
    if len(arrays) == n_model:  # the problem holds no array values
        return rule, host_model
    host_rule = copy.copy(rule)
    host_rule.problem = jax.tree.unflatten(p_def, arrays[n_model:])
    return host_rule, host_model


def _forward(rule: Any, model: Any, *, keep: bool, request: SolveRequest) -> tuple:
    from gfem._curved import check_moved
    from gfem._materials import check_field_values
    from gfem._periodic import check_nodes
    from gfem._pml import check_layer_nodes
    from gfem.shape import _check_quality

    # Traced material values skipped the constructor's check; the host holds
    # their concrete values here.
    check_field_values(model.materials)
    # A moved mesh must keep its periodic pairs, keep its physical elements
    # off the layer's stretch and stay valid on the solve domain.
    check_nodes(model.mesh.nodes, model._periodic)  # noqa: SLF001
    check_layer_nodes(model, rule.problem)
    check_moved(model)
    _check_quality(
        model.mesh,
        model._domain_of(getattr(rule, "physics", "em")).active,  # noqa: SLF001
        stacklevel=2,
        comm=getattr(rule, "comm", None),
    )
    problem = _timing.problem_name(_problem_of(rule))
    with _timing.scope(problem, getattr(rule, "comm", None)):
        return rule.forward(model, keep=keep, request=request)


def _problem_of(rule: Any) -> Any:
    """The problem a rule solves: its static ``spec`` where ``problem`` holds loads."""
    return getattr(rule, "spec", getattr(rule, "problem", None))


@contextlib.contextmanager
def _coarse_pool_bounded(options: Any, comm: Any) -> Generator[None]:
    """With ``retain_memory=0``, no pooled coarse factorization outlives a host call.

    A traced solve's host calls run when JAX runs them, after `gfem.solve` returned
    under ``jax.jit``, so `gfem.solve` cannot empty the pool itself. Emptying is
    collective: on an error it runs only when every rank raises that error
    together; inside `_mpi.section`, any other error ends the job first.
    """
    try:
        yield
    except BaseException as error:
        if comm is None or _mpi.is_agreed(error, _mpi.FORWARD_ERRORS):
            _empty_coarse_pool(options, comm)
        raise
    _empty_coarse_pool(options, comm)


def _empty_coarse_pool(options: Any, comm: Any) -> None:
    coarse = sys.modules.get("gfem._linalg.coarse")
    if options.retain_memory == 0 and coarse is not None:
        coarse.empty(comm)


def _free_consumed(comm: Any, options: Any) -> None:
    """Free the entries a backward has read, before this forward assembles.

    Their base handles return to the plan pool, for this forward to take,
    within ``options.retain_memory``.
    """
    from gfem._engine.cache import ENTRIES, retain_budget

    ENTRIES.free_consumed(comm, retain_budget(options.retain_memory))


def _keep(entry: Any, options: Any, comm: Any) -> int:
    """Store a forward's entry under ``options.retain_memory``; its key."""
    from gfem._engine.cache import ENTRIES, retain_budget

    return ENTRIES.put(entry, retain_budget(options.retain_memory), comm)


def _wanted(rule: Any, kinds: list[tuple], perturbed: list[bool]) -> tuple[int, ...]:
    """Indices of the inputs the backward computes; raise for unsupported ones.

    The rule's own refusal (`unsupported_input`) comes first, then the
    kernel's (`gfem._engine.sensitivity.unsupported`).
    """
    from gfem._engine.sensitivity import unsupported

    wanted = []
    for i, (kind, moving) in enumerate(zip(kinds, perturbed, strict=True)):
        if not moving or rule.role(kind) == "unread":
            continue
        own = getattr(rule, "unsupported_input", None)
        reason = None if own is None else own(kind)
        if reason is None:
            reason = unsupported(rule.form, kind)
        if reason is not None:
            raise NonDifferentiableError(
                f"gfem cannot differentiate {rule.form} results with respect to "
                f"{_describe(kind)}: {reason}. Differentiate with respect to "
                "other inputs, or wrap this one in jax.lax.stop_gradient to use "
                "its value without its derivative."
            )
        wanted.append(i)
    return tuple(wanted)


def _host_backward(
    key: Any,
    *args: Any,
    rebuild: Any,
    request: SolveRequest,
    kinds: list[tuple],
    wanted: tuple[int, ...],
    live_names: tuple[str, ...],
) -> tuple:
    n_inputs = len(kinds)
    values, cts = args[:n_inputs], args[n_inputs:]
    rule, model = rebuild(values)
    comm = getattr(rule, "comm", None)
    cotangents = {n: np.asarray(c) for n, c in zip(live_names, cts, strict=True)}
    with (
        _mpi.section(comm, agreed=_mpi.FORWARD_ERRORS),
        _coarse_pool_bounded(rule.options, comm),
        _timing.scope(_timing.problem_name(_problem_of(rule)), comm),
    ):
        if comm is not None:
            _mpi.entry_check(
                comm,
                f"the backward of a {type(rule).__name__}",
                model,
                rule.problem,
                rule.options,
                arrays=_replicated_cotangents(
                    cotangents, getattr(rule, "pieces", frozenset())
                ),
            )
        grads = _local_cotangents(
            key,
            rule,
            model,
            cotangents,
            wants={kinds[i] for i in wanted},
            comm=comm,
            request=request,
        )
    out = []
    for i in wanted:
        # every step from a pass to here is real-linear on replicated values, so one
        # sum, before the leaf's shape, equals a sum per pass
        g = _mpi.reduce_bcast(comm, np.asarray(grads[kinds[i]]))
        leaf = np.asarray(values[i])
        if not leaf.ndim:
            g = g.sum()
        elif g.ndim > leaf.ndim:  # a uniform leaf of principal values, (3,)
            g = g.sum(axis=0)
        else:
            g = g.reshape(leaf.shape)
        if not np.iscomplexobj(leaf):
            g = np.real(g)
        out.append(np.asarray(g, dtype=leaf.dtype))
    return tuple(out)


def _replicated_cotangents(
    cotangents: dict[str, np.ndarray], pieces: Collection[str]
) -> dict[str, np.ndarray]:
    """What the ranks must hold alike: a cotangent, or a rank's piece's shape.

    Args:
        cotangents: The live cotangents by output name.
        pieces: The rule's outputs that are a rank's own piece (``rule.pieces``).
    """
    return {
        f"the cotangent of {n}": np.asarray(c.shape) if n in pieces else c
        for n, c in cotangents.items()
    }


_LUMPED_VALUES = ("resistance", "inductance", "capacitance")


def _has_lumped_pass(kind: Hashable) -> bool:
    """Whether the input ``kind`` is a lumped port or sheet value (R, L or C)."""
    return (
        isinstance(kind, tuple)
        and kind[0] in ("port", "boundary")
        and kind[-1] in _LUMPED_VALUES
    )


def _add_explicit_inputs(
    out: dict, explicit: dict[tuple, Any], *, wants: set[Hashable]
) -> None:
    """Add a rule's explicit terms by input kind to the values in ``out``.

    A term of an input that is not wanted is dropped (the rule computes them
    for every input it reads itself). A wanted lumped value (R, L or C of a
    port or sheet) always has its lumped pass in ``out``: one without is a
    programming error. Any other wanted kind (a law, a sheet fill, an
    impedance value read by the rule alone) takes the term as its value when
    no pass produced one. Values add as they are: scalars or arrays.
    """
    for kind, value in explicit.items():
        if kind not in wants:
            continue
        if kind in out:
            out[kind] = out[kind] + value
        elif _has_lumped_pass(kind):
            msg = (
                f"The explicit lumped term of {kind} has no lumped pass value to "
                "add to (an internal error)."
            )
            raise RuntimeError(msg)
        else:
            out[kind] = value


def _local_cotangents(
    key: Any,
    rule: Any,
    model: Any,
    cotangents: dict[str, np.ndarray],
    *,
    wants: set[Hashable],
    comm: Any,
    request: SolveRequest,
) -> dict:
    """This rank's part of each wanted cotangent.

    The forward runs again unless every rank still holds its entry: a backward on
    some ranks alone would hang. A rank that holds one frees it, since the next
    forward on the same design would otherwise keep a second. A rule's
    ``explicit_inputs`` values (by input kind) add to the pass values.
    ``request`` is the forward's, for a rerun.
    """
    from gfem._engine.cache import ENTRIES, retain_budget
    from gfem._engine.runtime import task_manager
    from gfem._engine.sensitivity import sensitivity

    entry = ENTRIES.get(int(key)) if int(key) >= 0 else None
    fresh = None
    wanted = frozenset(wants)
    if not _mpi.agree_all(comm, flag=entry is not None):
        ENTRIES.discard(int(key))
        # a rerun keeps the least it can
        again = dataclasses.replace(request, wanted=wanted, rerun=True)
        _, entry = rule.forward(model, keep=True, request=again)
        fresh = entry
    else:
        if entry is not None:  # freed when the next differentiated forward starts
            entry.consumed = True
        # the base handles of other entries go before this one prepares
        ENTRIES.free_bases(comm, keep=entry)
    try:
        pairs = rule.backward(model, entry, cotangents, wanted=wanted)
        # the element passes assemble on the solve's threads
        with task_manager(rule.options.threads, comm):
            out = sensitivity(
                model,
                pairs,
                wants,
                form=rule.form,
                terms=(lambda nodes: rule.terms(model, nodes=nodes))
                if hasattr(rule, "terms")
                else None,
                lumped=(lambda: rule.lumped_terms(model))
                if hasattr(rule, "lumped_terms")
                else None,
                tensors=(lambda: rule.tensors(model, wanted))
                if hasattr(rule, "tensors")
                else None,
                comm=comm,
            )
        # the rule's explicit terms by input kind (outside any piece: the
        # eigenmode ratios' 1 / L and C |V|^2 / 2 in the lumped values), once
        # over the ranks
        if hasattr(rule, "explicit_inputs") and _mpi.rank(comm) == 0:
            _add_explicit_inputs(out, rule.explicit_inputs(), wants=wants)
        # the node coordinates' terms outside the operator (a point load's
        # element moving), each rank's own part
        explicit = getattr(rule, "explicit_nodes", None)
        if ("nodes",) in wants and explicit is not None:
            extra = explicit()
            if extra is not None:
                out["nodes",] = out["nodes",] + extra
        # a load form's problem leaves read by the rule itself (the
        # waveform scales, a trace's weights, and a heat trace's direct
        # reading of the densities, added to their load pass)
        if hasattr(rule, "explicit_problem"):
            for k, v in rule.explicit_problem().items():
                if k in wants:
                    out[k] = out[k] + v if k in out else v
    finally:
        if fresh is not None:
            fresh.free()
    if fresh is not None:
        # as an entry `put` frees: the kept state its forward rebuilt goes down
        # to the budget too (collective, so only once every rank got here)
        ENTRIES.trim(retain_budget(rule.options.retain_memory), comm)
    return out
