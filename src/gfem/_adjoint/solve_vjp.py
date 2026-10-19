"""`solve_vjp`: the one JAX ``custom_vjp`` around every host solve.

A rule (``gfem._engine.<problem>.<Problem>Rule``) runs the solve on the
host; this module owns everything JAX sees:

- **Inputs** are the model's and the problem's array leaves. Each leaf gets
  a kind from its pytree path: ``("nodes",)`` for the mesh node coordinates,
  ``("material", region, property)`` for a `MaterialField` value,
  ``("boundary", index, ...)``, ``("port", index, ...)`` and
  ``("problem", path)`` for the rest.
- **Static rules, at trace time** (``symbolic_zeros=True`` tells the forward
  which inputs are perturbed and the backward which outputs carry a
  cotangent): a perturbed input that the rule does not read gets an exact
  zero; a perturbed input it reads but the sensitivity kernel cannot
  differentiate, or a cotangent on an output the rule cannot differentiate,
  raises `gfem.NonDifferentiableError`. ``jax.lax.stop_gradient`` is the
  opt-out: it makes the input unperturbed or the cotangent a symbolic zero.
- **Residuals** are an integer cache key and the design state (the input
  leaves). The forward stores its heavy objects (solver handle, fields) in
  ``gfem._engine.cache.ENTRIES``; the backward rebuilds the model from the
  residual leaves, never from engine state, and re-runs the forward when
  the entry was evicted, so eviction costs time, never correctness.
- **Backward**: the rule folds the cotangents into field pairs
  (``rule.backward``), and `gfem._engine.sensitivity.sensitivity` contracts
  them per element. The cotangent convention is JAX's: for a real input,
  ``ct_p = Re(sum ct * dy/dp)``.

Host calls go through ``jax.pure_callback(..., vmap_method="sequential")``,
so ``vmap`` runs one solve per batch member. Errors raised on the host
during the backward reach the user as ``jax.errors.JaxRuntimeError`` with
the message kept.

A rule provides: ``form`` (its operator family in the property-to-form-term
table), ``differentiable_outputs``, ``output_specs(model)`` (name ->
``jax.ShapeDtypeStruct``), ``role(kind)`` (``"read"`` or ``"unread"``),
``forward(model, *, keep)``, ``backward(model, entry, cotangents)`` and
``options``.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from gfem.errors import NonDifferentiableError

__all__ = ["input_kind", "solve_vjp"]

_KEY = jax.ShapeDtypeStruct((), jnp.int64)


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
        return ("material", names[1], names[2])
    if names[0] == "boundaries":
        return ("boundary", *names[1:])
    if names[0] == "ports":
        return ("port", *names[1:])
    return ("model", *names)


def _describe(kind: tuple) -> str:
    if kind[0] == "nodes":
        return "the mesh node coordinates (model.mesh.nodes)"
    if kind[0] == "material":
        return f"model.materials[{kind[1]!r}].{kind[2]}"
    if kind[0] == "problem":
        return f"the problem leaf {kind[1]}"
    return "model." + ".".join(str(k) for k in kind)


def solve_vjp(rule: Any, model: Any, problem: Any) -> dict[str, jax.Array]:
    """Run ``rule``'s solve as a differentiable JAX function of the leaves.

    Args:
        rule: The problem's rule, constructed (static checks done).
        model: The model; its leaves may be JAX tracers.
        problem: The problem; its leaves may be JAX tracers.

    Returns:
        The rule's outputs by name, as JAX arrays.
    """
    m_paths, m_def = jax.tree_util.tree_flatten_with_path(model)
    p_paths, p_def = jax.tree_util.tree_flatten_with_path(problem)
    n_model = len(m_paths)
    kinds = [input_kind(path, "model") for path, _ in m_paths] + [
        input_kind(path, "problem") for path, _ in p_paths
    ]
    specs = rule.output_specs(model)
    names = tuple(specs)
    out_specs = tuple(specs[n] for n in names)
    state: dict[str, Any] = {}

    def rebuild(values: Sequence[Any]) -> tuple[Any, Any]:
        arrays = [np.asarray(v) for v in values]
        return (
            jax.tree.unflatten(m_def, arrays[:n_model]),
            jax.tree.unflatten(p_def, arrays[n_model:]),
        )

    def host_forward(keep: bool, *values: Any) -> tuple:
        from gfem._engine.cache import ENTRIES, default_retain_memory

        host_model, _ = rebuild(values)
        outputs, entry = _forward(rule, host_model, keep=keep)
        key = -1
        if entry is not None:
            budget = rule.options.retain_memory
            if budget is None:
                budget = default_retain_memory()
            key = ENTRIES.put(entry, budget)
        arrays = tuple(
            np.asarray(outputs[n], dtype=s.dtype).reshape(s.shape)
            for n, s in zip(names, out_specs, strict=True)
        )
        return (*arrays, np.int64(key))

    def call_forward(keep: bool, values: Sequence[Any]) -> tuple:
        *outputs, key = jax.pure_callback(
            functools.partial(host_forward, keep),
            (*out_specs, _KEY),
            *values,
            vmap_method="sequential",
        )
        return tuple(outputs), key

    @jax.custom_vjp
    def primal(m_leaves: list, p_leaves: list) -> tuple:
        outputs, _ = call_forward(False, [*m_leaves, *p_leaves])
        return outputs

    def fwd(m_leaves: list, p_leaves: list) -> tuple:
        inputs = [*m_leaves, *p_leaves]
        state["wanted"] = _wanted(rule, kinds, [x.perturbed for x in inputs])
        values = [x.value for x in inputs]
        # nothing to keep when every perturbed input is unread
        outputs, key = call_forward(bool(state["wanted"]), values)
        return outputs, (key, values)

    def bwd(residuals: tuple, cotangents: tuple) -> tuple:
        key, values = residuals
        live = []
        for name, ct in zip(names, cotangents, strict=True):
            if isinstance(ct, SymbolicZero):
                continue
            if name not in rule.differentiable_outputs:
                raise NonDifferentiableError(rule.nondifferentiable_output(name))
            live.append((name, ct))
        wanted = state["wanted"]
        grads: list[Any] = [jnp.zeros_like(v) for v in values]
        if live and wanted:
            computed = jax.pure_callback(
                functools.partial(
                    _host_backward,
                    rule=rule,
                    rebuild=rebuild,
                    kinds=kinds,
                    wanted=wanted,
                    live_names=tuple(n for n, _ in live),
                ),
                tuple(
                    jax.ShapeDtypeStruct(values[i].shape, values[i].dtype)
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


def _forward(rule: Any, model: Any, *, keep: bool) -> tuple:
    from gfem.shape import _check_quality

    # A moved mesh must still be valid on the solve domain.
    _check_quality(model.mesh, model._domain.active, stacklevel=2)  # noqa: SLF001
    return rule.forward(model, keep=keep)


def _wanted(rule: Any, kinds: list[tuple], perturbed: list[bool]) -> tuple[int, ...]:
    """Indices of the inputs the backward computes; raise for unsupported ones."""
    from gfem._engine.sensitivity import unsupported

    wanted = []
    for i, (kind, moving) in enumerate(zip(kinds, perturbed, strict=True)):
        if not moving or rule.role(kind) == "unread":
            continue
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
    rule: Any,
    rebuild: Any,
    kinds: list[tuple],
    wanted: tuple[int, ...],
    live_names: tuple[str, ...],
) -> tuple:
    from gfem._engine.cache import ENTRIES
    from gfem._engine.sensitivity import sensitivity

    n_inputs = len(kinds)
    values, cts = args[:n_inputs], args[n_inputs:]
    model, _ = rebuild(values)
    entry = ENTRIES.get(int(key)) if int(key) >= 0 else None
    fresh = None
    if entry is None:  # evicted: recompute from the design state
        _, entry = rule.forward(model, keep=True)
        fresh = entry
    try:
        cotangents = {n: np.asarray(c) for n, c in zip(live_names, cts, strict=True)}
        pairs = rule.backward(model, entry, cotangents)
        grads = sensitivity(model, pairs, {kinds[i] for i in wanted}, form=rule.form)
    finally:
        if fresh is not None:
            fresh.free()
    out = []
    for i in wanted:
        g = grads[kinds[i]]
        leaf = np.asarray(values[i])
        g = g.reshape(leaf.shape) if leaf.ndim else g.sum()
        if not np.iscomplexobj(leaf):
            g = np.real(g)
        out.append(np.asarray(g, dtype=leaf.dtype))
    return tuple(out)
