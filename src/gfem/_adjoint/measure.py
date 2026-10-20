"""`measure_vjp`: the JAX ``custom_vjp`` around host-kernel measurements.

A measurement that needs finite-element evaluation of a field (a point
probe, a surface integral) runs on the host, like a solve, but needs no
solve: its backward pass is the transpose of its forward kernel. This
module gives such a kernel the same JAX semantics as `gfem.solve`:

- **Inputs** are named arrays (the field coefficients, the mesh node
  coordinates, the probe points, ...). ``symbolic_zeros=True`` tells the
  forward which inputs are perturbed.
- **At trace time** a perturbed input the kernel does not read gets an exact
  zero; a perturbed input it reads but cannot differentiate raises
  `gfem.NonDifferentiableError` with the kernel's reason.
  ``jax.lax.stop_gradient`` is the opt-out.
- **Residuals** are the input values; the backward re-enters the kernel with
  them (kernels cache what their forward built).

Host calls go through ``jax.pure_callback(..., vmap_method="sequential")``.

A kernel provides ``output_specs`` (name -> ``jax.ShapeDtypeStruct``),
``role(name)`` (``"read"`` / ``"unread"``), ``unsupported(name)`` (``None``
or the reason the input cannot be differentiated), ``forward(values)`` and
``backward(values, cotangents, wanted)`` (gradients of the ``wanted``
inputs, by name), all on NumPy values, and ``describe(name)`` for errors.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from gfem._checks import is_traced
from gfem.errors import NonDifferentiableError

__all__ = ["measure", "measure_vjp"]


def measure(kernel: Any, inputs: Mapping[str, Any]) -> dict[str, jax.Array]:
    """Run ``kernel`` on ``inputs``: directly, or through `measure_vjp`.

    With no JAX tracer among the inputs the kernel runs at once, so host
    errors reach the caller as they are.
    """
    if is_traced(dict(inputs)):
        return measure_vjp(kernel, inputs)
    values = {k: np.asarray(v) for k, v in inputs.items()}
    outputs = kernel.forward(values)
    specs = kernel.output_specs
    return {
        n: jnp.asarray(np.asarray(outputs[n], dtype=s.dtype).reshape(s.shape))
        for n, s in specs.items()
    }


def measure_vjp(kernel: Any, inputs: Mapping[str, Any]) -> dict[str, jax.Array]:
    """``kernel`` as a differentiable JAX function of ``inputs``."""
    names = tuple(inputs)
    specs = kernel.output_specs
    out_names = tuple(specs)
    out_specs = tuple(specs[n] for n in out_names)
    state: dict[str, Any] = {}

    def host_forward(*values: Any) -> tuple:
        outputs = kernel.forward(
            {n: np.asarray(v) for n, v in zip(names, values, strict=True)}
        )
        return tuple(
            np.asarray(outputs[n], dtype=s.dtype).reshape(s.shape)
            for n, s in zip(out_names, out_specs, strict=True)
        )

    def call_forward(values: list) -> tuple:
        return jax.pure_callback(
            host_forward, out_specs, *values, vmap_method="sequential"
        )

    @jax.custom_vjp
    def primal(leaves: list) -> tuple:
        return call_forward(leaves)

    def fwd(leaves: list) -> tuple:
        wanted = []
        for i, (name, x) in enumerate(zip(names, leaves, strict=True)):
            if not x.perturbed or kernel.role(name) == "unread":
                continue
            reason = kernel.unsupported(name)
            if reason is not None:
                raise NonDifferentiableError(
                    f"gfem cannot differentiate {kernel.name} with respect to "
                    f"{kernel.describe(name)}: {reason}. Differentiate with "
                    "respect to other inputs, or wrap this one in "
                    "jax.lax.stop_gradient to use its value without its "
                    "derivative."
                )
            wanted.append(i)
        state["wanted"] = tuple(wanted)
        values = [x.value for x in leaves]
        return call_forward(values), values

    def bwd(values: list, cotangents: tuple) -> tuple:
        wanted = state["wanted"]
        live = [
            (n, ct)
            for n, ct in zip(out_names, cotangents, strict=True)
            if not isinstance(ct, SymbolicZero)
        ]
        grads: list[Any] = [jnp.zeros_like(v) for v in values]
        if live and wanted:
            computed = jax.pure_callback(
                functools.partial(
                    _host_backward,
                    kernel=kernel,
                    names=names,
                    wanted=wanted,
                    live=tuple(n for n, _ in live),
                ),
                tuple(
                    jax.ShapeDtypeStruct(values[i].shape, values[i].dtype)
                    for i in wanted
                ),
                *values,
                *(ct for _, ct in live),
                vmap_method="sequential",
            )
            for i, g in zip(wanted, computed, strict=True):
                grads[i] = g
        return (grads,)

    primal.defvjp(fwd, bwd, symbolic_zeros=True)
    outputs = primal([jnp.asarray(inputs[n]) for n in names])
    return dict(zip(out_names, outputs, strict=True))


def _host_backward(
    *args: Any,
    kernel: Any,
    names: tuple[str, ...],
    wanted: tuple[int, ...],
    live: tuple[str, ...],
) -> tuple:
    values = {n: np.asarray(v) for n, v in zip(names, args[: len(names)], strict=True)}
    cts = {n: np.asarray(c) for n, c in zip(live, args[len(names) :], strict=True)}
    grads = kernel.backward(values, cts, tuple(names[i] for i in wanted))
    out = []
    for i in wanted:
        leaf = values[names[i]]
        g = np.asarray(grads[names[i]]).reshape(leaf.shape)
        if not np.iscomplexobj(leaf):
            g = np.real(g)
        out.append(np.asarray(g, dtype=leaf.dtype))
    return tuple(out)
