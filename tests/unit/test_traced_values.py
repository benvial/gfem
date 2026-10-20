"""Fixed physics values raise `gfem.NonDifferentiableError` when traced.

Frequencies, kept or frozen sample frequencies, eigen targets, port
directions and voltage paths are fixed numbers of a problem or a port:
under `jax.jit` or `jax.grad` they raise `gfem.NonDifferentiableError`
(which an existing ``except ValueError`` still catches) naming the value.
"""

import jax
import jax.numpy as jnp
import pytest

import gfem

CASES = {
    "PortMode.frequency": lambda x: gfem.PortMode("p1", x),
    "Driven.frequency": lambda x: gfem.Driven(frequency=x),
    "Driven.keep_fields": lambda x: gfem.Driven(frequency=1e9, keep_fields=x),
    "Driven.samples": lambda x: gfem.Driven(
        frequency=(1e9, 2e9), sweep="adaptive", samples=x
    ),
    "Eigenmode.target": lambda x: gfem.Eigenmode(target=x),
    "direction": lambda x: gfem.LumpedPort("p1", "port1", x * jnp.ones(3)),
    "voltage_path": lambda x: gfem.WavePort(
        "p1", "port1", voltage_path=x * jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("transform", ["jit", "grad"])
def test_a_traced_fixed_value_raises_nondifferentiable(name, transform):
    build = CASES[name]

    def objective(x):
        build(x)
        return x

    wrapped = jax.jit(objective) if transform == "jit" else jax.grad(objective)
    with pytest.raises(gfem.NonDifferentiableError, match=name.split(".")[-1]):
        wrapped(1e9)


def test_the_error_is_still_a_value_error():
    with pytest.raises(ValueError, match="frequency"):
        jax.jit(lambda f: gfem.Driven(frequency=f))(1e9)
