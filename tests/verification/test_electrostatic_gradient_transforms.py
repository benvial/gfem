"""Capacitance gradients under the JAX transformations users apply.

Every way of asking for the same derivative must give the same numbers as
a fresh `jax.grad`: a batch under `jax.vmap` (sequential solves), the rows
of `jax.jacrev`, a pullback called more than once, and a solve whose kept
entry was never stored (``retain_memory=0``) or was evicted before the
backward pass. The backward pass rebuilds the design from its residuals,
so a batch member never sees another member's permittivity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants
import structured_meshes as sm

import gfem
from gfem._engine.cache import ENTRIES

EPS0 = scipy.constants.epsilon_0
UM = 1e-6
PROBLEM = gfem.Electrostatic(
    terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
)
# solve-vs-solve differs by about 1 ulp (threaded assembly)
RTOL = 1e-11


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    layers = (("oxide", 0.5), ("nitride", 0.25))
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    sm.write_msh22(path, sm.plate(40.0, 30.0, layers, (2, 2, 2)))
    mesh = gfem.Mesh.from_gmsh(path)
    return gfem.Model(
        mesh, materials={"oxide": gfem.Material(), "nitride": gfem.Material()}
    )


def _capacitance(stack, eps, options=None):
    """C(eps) / (eps0 um) for one P0 permittivity over both layers."""
    n_ox = len(stack.mesh.elements("oxide"))
    model = stack.replace(
        materials={
            "oxide": gfem.MaterialField(eps_r=eps[:n_ox]),
            "nitride": gfem.MaterialField(eps_r=eps[n_ox:]),
        }
    )
    return gfem.solve(model, PROBLEM, options=options).capacitance / (EPS0 * UM)


WEIGHTS = jnp.asarray([[1.0, 0.3], [-0.7, 2.0]])


def _objective(stack, options=None):
    return lambda eps: jnp.sum(WEIGHTS * _capacitance(stack, eps, options))


def _designs(stack, k, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(2.0, 9.0, (k, len(stack.mesh.tetrahedra))))


def test_vmap_of_grad_equals_a_python_loop(stack):
    designs = _designs(stack, 3)
    grad = jax.grad(_objective(stack))
    batched = jax.vmap(grad)(designs)
    looped = jnp.stack([grad(d) for d in designs])
    np.testing.assert_allclose(batched, looped, rtol=RTOL)
    # the members differ, so a backward reading another member's state shows
    assert not np.allclose(looped[0], looped[1], rtol=1e-3)


def test_grad_of_a_batch_objective_equals_the_loop(stack):
    designs = _designs(stack, 3, seed=1)
    objective = _objective(stack)

    def worst(ds):
        return jnp.max(jax.vmap(objective)(ds))

    values = [objective(d) for d in designs]
    expected = (
        jnp.zeros_like(designs)
        .at[int(np.argmax(values))]
        .set(jax.grad(objective)(designs[int(np.argmax(values))]))
    )
    np.testing.assert_allclose(jax.grad(worst)(designs), expected, rtol=RTOL)


def test_jacrev_rows_equal_fresh_gradients(stack):
    eps = _designs(stack, 1)[0]
    jac = jax.jacrev(lambda e: _capacitance(stack, e))(eps)
    assert jac.shape == (2, 2, eps.size)
    for i in range(2):
        for j in range(2):
            fresh = jax.grad(lambda e, i=i, j=j: _capacitance(stack, e)[i, j])(eps)
            np.testing.assert_allclose(jac[i, j], fresh, rtol=RTOL, atol=0)


def test_a_pullback_called_twice_equals_fresh_gradients(stack):
    eps = _designs(stack, 1, seed=2)[0]
    c, pullback = jax.vjp(lambda e: _capacitance(stack, e), eps)
    first = pullback(WEIGHTS)[0]
    other = jnp.asarray([[0.0, 1.0], [0.0, 0.0]])
    second = pullback(other)[0]
    again = pullback(WEIGHTS)[0]
    np.testing.assert_allclose(first, jax.grad(_objective(stack))(eps), rtol=RTOL)
    np.testing.assert_allclose(
        second, jax.grad(lambda e: _capacitance(stack, e)[0, 1])(eps), rtol=RTOL
    )
    np.testing.assert_array_equal(again, first)
    assert c.shape == (2, 2)


def test_retain_memory_zero_recomputes_the_same_gradient(stack):
    eps = _designs(stack, 1, seed=3)[0]
    cached = jax.grad(_objective(stack))(eps)
    held = len(ENTRIES)
    options = gfem.SolverOptions(retain_memory=0)
    _, pullback = jax.vjp(lambda e: _capacitance(stack, e, options), eps)
    assert len(ENTRIES) == held  # nothing kept between forward and backward
    np.testing.assert_allclose(pullback(WEIGHTS)[0], cached, rtol=RTOL)
    np.testing.assert_allclose(pullback(WEIGHTS)[0], cached, rtol=RTOL)
    assert len(ENTRIES) == held
    batched = jax.vmap(jax.grad(_objective(stack, options)))(_designs(stack, 2))
    looped = [jax.grad(_objective(stack))(d) for d in _designs(stack, 2)]
    np.testing.assert_allclose(batched, jnp.stack(looped), rtol=RTOL)


def test_an_evicted_entry_is_recomputed_from_the_residual_design(stack):
    designs = _designs(stack, 2, seed=4)
    pullbacks = [jax.vjp(lambda e: _capacitance(stack, e), d)[1] for d in designs]
    ENTRIES.clear()
    for d, pullback in zip(designs, pullbacks, strict=True):
        np.testing.assert_allclose(
            pullback(WEIGHTS)[0], jax.grad(_objective(stack))(d), rtol=RTOL
        )


def test_jit_of_vmap_of_grad_equals_the_loop(stack):
    designs = _designs(stack, 2, seed=5)
    grad = jax.grad(_objective(stack))
    looped = jnp.stack([grad(d) for d in designs])
    compiled = jax.jit(jax.vmap(grad))
    np.testing.assert_allclose(compiled(designs), looped, rtol=RTOL)
    np.testing.assert_allclose(compiled(designs), looped, rtol=RTOL)
