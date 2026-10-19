"""S-parameter and field gradients under the JAX transformations users apply.

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
import structured_meshes as sm

import gfem
from gfem._engine.cache import ENTRIES

FREQUENCIES = (15e9, 25e9)
PROBLEM = gfem.Driven(frequency=FREQUENCIES)
OPTIONS = gfem.SolverOptions(linear_solver="direct")
# solve-vs-solve differs by about 1 ulp (threaded assembly)
RTOL = 1e-10


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.Absorbing("side")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _s(line, eps, options=OPTIONS, problem=PROBLEM):
    model = line.with_material("dielectric", gfem.MaterialField(eps_r=eps))
    return gfem.solve(model, problem, options=options).s.values


WEIGHTS = jnp.asarray([[1.0, 0.3], [-0.7, 2.0]])


def _objective(line, options=OPTIONS):
    return lambda eps: jnp.sum(WEIGHTS * jnp.abs(_s(line, eps, options)) ** 2)


def _designs(line, k, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(2.0, 6.0, (k, len(line.mesh.tetrahedra))))


def test_vmap_of_grad_equals_a_python_loop(line):
    designs = _designs(line, 3)
    grad = jax.grad(_objective(line))
    batched = jax.vmap(grad)(designs)
    looped = jnp.stack([grad(d) for d in designs])
    np.testing.assert_allclose(batched, looped, rtol=RTOL)
    # the members differ, so a backward reading another member's state shows
    assert not np.allclose(looped[0], looped[1], rtol=1e-3)


def test_grad_of_a_batch_objective_equals_the_loop(line):
    designs = _designs(line, 3, seed=1)
    objective = _objective(line)

    def worst(ds):
        return jnp.max(jax.vmap(objective)(ds))

    values = [objective(d) for d in designs]
    k = int(np.argmax(values))
    expected = jnp.zeros_like(designs).at[k].set(jax.grad(objective)(designs[k]))
    np.testing.assert_allclose(jax.grad(worst)(designs), expected, rtol=RTOL)


def test_jacrev_rows_equal_fresh_gradients(line):
    eps = _designs(line, 1)[0]

    def s_abs(e):
        return jnp.abs(_s(line, e)) ** 2

    jac = jax.jacrev(s_abs)(eps)
    assert jac.shape == (len(FREQUENCIES), 2, 2, eps.size)
    for n in range(len(FREQUENCIES)):
        for i in range(2):
            for j in range(2):
                fresh = jax.grad(lambda e, n=n, i=i, j=j: s_abs(e)[n, i, j])(eps)
                np.testing.assert_allclose(jac[n, i, j], fresh, rtol=RTOL, atol=0)


def test_a_pullback_called_twice_equals_fresh_gradients(line):
    eps = _designs(line, 1, seed=2)[0]
    s, pullback = jax.vjp(lambda e: _s(line, e), eps)
    # the cotangent of Re(c S21) is c (JAX transposes, never conjugates)
    ct = jnp.zeros(s.shape, complex).at[:, 1, 0].set(0.3 + 0.4j)
    first = pullback(ct)[0]
    second = pullback(2 * ct)[0]
    again = pullback(ct)[0]

    def fresh(e):
        return jnp.sum(jnp.real((0.3 + 0.4j) * _s(line, e)[:, 1, 0]))

    np.testing.assert_allclose(first, jax.grad(fresh)(eps), rtol=RTOL)
    np.testing.assert_allclose(second, 2 * first, rtol=RTOL)
    np.testing.assert_array_equal(again, first)


def test_retain_memory_zero_recomputes_the_same_gradient(line):
    eps = _designs(line, 1, seed=3)[0]
    cached = jax.grad(_objective(line))(eps)
    held = len(ENTRIES)
    options = gfem.SolverOptions(linear_solver="direct", retain_memory=0)
    s, pullback = jax.vjp(lambda e: _s(line, e, options), eps)
    assert len(ENTRIES) == held  # nothing kept between forward and backward
    ct = 2 * WEIGHTS * jnp.conj(s)  # the cotangent of sum W |S|^2
    np.testing.assert_allclose(pullback(ct)[0], cached, rtol=RTOL)
    np.testing.assert_allclose(pullback(ct)[0], cached, rtol=RTOL)
    assert len(ENTRIES) == held


def test_an_evicted_entry_is_recomputed_from_the_residual_design(line):
    designs = _designs(line, 2, seed=4)
    objective = _objective(line)
    pullbacks = [jax.vjp(objective, d)[1] for d in designs]
    ENTRIES.clear()
    for d, pullback in zip(designs, pullbacks, strict=True):
        np.testing.assert_allclose(pullback(1.0)[0], jax.grad(objective)(d), rtol=RTOL)


def test_jit_of_vmap_of_grad_equals_the_loop(line):
    designs = _designs(line, 2, seed=5)
    grad = jax.grad(_objective(line))
    looped = jnp.stack([grad(d) for d in designs])
    compiled = jax.jit(jax.vmap(grad))
    np.testing.assert_allclose(compiled(designs), looped, rtol=RTOL)
    np.testing.assert_allclose(compiled(designs), looped, rtol=RTOL)


# --- kept-field functionals: an adjoint solve in the backward pass ---------------


def _field_objective(line, options=OPTIONS):
    problem = gfem.Driven(frequency=FREQUENCIES, keep_fields=True)

    def objective(eps):
        model = line.with_material("dielectric", gfem.MaterialField(eps_r=eps))
        result = gfem.solve(model, problem, options=options)
        total = 0.0
        for f in FREQUENCIES:
            e = result.e_field(f, "p2").coefficients
            total = total + jnp.abs(jnp.sum(jnp.cos(jnp.arange(e.size)) * e)) ** 2
        return total

    return objective


def test_field_gradients_agree_under_every_transformation(line):
    designs = _designs(line, 2, seed=6)
    objective = _field_objective(line)
    grad = jax.grad(objective)
    looped = jnp.stack([grad(d) for d in designs])
    np.testing.assert_allclose(jax.vmap(grad)(designs), looped, rtol=RTOL)
    _, pullback = jax.vjp(objective, designs[0])
    np.testing.assert_allclose(pullback(1.0)[0], looped[0], rtol=RTOL)
    np.testing.assert_allclose(pullback(2.0)[0], 2 * looped[0], rtol=RTOL)
    options = gfem.SolverOptions(linear_solver="direct", retain_memory=0)
    recompute = jax.grad(_field_objective(line, options))
    np.testing.assert_allclose(recompute(designs[1]), looped[1], rtol=RTOL)


def test_adjoint_tolerance_sets_the_field_gradient_accuracy(line):
    eps = _designs(line, 1, seed=7)[0]
    direct = gfem.SolverOptions(order=3, linear_solver="direct")
    exact = jax.grad(_field_objective(line, direct))(eps)

    def error(**kw):
        options = gfem.SolverOptions(order=3, linear_solver="bddc_gmres", **kw)
        grad = jax.grad(_field_objective(line, options))(eps)
        return float(jnp.abs(grad - exact).max() / jnp.abs(exact).max())

    tight = error(tolerance=1e-10)  # the adjoint follows the forward tolerance
    loose = error(tolerance=1e-10, adjoint_tolerance=1e-2)
    assert tight < 1e-6
    assert loose > 100 * tight
