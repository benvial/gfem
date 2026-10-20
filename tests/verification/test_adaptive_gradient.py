"""Gradients through adaptive sweeps: the reduced basis held fixed.

The two-layer line of the uniform-sweep cells (random per-element lossy
permittivity and conductivity, second-order absorbing side walls) swept
over 10-30 GHz with a frozen sample set. At a sample frequency the full
solution lies in the reduced basis, so S and its frozen-basis gradient
equal the full solve's to solver tolerance. Between samples the gradient
error follows the reduced model's field error.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import (
    assert_gradient_matches,
    assert_taylor_second_order,
    unit_directions,
)

import gfem

SOLVERS = ["direct", "bddc_gmres"]
GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 21))
SAMPLES = (GRID[0], GRID[20], GRID[10], GRID[4], GRID[16])
ON = [GRID.index(f) for f in SAMPLES]
LAYERS = (("sub", 0.5), ("cover", 0.5))
ADAPTIVE = gfem.Driven(frequency=GRID, sweep="adaptive", samples=SAMPLES)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1), layers=LAYERS))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"sub": gfem.Material(), "cover": gfem.Material()},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _design(line, seed=0):
    """Random P0 permittivity (lossy) and conductivity over both layers."""
    rng = np.random.default_rng(seed)
    n = len(line.mesh.tetrahedra)
    eps = rng.uniform(2.0, 6.0, n) * (1 - 0.02j)
    sigma = rng.uniform(0.5, 2.0, n)
    return {"eps_r": jnp.asarray(eps), "conductivity": jnp.asarray(sigma)}, rng


def _model(line, values):
    n_sub = len(line.mesh.elements("sub"))
    parts = {
        "sub": {k: v[:n_sub] for k, v in values.items()},
        "cover": {k: v[n_sub:] for k, v in values.items()},
    }
    return line.replace(
        materials={r: gfem.MaterialField(**p) for r, p in parts.items()}
    )


def _options(solver):
    return gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12)


WEIGHTS = np.array([[1.0, 0.4], [-0.6, 1.5]])


def _s_objective(line, values, name, options, problem, rows):  # noqa: PLR0917
    """<W, |S|^2> + Re(c S21) summed over the grid ``rows`` of ``problem``."""

    def objective(p):
        s = gfem.solve(
            _model(line, {**values, name: p}), problem, options=options
        ).s.values[jnp.asarray(rows)]
        return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) + jnp.sum(
            jnp.real((0.3 - 0.7j) * s[:, 1, 0])
        )

    return objective


def _relative(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


# --- exact at the samples --------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["eps_r", "conductivity"])
def test_gradient_equals_the_full_solve_at_sample_frequencies(line, solver, name):
    values, _ = _design(line)
    options = _options(solver)
    adaptive = _s_objective(line, values, name, options, ADAPTIVE, ON)
    full = _s_objective(
        line, values, name, options, gfem.Driven(frequency=SAMPLES), range(5)
    )
    p = values[name]
    value, grad = jax.value_and_grad(adaptive)(p)
    full_value, full_grad = jax.value_and_grad(full)(p)
    assert float(value) == pytest.approx(float(full_value), rel=1e-9)
    assert _relative(grad, full_grad) <= 1e-7


# --- between samples: the error follows the field error --------------------------

OFF = [2, 7, 13, 18]


def _field_errors(line, values, options):
    """Relative field error of the reduced model at each off-sample index."""
    model = _model(line, values)
    kept = [GRID[i] for i in OFF]
    full = gfem.solve(
        model, gfem.Driven(frequency=kept, keep_fields=True), options=options
    )
    problem = gfem.Driven(
        frequency=GRID, sweep="adaptive", samples=SAMPLES, keep_fields=kept
    )
    reduced = gfem.solve(model, problem, options=options)
    errors = []
    for f in kept:
        worst = 0.0
        for port in ("p1", "p2"):
            x = full.e_field(f, port).coefficients
            y = reduced.e_field(f, port).coefficients
            worst = max(worst, _relative(y, x))
        errors.append(worst)
    return errors


@pytest.mark.parametrize("name", ["eps_r", "conductivity"])
def test_gradient_error_between_samples_follows_the_field_error(line, name):
    values, _ = _design(line, seed=1)
    options = _options("direct")
    errors = _field_errors(line, values, options)
    for i, error in zip(OFF, errors, strict=True):
        adaptive = _s_objective(line, values, name, options, ADAPTIVE, [i])
        full = _s_objective(
            line, values, name, options, gfem.Driven(frequency=GRID[i]), [0]
        )
        grad = jax.grad(adaptive)(values[name])
        full_grad = jax.grad(full)(values[name])
        assert 0 < error < 1e-1
        assert _relative(grad, full_grad) <= 10 * error


# --- the verification matrix: driven x {eps, sigma} x solver x adaptive sweep ----
# At the sample frequencies the reduced model reproduces the full solves of
# the perturbed design too, so differences of the adaptive pipeline itself
# check the frozen-basis gradient there.

# Fourth-order differences (see tests/finite_differences.py).
STEPS = {"eps_r": 1e-2, "conductivity": 5e-2}
TAYLOR = {"eps_r": [1e-1, 1e-2, 1e-3], "conductivity": [1.0, 1e-1, 1e-2]}


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["eps_r", "conductivity"])
def test_s_gradient_matches_central_differences(line, solver, name):
    values, rng = _design(line, seed=2)
    objective = _s_objective(line, values, name, _options(solver), ADAPTIVE, ON)
    p = values[name]
    directions = unit_directions(rng, p.shape, 3)
    assert_gradient_matches(objective, p, directions, step=STEPS[name])


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["eps_r", "conductivity"])
def test_s_taylor_remainder_is_second_order(line, solver, name):
    values, rng = _design(line, seed=3)
    objective = _s_objective(line, values, name, _options(solver), ADAPTIVE, ON)
    p = values[name]
    (d,) = unit_directions(rng, p.shape, 1)
    assert_taylor_second_order(objective, p, d, TAYLOR[name])


# --- kept fields: their cotangents take the same reduced adjoint as S ------------


def test_a_port_voltage_read_from_a_field_differentiates_like_s(line):
    # Port 2's voltage functional applied to port 1's field is S21 (times
    # sqrt(2 R)): through the kept field's reduced coordinates it must give
    # the S gradient, on and between the samples alike.
    from gfem._engine import terms
    from gfem._engine.driven import driven_operator

    values, _ = _design(line, seed=4)
    options = _options("direct")
    op = driven_operator(_model(line, values), options.order)
    voltage = terms.port_vectors(op.space, op.built, op.order, op.ports[1])[1]
    c = (0.3 - 0.7j) / np.sqrt(2 * 50.0)
    kept = [GRID[4], GRID[7]]  # a sample and a frequency between samples
    problem = gfem.Driven(
        frequency=GRID, sweep="adaptive", samples=SAMPLES, keep_fields=kept
    )

    def from_field(eps):
        result = gfem.solve(
            _model(line, {**values, "eps_r": eps}), problem, options=options
        )
        return sum(
            jnp.real(c * jnp.sum(voltage * result.e_field(f, "p1").coefficients))
            for f in kept
        )

    def from_s(eps):
        s = gfem.solve(
            _model(line, {**values, "eps_r": eps}), problem, options=options
        ).s
        return jnp.sum(jnp.real((0.3 - 0.7j) * s["p2", "p1"][jnp.asarray([4, 7])]))

    eps = values["eps_r"]
    assert float(from_field(eps)) == pytest.approx(float(from_s(eps)), rel=1e-10)
    np.testing.assert_allclose(
        jax.grad(from_field)(eps), jax.grad(from_s)(eps), rtol=1e-9
    )


# --- transformations and the kept reduced model ----------------------------------
# solve-vs-solve differs by about 1 ulp (threaded assembly)
RTOL = 1e-9


def _eps_objective(line, options):
    values, _ = _design(line, seed=5)
    return _s_objective(line, values, "eps_r", options, ADAPTIVE, range(len(GRID)))


def _designs(line, k, seed=0):
    rng = np.random.default_rng(seed)
    n = len(line.mesh.tetrahedra)
    return jnp.asarray(rng.uniform(2.0, 6.0, (k, n)) * (1 - 0.02j))


def test_the_backward_keeps_and_needs_no_full_size_solver(line, monkeypatch):
    from gfem._engine import driven
    from gfem._engine.cache import ENTRIES

    eps = _designs(line, 1)[0]
    objective = _eps_objective(line, _options("direct"))
    value, pullback = jax.vjp(objective, eps)
    entry = ENTRIES.get(max(ENTRIES._entries))
    assert entry.handle is None
    assert sorted(entry.fields) == ["basis", "coordinates", "operators", "voltage"]

    def refuse(*_, **__):
        raise AssertionError("the frozen-basis backward prepared a full solver")

    monkeypatch.setattr(driven, "prepare", refuse)
    grad = pullback(jnp.ones_like(value))[0]
    monkeypatch.undo()
    np.testing.assert_allclose(grad, jax.grad(objective)(eps), rtol=RTOL)


def test_vmap_of_grad_equals_a_python_loop(line):
    designs = _designs(line, 2, seed=1)
    grad = jax.grad(_eps_objective(line, _options("direct")))
    looped = jnp.stack([grad(d) for d in designs])
    np.testing.assert_allclose(jax.vmap(grad)(designs), looped, rtol=RTOL)
    np.testing.assert_allclose(jax.jit(jax.vmap(grad))(designs), looped, rtol=RTOL)
    assert not np.allclose(looped[0], looped[1], rtol=1e-3)


def test_pullbacks_twice_and_after_eviction_equal_fresh_gradients(line):
    from gfem._engine.cache import ENTRIES

    eps = _designs(line, 1, seed=2)[0]
    objective = _eps_objective(line, _options("direct"))
    fresh = jax.grad(objective)(eps)
    _, pullback = jax.vjp(objective, eps)
    np.testing.assert_allclose(pullback(1.0)[0], fresh, rtol=RTOL)
    np.testing.assert_allclose(pullback(2.0)[0], 2 * fresh, rtol=RTOL)
    ENTRIES.clear()
    np.testing.assert_allclose(pullback(1.0)[0], fresh, rtol=RTOL)
    options = gfem.SolverOptions(
        order=2, linear_solver="direct", tolerance=1e-12, retain_memory=0
    )
    recompute = jax.grad(_eps_objective(line, options))
    np.testing.assert_allclose(recompute(eps), fresh, rtol=RTOL)
