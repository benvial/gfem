"""Gradients through adaptive sweeps: the reduced basis held fixed.

The two-layer line of the uniform-sweep cells (`gradient_cases.layered_line`:
random per-element lossy permittivity and conductivity, second-order
absorbing side walls) swept over 10-30 GHz with a frozen sample set. At a
sample frequency the full solution lies in the reduced basis, so S and its
frozen-basis gradient equal the full solve's to solver tolerance (the
matrix cells and that check are in test_gradient_matrix.py). Between
samples the gradient error follows the reduced model's field error.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradient_cases import layered_line, line_design, s_functional, tight, with_fields

import gfem

GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 21))
SAMPLES = (GRID[0], GRID[20], GRID[10], GRID[4], GRID[16])
ADAPTIVE = gfem.Driven(frequency=GRID, sweep="adaptive", samples=SAMPLES)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return layered_line(tmp_path_factory.mktemp("line"))


def _s_objective(line, values, name, options, problem, rows):  # noqa: PLR0917
    """<W, |S|^2> + Re(c S21) summed over the grid ``rows`` of ``problem``."""

    def objective(p):
        s = gfem.solve(
            with_fields(line, {**values, name: p}), problem, options=options
        ).s.values[jnp.asarray(rows)]
        return s_functional(s)

    return objective


def _relative(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


# --- between samples: the error follows the field error --------------------------

OFF = [2, 7, 13, 18]


def _field_errors(line, values, options):
    """Relative field error of the reduced model at each off-sample index."""
    model = with_fields(line, values)
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
    values, _ = line_design(line, seed=1)
    options = tight("direct")
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


# --- kept fields: their cotangents take the same reduced adjoint as S ------------


def test_a_port_voltage_read_from_a_field_differentiates_like_s(line):
    # Port 2's voltage functional applied to port 1's field is S21 (times
    # sqrt(2 R)): through the kept field's reduced coordinates it must give
    # the S gradient, on and between the samples alike.
    from gfem._engine import terms
    from gfem._engine.driven import driven_operator

    values, _ = line_design(line, seed=4)
    options = tight("direct")
    op = driven_operator(with_fields(line, values), options.order)
    voltage = terms.port_vectors(op.space, op.built, op.order, op.ports[1])[1]
    c = (0.3 - 0.7j) / np.sqrt(2 * 50.0)
    kept = [GRID[4], GRID[7]]  # a sample and a frequency between samples
    problem = gfem.Driven(
        frequency=GRID, sweep="adaptive", samples=SAMPLES, keep_fields=kept
    )

    def from_field(eps):
        result = gfem.solve(
            with_fields(line, {**values, "eps_r": eps}), problem, options=options
        )
        return sum(
            jnp.real(c * jnp.sum(voltage * result.e_field(f, "p1").coefficients))
            for f in kept
        )

    def from_s(eps):
        s = gfem.solve(
            with_fields(line, {**values, "eps_r": eps}), problem, options=options
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
    values, _ = line_design(line, seed=5)
    return _s_objective(line, values, "eps_r", options, ADAPTIVE, range(len(GRID)))


def _designs(line, k, seed=0):
    rng = np.random.default_rng(seed)
    n = len(line.mesh.tetrahedra)
    return jnp.asarray(rng.uniform(2.0, 6.0, (k, n)) * (1 - 0.02j))


def test_the_backward_keeps_and_needs_no_full_size_solver(line, monkeypatch):
    from gfem._engine import driven
    from gfem._engine.cache import ENTRIES

    eps = _designs(line, 1)[0]
    objective = _eps_objective(line, tight("direct"))
    value, pullback = jax.vjp(objective, eps)
    entry = ENTRIES.get(max(ENTRIES._entries))
    assert entry.handle is None
    assert sorted(entry.fields) == [
        "basis",
        "coordinates",
        "deembedding",
        "functionals",
        "matrices",
    ]

    def refuse(*_, **__):
        raise AssertionError("the frozen-basis backward prepared a full solver")

    monkeypatch.setattr(driven, "prepare", refuse)
    grad = pullback(jnp.ones_like(value))[0]
    monkeypatch.undo()
    np.testing.assert_allclose(grad, jax.grad(objective)(eps), rtol=RTOL)


def test_vmap_of_grad_equals_a_python_loop(line):
    designs = _designs(line, 2, seed=1)
    grad = jax.grad(_eps_objective(line, tight("direct")))
    looped = jnp.stack([grad(d) for d in designs])
    np.testing.assert_allclose(jax.vmap(grad)(designs), looped, rtol=RTOL)
    np.testing.assert_allclose(jax.jit(jax.vmap(grad))(designs), looped, rtol=RTOL)
    assert not np.allclose(looped[0], looped[1], rtol=1e-3)


def test_pullbacks_twice_and_after_eviction_equal_fresh_gradients(line):
    from gfem._engine.cache import ENTRIES

    eps = _designs(line, 1, seed=2)[0]
    objective = _eps_objective(line, tight("direct"))
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
