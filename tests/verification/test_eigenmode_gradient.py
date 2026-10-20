"""Eigenfrequency and Q gradients against finite differences.

A simple mode's frequency differentiates with ``dk/dp = -x^T (dT/dp) x /
x^T T'(k) x`` (no solve). The verification matrix cells (a terminated line
resonator: a quadratic problem; eps, sigma, fill and node coordinates x
solver) and the degenerate-cluster tests are in test_gradient_matrix.py.
Here the two nonlinear problems, a conductor plate and a second-order
absorbing wall, whose ``T'(k)`` carries the derivative of their frequency
factors; the JAX transformations; and the mode-tracking recipe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import OMEGA_WEIGHT, ramp, resonator_mesh, terminated

import gfem

TIGHT = gfem.SolverOptions(eigen_tolerance=1e-12)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return resonator_mesh(tmp_path_factory.mktemp("line"))


# --- nonlinear problems: T'(k) carries the frequency factors' derivatives --------


def test_conductor_mode_gradient_matches_finite_differences(line):
    eps = ramp(line, "dielectric", 3.5, 4.5)

    def objective(p):
        model = gfem.Model(
            line,
            materials={"dielectric": gfem.MaterialField(eps_r=p)},
            boundaries=[gfem.PEC("bottom"), gfem.Conductor("top", 1e5)],
        )
        result = gfem.solve(model, gfem.Eigenmode(target=7e9), options=TIGHT)
        return jnp.real(result.omega[0] * OMEGA_WEIGHT)

    rng = np.random.default_rng(1)
    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


CORE, TAIL, WIDTH, HEIGHT = 3.0, 3.0, 10.0, 1.0  # mm


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    # a TE10 guide shorted at x = 0, a dielectric core, ended by an absorbing
    # wall: a mode near 24 GHz with Q about 10
    length = CORE + TAIL
    edges = (
        np.concatenate([np.linspace(0, CORE, 4), np.linspace(CORE, length, 4)[1:]]),
        np.linspace(0, WIDTH, 5),
        np.linspace(0, HEIGHT, 2),
    )

    def surface(c):
        on = [abs(c[0]), abs(c[0] - length)]
        if min(on) < 1e-9:
            return "short" if on[0] < 1e-9 else "end"
        if min(abs(c[1]), abs(c[1] - WIDTH), abs(c[2]), abs(c[2] - HEIGHT)) < 1e-9:
            return "wall"
        return ""

    path = tmp_path_factory.mktemp("guide") / "guide.msh"
    sm.write_msh22(
        path,
        sm.structured(edges, lambda c: "core" if c[0] < CORE else "tail", surface),
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.mark.parametrize("region", ["core", "tail"])
def test_second_order_absorbing_mode_gradient_matches_finite_differences(guide, region):
    # the tail's elements own the absorbing wall's faces: its permittivity
    # also enters the absorbing coefficients
    values = {
        "core": ramp(guide, "core", 9.0, 11.0),
        "tail": ramp(guide, "tail", 1.0, 1.5),
    }

    def objective(p):
        fields = values | {region: p}
        model = gfem.Model(
            guide,
            materials={r: gfem.MaterialField(eps_r=v) for r, v in fields.items()},
            boundaries=[
                gfem.PEC("short"),
                gfem.PEC("wall"),
                gfem.Absorbing("end", order=2),
            ],
        )
        result = gfem.solve(model, gfem.Eigenmode(target=24e9), options=TIGHT)
        return jnp.real(result.omega[0] * OMEGA_WEIGHT)

    p = values[region]
    rng = np.random.default_rng(2)
    assert_gradient_matches(objective, p, unit_directions(rng, p.shape, 3), step=1e-2)


# --- JAX transforms ----------------------------------------------------------------


def _line_objective(line):
    sigma = ramp(line, "dielectric", 0.2, 0.6)

    def objective(eps, options=TIGHT):
        model = terminated(line, eps, sigma)
        result = gfem.solve(model, gfem.Eigenmode(target=7e9), options=options)
        return jnp.real(1 / result.q[0] + result.frequency[0] / 1e9)

    return objective


def test_gradient_under_jit_equals_eager(line):
    objective = _line_objective(line)
    eps = ramp(line, "dielectric", 3.5, 4.5)
    eager = jax.grad(objective)(eps)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(eps), eager, rtol=1e-9)


def test_vmap_of_grad_equals_a_loop(line):
    objective = _line_objective(line)
    batch = jnp.stack(
        [ramp(line, "dielectric", 3.5, 4.5), ramp(line, "dielectric", 4.0, 3.0)]
    )
    looped = jnp.stack([jax.grad(objective)(e) for e in batch])
    np.testing.assert_allclose(jax.vmap(jax.grad(objective))(batch), looped, rtol=1e-9)


def test_evicted_entry_recomputes_the_same_gradient(line):
    objective = _line_objective(line)
    eps = ramp(line, "dielectric", 3.5, 4.5)
    cached = jax.grad(objective)(eps)
    evicted = gfem.SolverOptions(eigen_tolerance=1e-12, retain_memory=0)
    recomputed = jax.grad(lambda e: objective(e, evicted))(eps)
    np.testing.assert_allclose(recomputed, cached, rtol=1e-9)


def test_pullback_called_twice_gives_the_same_gradient(line):
    objective = _line_objective(line)
    eps = ramp(line, "dielectric", 3.5, 4.5)
    _, pullback = jax.vjp(objective, eps)
    (first,) = pullback(1.0)
    (second,) = pullback(1.0)
    np.testing.assert_array_equal(first, second)


# --- the mode-tracking recipe ------------------------------------------------------


def _tracked(modes, reference):
    """Index of the mode whose field is most like ``reference`` (the recipe's)."""
    fields = jax.lax.stop_gradient(jnp.stack([f.coefficients for f in modes.e_fields]))
    overlap = jnp.abs(fields.conj() @ reference) / jnp.linalg.norm(fields, axis=1)
    return jnp.argmax(overlap)


def test_mode_picked_by_overlap_differentiates_like_its_index(line):
    sigma = ramp(line, "dielectric", 0.2, 0.6)
    problem = gfem.Eigenmode(target=10e9, n_modes=2)

    def modes(eps):
        return gfem.solve(terminated(line, eps, sigma), problem, options=TIGHT)

    eps = ramp(line, "dielectric", 3.5, 4.5)
    start = modes(eps)
    # follow the second mode (the one farther from the target)
    reference = start.e_field(1).coefficients

    def tracked(e):
        result = modes(e)
        return result.frequency[_tracked(result, reference)] / 1e9

    def indexed(e):
        return modes(e).frequency[1] / 1e9

    assert int(_tracked(start, reference)) == 1
    np.testing.assert_allclose(
        jax.grad(tracked)(eps), jax.grad(indexed)(eps), rtol=1e-9
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(tracked))(eps), jax.grad(indexed)(eps), rtol=1e-9
    )
