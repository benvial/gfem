"""Eigenfrequency and Q gradients against finite differences.

A simple mode's frequency differentiates with ``dk/dp = -x^T (dT/dp) x /
x^T T'(k) x`` (no solve): checked on a parallel-plate line resonator with
per-element permittivity and conductivity, terminated in resistors (a
quadratic problem), and on the two nonlinear problems: a conductor plate
and a second-order absorbing wall, whose ``T'(k)`` carries the derivative
of their frequency factors.

A degenerate pair (the TE101 / TE011 modes of a square-based PEC box on a
mesh with the square's symmetry) differentiates only through functions
symmetric in the pair: the sum passes finite differences, one member alone
raises `gfem.DegenerateModeError`.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions

import gfem

L, W, H = 10.0, 2.0, 1.0  # mm
TIGHT = gfem.SolverOptions(eigen_tolerance=1e-12)
#: Weights the real and the imaginary part of omega (GHz): frequency and loss.
WEIGHT = (1 - 30j) / (2 * math.pi * 1e9)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (8, 1, 1)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _ramp(mesh, region, start, stop):
    return jnp.asarray(np.linspace(start, stop, len(mesh.elements(region))))


def _terminated(mesh, eps, sigma):
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.MaterialField(eps_r=eps, conductivity=sigma)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=500.0),
            gfem.LumpedPort("p2", "port2", "+z", impedance=500.0),
        ],
    )


@pytest.mark.parametrize(("prop", "step"), [("eps_r", 1e-2), ("conductivity", 5e-2)])
def test_simple_mode_gradient_matches_finite_differences(line, prop, step):
    eps = _ramp(line, "dielectric", 3.5, 4.5)
    sigma = _ramp(line, "dielectric", 0.2, 0.6)

    def objective(p):
        values = {"eps_r": eps, "conductivity": sigma} | {prop: p}
        model = _terminated(line, values["eps_r"], values["conductivity"])
        result = gfem.solve(model, gfem.Eigenmode(target=7e9), options=TIGHT)
        return jnp.real(result.omega[0] * WEIGHT)

    p = eps if prop == "eps_r" else sigma
    rng = np.random.default_rng(0)
    assert_gradient_matches(objective, p, unit_directions(rng, p.shape, 3), step=step)


# --- nonlinear problems: T'(k) carries the frequency factors' derivatives --------


def test_conductor_mode_gradient_matches_finite_differences(line):
    eps = _ramp(line, "dielectric", 3.5, 4.5)

    def objective(p):
        model = gfem.Model(
            line,
            materials={"dielectric": gfem.MaterialField(eps_r=p)},
            boundaries=[gfem.PEC("bottom"), gfem.Conductor("top", 1e5)],
        )
        result = gfem.solve(model, gfem.Eigenmode(target=7e9), options=TIGHT)
        return jnp.real(result.omega[0] * WEIGHT)

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
        "core": _ramp(guide, "core", 9.0, 11.0),
        "tail": _ramp(guide, "tail", 1.0, 1.5),
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
        return jnp.real(result.omega[0] * WEIGHT)

    p = values[region]
    rng = np.random.default_rng(2)
    assert_gradient_matches(objective, p, unit_directions(rng, p.shape, 3), step=1e-2)


# --- a degenerate pair -------------------------------------------------------------

SIDE, DEPTH = 10.0, 15.0  # mm: TE101 = TE011 at 18.0 GHz, TM110 at 21.2 GHz
CAVITY_TARGET = 17e9


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    # mirrored Kuhn cells: the mesh has the square's symmetry, so the pair is
    # degenerate in the discrete problem too
    edges = (
        np.linspace(0, SIDE, 5),
        np.linspace(0, SIDE, 5),
        np.linspace(0, DEPTH, 7),
    )

    def surface(c):
        on = (c[0], SIDE - c[0], c[1], SIDE - c[1], c[2], DEPTH - c[2])
        return "wall" if min(np.abs(on)) < 1e-9 else ""

    path = tmp_path_factory.mktemp("cavity") / "cavity.msh"
    mesh = sm.structured(edges, lambda _: "fill", surface, mirror=(True, True, False))
    sm.write_msh22(path, mesh)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _cavity_modes(mesh, eps, n_modes):
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.MaterialField(eps_r=eps, tan_delta=1e-3)},
        boundaries=[gfem.PEC("wall")],
    )
    problem = gfem.Eigenmode(target=CAVITY_TARGET, n_modes=n_modes)
    return gfem.solve(model, problem, options=TIGHT)


def _uniform(mesh):
    return jnp.ones(len(mesh.elements("fill")))


def test_degenerate_pair_is_labelled_as_one_cluster(cavity):
    result = _cavity_modes(cavity, _uniform(cavity), 3)
    omega = np.asarray(result.omega)
    assert abs(omega[0] - omega[1]) <= 1e-12 * abs(omega[0])
    np.testing.assert_array_equal(np.asarray(result.cluster), [0, 0, 1])


def test_cluster_sum_matches_finite_differences(cavity):
    def objective(eps):
        omega = _cavity_modes(cavity, eps, 2).omega
        return jnp.real(jnp.sum(omega) * WEIGHT)

    eps = _uniform(cavity)
    rng = np.random.default_rng(3)
    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_cluster_smooth_max_matches_finite_differences(cavity):
    # any function symmetric in the pair: here a smooth max of the frequencies
    def objective(eps):
        f = _cavity_modes(cavity, eps, 3).frequency[:2] / 1e9
        return jax.nn.logsumexp(10 * f) / 10

    eps = _uniform(cavity)
    rng = np.random.default_rng(4)
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


# DegenerateModeError is raised on the host during the backward pass: JAX
# re-raises it as a JaxRuntimeError that keeps the message.
DEGENERATE = jax.errors.JaxRuntimeError


def test_single_mode_of_a_cluster_raises(cavity):
    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 2).omega[0])

    with pytest.raises(DEGENERATE, match="DegenerateModeError") as info:
        jax.grad(objective)(_uniform(cavity))
    message = str(info.value)
    assert "18.0" in message  # the cluster's frequencies
    assert "sum or mean" in message  # the remedy


def test_cluster_cut_by_n_modes_raises(cavity):
    # n_modes=1 keeps one member of the pair: the guard mode shows the rest
    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 1).omega[0])

    with pytest.raises(DEGENERATE, match=r"DegenerateModeError.*n_modes"):
        jax.grad(objective)(_uniform(cavity))


def test_simple_mode_beside_a_cluster_differentiates(cavity):
    # TM110 alone: no cotangent reaches the pair
    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 3).omega[2] * WEIGHT)

    eps = _uniform(cavity)
    rng = np.random.default_rng(5)
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_stop_gradient_on_the_other_member_still_raises(cavity):
    # the cotangents differ (one is zero): not symmetric, whatever the opt-out
    def objective(eps):
        omega = _cavity_modes(cavity, eps, 2).omega
        return jnp.real(omega[0] + jax.lax.stop_gradient(omega[1]))

    with pytest.raises(DEGENERATE, match="DegenerateModeError"):
        jax.grad(objective)(_uniform(cavity))


# --- JAX transforms ----------------------------------------------------------------


def _line_objective(line):
    sigma = _ramp(line, "dielectric", 0.2, 0.6)

    def objective(eps, options=TIGHT):
        model = _terminated(line, eps, sigma)
        result = gfem.solve(model, gfem.Eigenmode(target=7e9), options=options)
        return jnp.real(1 / result.q[0] + result.frequency[0] / 1e9)

    return objective


def test_gradient_under_jit_equals_eager(line):
    objective = _line_objective(line)
    eps = _ramp(line, "dielectric", 3.5, 4.5)
    eager = jax.grad(objective)(eps)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(eps), eager, rtol=1e-9)


def test_vmap_of_grad_equals_a_loop(line):
    objective = _line_objective(line)
    batch = jnp.stack(
        [_ramp(line, "dielectric", 3.5, 4.5), _ramp(line, "dielectric", 4.0, 3.0)]
    )
    looped = jnp.stack([jax.grad(objective)(e) for e in batch])
    np.testing.assert_allclose(jax.vmap(jax.grad(objective))(batch), looped, rtol=1e-9)


def test_evicted_entry_recomputes_the_same_gradient(line):
    objective = _line_objective(line)
    eps = _ramp(line, "dielectric", 3.5, 4.5)
    cached = jax.grad(objective)(eps)
    evicted = gfem.SolverOptions(eigen_tolerance=1e-12, retain_memory=0)
    recomputed = jax.grad(lambda e: objective(e, evicted))(eps)
    np.testing.assert_allclose(recomputed, cached, rtol=1e-9)


def test_pullback_called_twice_gives_the_same_gradient(line):
    objective = _line_objective(line)
    eps = _ramp(line, "dielectric", 3.5, 4.5)
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
    sigma = _ramp(line, "dielectric", 0.2, 0.6)
    problem = gfem.Eigenmode(target=10e9, n_modes=2)

    def modes(eps):
        return gfem.solve(_terminated(line, eps, sigma), problem, options=TIGHT)

    eps = _ramp(line, "dielectric", 3.5, 4.5)
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
