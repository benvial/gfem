r"""Junction coupling of two port modes: symmetry, limits, refusals, jit, gradients."""

from __future__ import annotations

import dataclasses

import bend_cases as bc
import finite_differences as fd
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

OPTIONS = gfem.SolverOptions(order=2, threads=1)
RADIUS = 30e-3


@pytest.fixture(scope="module")
def wr90(tmp_path_factory):
    path = tmp_path_factory.mktemp("bend_junction") / "wr90.msh"
    return bc.wr90_model(bc.wr90_mesh(path))


@pytest.fixture(scope="module")
def straight(wr90):
    return bc.solve_mode(wr90, bc.WR90_F, order=2)


@pytest.fixture(scope="module")
def bent(wr90):
    return bc.solve_mode(wr90, bc.WR90_F, bc.wr90_bend_m(RADIUS), order=2)


def test_the_coupling_is_symmetric(wr90, straight, bent):
    ab = gfem.junction_coupling(wr90, straight, bent)
    ba = gfem.junction_coupling(wr90, bent, straight)
    assert ab.shape == (1,)
    # the two orders sum and multiply the same four pairings: they differ by
    # a few rounding errors of the sums, $\sim 10^2 \epsilon_\mathrm{mach}$
    np.testing.assert_allclose(ab, ba, rtol=1e-13)


def test_a_mode_with_itself_couples_with_one(wr90, straight, bent):
    # numerator and denominator are the same sparse products up to the
    # order of a few sums: $\sim 10^3\epsilon_\mathrm{mach}$ over $10^4$ DOFs
    for mode in (straight, bent):
        np.testing.assert_allclose(
            gfem.junction_coupling(wr90, mode, mode), 1.0, rtol=1e-12
        )


def test_two_straight_modes_of_different_order_do_not_couple(wr90):
    f = 16e9  # TE10 and TE20 both propagate
    first = bc.solve_mode(wr90, f, order=2)
    second = bc.solve_mode(wr90, f, order=2, mode=2)
    # even and odd about the guide's axis on a mesh symmetric about it: the
    # pairing vanishes to round-off, and $\tau^2$ is its square
    assert abs(gfem.junction_coupling(wr90, first, second)[0]) <= 1e-10


def test_a_very_large_radius_is_the_straight_mode(wr90, straight):
    far = bc.solve_mode(
        wr90, bc.WR90_F, bc.wr90_bend_m(1e6 * bc.WR90_A * 1e-3), order=2
    )
    # the metric differs from the identity by $a/R = 10^{-6}$, so $1 - \tau^2$
    # is of order $10^{-12}$ (second order in the difference of the modes);
    # the solves add their eigen-solve error (tolerance $10^{-10}$), also
    # entering squared
    np.testing.assert_allclose(
        gfem.junction_coupling(wr90, straight, far), 1.0, rtol=1e-9
    )


def test_modes_of_different_orders_grids_or_ports_are_refused(wr90, straight):
    cubic = bc.solve_mode(wr90, bc.WR90_F, order=3)
    with pytest.raises(ValueError, match="element order"):
        gfem.junction_coupling(wr90, straight, cubic)
    shifted = bc.solve_mode(wr90, 1.1 * bc.WR90_F, order=2)
    with pytest.raises(ValueError, match="frequency grid"):
        gfem.junction_coupling(wr90, straight, shifted)
    longer = bc.solve_mode(wr90, bc.WR90_F, order=2)
    longer = dataclasses.replace(
        longer, frequency=jnp.asarray([bc.WR90_F, 1.1 * bc.WR90_F])
    )
    with pytest.raises(ValueError, match="frequency grid"):
        gfem.junction_coupling(wr90, straight, longer)
    other = dataclasses.replace(straight, port="p2")
    with pytest.raises(ValueError, match="one port"):
        gfem.junction_coupling(wr90, straight, other)


def test_arguments_of_the_wrong_kind_are_type_errors(wr90, straight):
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        gfem.junction_coupling(straight, straight, straight)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.junction_coupling(wr90, straight, "mode")  # ty: ignore[invalid-argument-type]


def test_node_coordinates_are_not_differentiable(wr90):
    nodes = np.array(wr90.mesh.imported_nodes)

    def objective(scale):
        mesh = jax.tree.unflatten(
            jax.tree.structure(wr90.mesh), [jnp.asarray(nodes) * scale]
        )
        moved = wr90.replace(mesh=mesh)
        mode = gfem.solve(moved, gfem.PortMode("p1", (bc.WR90_F,)), options=OPTIONS)
        return jnp.abs(gfem.junction_coupling(moved, mode, mode)[0])

    with pytest.raises(gfem.NonDifferentiableError, match="junction_coupling"):
        jax.grad(objective)(1.0)


def test_an_inverse_permeability_leaf_is_refused_by_name(wr90):
    def objective(mu):
        field = gfem.MaterialField.uniform("air", gfem.Material(mu_r=1.0))
        model = wr90.replace(materials={"air": _scaled(field, "mu_r", mu)})
        mode = gfem.solve(model, gfem.PortMode("p1", (bc.WR90_F,)), options=OPTIONS)
        return jnp.abs(gfem.junction_coupling(model, mode, mode)[0])

    with pytest.raises(gfem.NonDifferentiableError, match="mu_r"):
        jax.grad(objective)(1.0)


def _scaled(field, name, factor):
    """``field`` with its leaf ``name`` multiplied by ``factor`` (the others fixed)."""
    return jax.tree_util.tree_map_with_path(
        lambda path, v: v * factor if name in jax.tree_util.keystr(path) else v, field
    )


def _tau(model, fixed, radius):
    bend = bc.wr90_bend_m(radius)
    mode = gfem.solve(
        model, gfem.PortMode("p1", (bc.WR90_F,), bend=bend), options=OPTIONS
    )
    return jnp.abs(gfem.junction_coupling(model, fixed, mode)[0])


def test_the_radius_gradient_matches_differences(wr90, straight):
    rho = RADIUS - bc.WR90_A / 2 * 1e-3
    # fourth-order central difference, step $10^{-3}\rho$: truncation
    # $(\delta/\rho)^4 \sim 10^{-12}$, round-off $1.5\,\epsilon/\delta\rho^{-1}
    # \approx 1.5\times10^{-10}$ with $\epsilon \approx 10^{-13}$ the eigen-solve
    # error; the gate is the measured agreement (6e-11, one solve) times 30
    fd.assert_gradient_matches(
        lambda r: _tau(wr90, straight, r),
        jnp.asarray(RADIUS),
        [1.0],
        step=1e-3 * rho,
        rel=2e-9,
    )


def test_the_gradient_with_both_modes_bent_matches_differences(wr90, bent):
    rho = RADIUS - bc.WR90_A / 2 * 1e-3
    # the second mode's radius moves with the first one's fixed: the same
    # derivation and gate as the straight partner
    fd.assert_gradient_matches(
        lambda r: _tau(wr90, bent, r),
        jnp.asarray(1.5 * RADIUS),
        [1.0],
        step=1e-3 * rho,
        rel=2e-9,
    )


def test_the_permittivity_gradient_matches_differences(wr90):
    def objective(eps):
        field = gfem.MaterialField.uniform("air", gfem.Material(eps_r=1.5))
        model = wr90.replace(materials={"air": _scaled(field, "eps_r", eps)})
        straight = gfem.solve(model, gfem.PortMode("p1", (bc.WR90_F,)), options=OPTIONS)
        bent = gfem.solve(
            model,
            gfem.PortMode("p1", (bc.WR90_F,), bend=bc.wr90_bend_m(RADIUS)),
            options=OPTIONS,
        )
        return jnp.abs(gfem.junction_coupling(model, straight, bent)[0])

    # $\tau^2$ is smooth in $\epsilon_r$ on the scale 1: step $10^{-2}$ gives a
    # truncation $\sim 10^{-8}$ and round-off $\sim 10^{-11}$ (fourth order)
    fd.assert_gradient_matches(objective, jnp.asarray(1.0), [1.0], step=1e-2, rel=1e-6)


def test_jit_equals_eager(wr90, straight):
    def objective(r):
        return _tau(wr90, straight, r)

    radius = jnp.asarray(RADIUS)
    # the same host solves and pairings run either way; a different reduction
    # order on another platform moves the result by $\sim 10^2\epsilon_\mathrm{mach}$
    np.testing.assert_allclose(
        jax.jit(objective)(radius), objective(radius), rtol=1e-10
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(radius),
        jax.grad(objective)(radius),
        rtol=1e-10,
    )
