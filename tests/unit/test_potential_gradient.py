"""jax.grad of terminal-potential functionals with respect to material fields."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions

import gfem

LAYERS = (("oxide", 0.5), ("nitride", 0.25))
PROBLEM = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (2, 2, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={
            "oxide": gfem.Material(eps_r=3.9),
            "nitride": gfem.Material(eps_r=7.5),
        },
    )


def _potential(stack, eps, order=2):
    model = stack.with_material("oxide", gfem.MaterialField(eps_r=eps))
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    return gfem.solve(model, PROBLEM, options=options).potential["top"]


def test_interface_potential_follows_the_series_divider(stack):
    # V at the oxide / nitride interface = (e2 / d2) / (e1 / d1 + e2 / d2):
    # dV / de1 = -(V / (e1 / d1 + e2 / d2)) / d1.
    n = len(stack.mesh.elements("oxide"))
    e1, e2, d1, d2 = 3.9, 7.5, 0.5, 0.25

    def interface(e):
        out = _potential(stack, jnp.full(n, e), order=1).coefficients
        # order 1: the coefficients are the nodal values; take the interface
        # nodes through an export-independent mask built from the mesh
        return out @ mask

    mesh = stack.mesh
    tag = mesh.region("oxide__nitride").tag
    on = np.unique(mesh.triangles[mesh.triangle_tags == tag])
    v = _potential(stack, jnp.full(n, e1), order=1)
    # order-1 coefficients are the potential at the mesh nodes, in mesh order
    nodal = np.asarray(v.coefficients)
    assert nodal.shape == (len(mesh.imported_nodes),)
    mask = np.zeros(len(nodal))
    mask[on] = 1.0 / len(on)
    g = e1 / d1 + e2 / d2
    expected_v = (e2 / d2) / g
    assert float(interface(e1)) == pytest.approx(expected_v, rel=1e-12)
    grad = float(jax.grad(interface)(e1))
    assert grad == pytest.approx(-expected_v / g / d1, rel=1e-10)


def test_potential_gradient_matches_finite_differences(stack):
    n = len(stack.mesh.elements("oxide"))
    rng = np.random.default_rng(0)
    eps = jnp.asarray(rng.uniform(3.0, 5.0, n))
    ndof = _potential(stack, eps).coefficients.shape[0]
    weights = jnp.asarray(rng.uniform(-1.0, 1.0, ndof))

    def objective(e):
        return _potential(stack, e).coefficients @ weights

    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_capacitance_and_potential_objectives_add(stack):
    n = len(stack.mesh.elements("oxide"))
    eps = jnp.linspace(3.0, 5.0, n)

    def solve(e):
        model = stack.with_material("oxide", gfem.MaterialField(eps_r=e))
        return gfem.solve(model, PROBLEM)

    def c(e):
        return solve(e).capacitance[0, 0] * 1e15

    def v(e):
        return solve(e).potential["top"].coefficients.sum()

    both = jax.grad(lambda e: c(e) + v(e))(eps)
    np.testing.assert_allclose(both, jax.grad(c)(eps) + jax.grad(v)(eps), rtol=1e-10)
