"""Material-field gradients of probed driven fields against finite differences.

A two-layer line between PEC plates with lumped ports and absorbing side
walls, a random per-element permittivity, and a field kept at one
frequency. The objective probes E at a few points; its gradient chains the
probe's host kernel (the transpose of point evaluation) into the driven
solve's adjoint.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions

import gfem

LAYERS = (("sub", 0.5), ("cover", 0.5))
POINTS = np.array([[1.1, 0.7, 0.3], [2.9, 1.3, 0.8], [2.0, 1.0, 0.5]]) * 1e-3
WEIGHTS = np.array([[0.3, -1.0, 2.0], [1.0, 0.5, -0.2], [0.0, 0.7, 1.1]])


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1), layers=LAYERS))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"sub": gfem.Material(), "cover": gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.Absorbing("side")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _model(line, eps):
    n_sub = len(line.mesh.elements("sub"))
    return line.replace(
        materials={
            "sub": gfem.MaterialField(eps_r=eps[:n_sub]),
            "cover": gfem.MaterialField(eps_r=eps[n_sub:]),
        }
    )


@pytest.mark.parametrize("solver", ["direct", "bddc_gmres"])
def test_probe_objective_gradient_matches_finite_differences(line, solver):
    options = gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12)
    problem = gfem.Driven(frequency=(20e9, 25e9), keep_fields=25e9, excite="p1")

    def objective(eps):
        result = gfem.solve(_model(line, eps), problem, options=options)
        e = result.e_field(25e9).at(POINTS)
        return jnp.sum(WEIGHTS * jnp.abs(e) ** 2) / 1e6

    rng = np.random.default_rng(0)
    n = len(line.mesh.tetrahedra)
    eps = jnp.asarray(rng.uniform(2.0, 6.0, n))
    grad = jax.grad(objective)(eps)
    assert np.abs(np.asarray(grad)).max() > 0
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2, grad=grad)


def test_probe_under_jit_matches_eager(line):
    problem = gfem.Driven(frequency=25e9, keep_fields=True, excite="p1")
    eps = jnp.full(len(line.mesh.tetrahedra), 3.0)

    def probe(eps):
        result = gfem.solve(_model(line, eps), problem)
        return result.e_field(25e9).at(POINTS)

    np.testing.assert_allclose(jax.jit(probe)(eps), probe(eps), rtol=1e-12)
