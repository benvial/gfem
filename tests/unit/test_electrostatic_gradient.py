"""jax.grad of a capacitance objective with respect to material fields."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants
import structured_meshes as sm

import gfem

DATA = Path(__file__).parent / "data"
EPS0 = scipy.constants.epsilon_0
UM = 1e-6
PROBLEM = gfem.Electrostatic(terminals={"plate": "top"}, ground="bottom")


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "plate.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, (("oxide", 0.5),), (2, 1, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"oxide": gfem.Material(eps_r=3.9)}
    )


def _c(model, field):
    return gfem.solve(model.with_material("oxide", field), PROBLEM).capacitance[0, 0]


def test_gradient_of_plate_capacitance_by_uniform_permittivity(plate):
    # C = eps0 eps_r A / d, so dC / d eps_r = eps0 A / d.
    def c_of(eps):
        return _c(plate, gfem.MaterialField(eps_r=eps, region="oxide"))

    grad = jax.grad(c_of)(jnp.asarray(3.9))
    assert float(grad) == pytest.approx(EPS0 * (4 * UM) * (3 * UM) / (0.5 * UM))


def _field_c(plate, eps):
    return _c(plate, gfem.MaterialField(eps_r=eps))


def test_per_element_gradient_sums_to_capacitance_over_permittivity(plate):
    # C is homogeneous of degree 1 in eps: sum_e eps_e dC/deps_e = C.
    n = len(plate.mesh.elements("oxide"))
    eps = jnp.asarray(np.random.default_rng(0).uniform(2.0, 6.0, n))
    c, grad = jax.value_and_grad(lambda e: _field_c(plate, e))(eps)
    assert grad.shape == (n,)
    assert np.all(np.asarray(grad) > 0)
    assert float(grad @ eps) == pytest.approx(float(c), rel=1e-12)


def test_solve_and_its_gradient_run_under_jit(plate):
    n = len(plate.mesh.elements("oxide"))
    eps = jnp.linspace(3.0, 5.0, n)
    c_of = jax.value_and_grad(lambda e: _field_c(plate, e))
    c, grad = c_of(eps)
    c_jit, grad_jit = jax.jit(c_of)(eps)
    assert float(c_jit) == float(c)
    np.testing.assert_allclose(grad_jit, grad, rtol=1e-14)
    assert float(jax.jit(lambda e: _field_c(plate, e))(eps)) == float(c)


def test_elements_of_an_excluded_conductor_have_an_exact_zero_gradient():
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    model = gfem.Model(
        mesh,
        materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()},
        excluded=(gfem.enclosed_by("metal_xy", "metal_z"),),
    )
    problem = gfem.Electrostatic(
        terminals={"metal": ["metal_xy", "metal_z"]}, ground="sub__None"
    )
    # the metal entity is inside region "sub": its tets are in the field
    sub = mesh.elements("sub")
    inside = ~model._domain.active[sub]
    assert inside.any()

    def c_of(eps):
        m = model.with_material("sub", gfem.MaterialField(eps_r=eps))
        return gfem.solve(m, problem).capacitance[0, 0]

    grad = np.asarray(jax.grad(c_of)(jnp.full(len(sub), 11.7)))
    assert not grad[inside].any()
    assert (grad[~inside] >= 0).all()
    assert grad[~inside].max() > 0
