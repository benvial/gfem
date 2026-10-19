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


@pytest.mark.parametrize("matrix", ["mutual", "inverse"])
def test_gradient_through_derived_matrices_matches_central_differences(
    tmp_path, matrix
):
    # mutual and inverse are jnp functions of the capacitance leaf, so they
    # differentiate through the capacitance pullback.
    layers = (("oxide", 0.5), ("nitride", 0.25))
    path = sm.write_msh22(tmp_path / "stack.msh", sm.plate(4.0, 3.0, layers, (2, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(
        mesh, materials={"oxide": gfem.Material(), "nitride": gfem.Material(eps_r=7.5)}
    )
    problem = gfem.Electrostatic(
        terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
    )
    weights = np.array([[1.0, 0.3], [-0.7, 2.0]])
    scale = EPS0 * UM if matrix == "mutual" else 1 / (EPS0 * UM)

    def objective(eps):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=eps))
        value = getattr(gfem.solve(m, problem), matrix)
        return jnp.sum(weights * value) / scale  # O(1) values

    rng = np.random.default_rng(3)
    eps = jnp.asarray(rng.uniform(2.0, 6.0, len(mesh.elements("oxide"))))
    grad = jax.grad(objective)(eps)
    assert np.abs(np.asarray(grad)).max() > 0
    h = 1e-4
    for _ in range(2):
        d = rng.standard_normal(eps.shape)
        fd = (objective(eps + h * d) - objective(eps - h * d)) / (2 * h)
        assert float(grad @ d) == pytest.approx(float(fd), rel=1e-6)


@pytest.mark.parametrize("solver", ["direct", "bddc_gmres"])
def test_kept_entry_and_recomputed_entry_give_the_same_gradient(plate, solver):
    # The forward keeps its potentials (not its solver: a capacitance
    # gradient needs no solve) in ENTRIES; with retain_memory=0 nothing is
    # kept and the backward re-runs the forward.
    from gfem._engine.cache import ENTRIES

    n = len(plate.mesh.elements("oxide"))
    eps = jnp.linspace(3.0, 5.0, n)

    def grad(**kw):
        options = gfem.SolverOptions(linear_solver=solver, tolerance=1e-12, **kw)

        def c_of(e):
            m = plate.with_material("oxide", gfem.MaterialField(eps_r=e))
            return gfem.solve(m, PROBLEM, options=options).capacitance[0, 0]

        return jax.grad(c_of)(eps)

    ENTRIES.clear()
    kept = grad()
    assert len(ENTRIES) == 1
    (entry,) = ENTRIES._entries.values()
    assert entry.handle is None
    assert entry.nbytes == entry.fields["potential"].nbytes
    ENTRIES.clear()
    assert entry.fields == {}
    recomputed = grad(retain_memory=0)
    assert len(ENTRIES) == 0
    np.testing.assert_allclose(recomputed, kept, rtol=1e-12)
