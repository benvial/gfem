"""Shape gradients: objectives differentiated with respect to node positions.

The node-coordinate cells of the verification matrix (electrostatic,
driven on lumped and wave ports, uniform and adaptive, eigenmode; both
solvers) are in test_gradient_matrix.py, on the designs of
`gradient_cases`. Here: shape gradients under jit and vmap, of a kept-field
functional, of a raw node displacement (no morph), and the raise when a
port's nodes move.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import WEIGHTS, capacitor, shape_line, wave_line, wave_mesh

import gfem
from gfem.shape import boundary_morph

STEP = 2e-6  # metres along a unit direction in t (layers are 0.5 mm)


@pytest.fixture(scope="module")
def capacitor_model(tmp_path_factory):
    return capacitor(tmp_path_factory.mktemp("plate"))


ELECTROSTATIC = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


def test_shape_gradient_under_jit_and_vmap(capacitor_model):
    morph = boundary_morph(capacitor_model, "oxide__nitride", grows="oxide")
    options = gfem.SolverOptions(order=2, linear_solver="direct")

    def objective(t):
        model = capacitor_model.replace(mesh=morph(t))
        return (
            gfem.solve(model, ELECTROSTATIC, options=options).capacitance[0, 0] * 1e12
        )

    rng = np.random.default_rng(6)
    batch = jnp.asarray(rng.uniform(-1e-5, 1e-5, (2, morph.n_params)))
    loop = np.stack([np.asarray(jax.grad(objective)(t)) for t in batch])
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(batch[0]), loop[0], rtol=1e-9
    )
    np.testing.assert_allclose(jax.vmap(jax.grad(objective))(batch), loop, rtol=1e-9)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return shape_line(tmp_path_factory.mktemp("line"))


GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 11))


def test_driven_kept_field_shape_gradient_matches_fd(line):
    # a field functional: one adjoint solve per frequency, then the same pass
    problem = gfem.Driven(frequency=GRID[5], excite="p1", keep_fields=True)
    morph = boundary_morph(line, "sub__cover", grows="sub")
    options = gfem.SolverOptions(order=2, linear_solver="direct")

    def objective(t):
        result = gfem.solve(line.replace(mesh=morph(t)), problem, options=options)
        e = result.e_field(GRID[5]).coefficients
        weights = np.random.default_rng(7).uniform(0.5, 1.5, e.shape[0])
        return jnp.sum(weights * jnp.abs(e) ** 2) * 1e-6

    t = jnp.zeros(morph.n_params)
    rng = np.random.default_rng(3)
    assert_gradient_matches(objective, t, unit_directions(rng, t.shape, 2), step=STEP)


def test_node_gradient_of_a_raw_displacement_matches_fd(line):
    # every node but the port surfaces' moves (no morph): the gradient
    # with respect to model.mesh.nodes itself
    problem = gfem.Driven(frequency=GRID[5])
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    mesh = line.mesh
    x0 = jnp.asarray(mesh.nodes)

    def objective(x):
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        s = gfem.solve(line.replace(mesh=moved), problem, options=options).s.values
        return jnp.sum(WEIGHTS * jnp.abs(s[0]) ** 2)

    ports = np.isin(
        mesh.triangle_tags, [mesh.region(s).tag for s in ("port1", "port2")]
    )
    held = np.unique(mesh.triangles[ports])
    directions = []
    for d in unit_directions(np.random.default_rng(5), x0.shape, 2):
        d[held] = 0.0
        directions.append(d / np.linalg.norm(d))
    assert_gradient_matches(objective, x0, directions, step=STEP)


def test_moved_port_nodes_raise_port_geometry_error(line):
    problem = gfem.Driven(frequency=GRID[5])
    mesh = line.mesh
    node = int(mesh.triangles[mesh.triangle_tags == mesh.region("port1").tag][0, 0])

    def objective(dx):
        x = jnp.asarray(mesh.nodes).at[node, 0].add(dx)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        s = gfem.solve(line.replace(mesh=moved), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    # the port's own geometry is frozen in shape gradients: moving it raises
    with pytest.raises(jax.errors.JaxRuntimeError, match=r"PortGeometryError.*port1"):
        jax.grad(objective)(1e-6)


def test_moved_wave_port_nodes_raise_port_geometry_error(tmp_path):
    line = wave_line(wave_mesh(tmp_path), "lossless")
    problem = gfem.Driven(frequency=12e9)
    mesh = line.mesh
    face = mesh.triangles[mesh.triangle_tags == mesh.region("port2").tag]
    node = int(face[0, 0])

    def objective(dy):
        x = jnp.asarray(mesh.nodes).at[node, 1].add(dy)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        s = gfem.solve(line.replace(mesh=moved), problem).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    with pytest.raises(jax.errors.JaxRuntimeError, match=r"PortGeometryError.*port2"):
        jax.grad(objective)(1e-6)
