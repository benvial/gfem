"""Shape gradients: objectives differentiated with respect to node positions.

Every design moves a surface by `gfem.shape.boundary_morph`, so the chain
is t -> node coordinates (the morph's linear map, differentiated by JAX)
-> the solve (`gfem.solve`'s node-coordinate cotangent from the transport
tensors and the surface ``DiffShape`` terms). Checked against fourth-order
central differences along random unit directions in t, and by the Taylor
remainder.

- Electrostatics: a dielectric interface of a two-layer capacitor moves
  (capacitance and a potential functional).
- Driven: the interface of a two-layer line between lumped ports moves,
  with a conductor top plate, second-order absorbing side walls and a
  lossy, conducting layer (uniform sweep, direct and BDDC, and an
  adaptive sweep with frozen samples).
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
from gfem.shape import boundary_morph

STEP = 2e-6  # metres along a unit direction in t (layers are 0.5 mm)


@pytest.fixture(scope="module")
def capacitor(tmp_path_factory):
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    layers = (("oxide", 0.5), ("nitride", 0.5))
    sm.write_msh22(path, sm.plate(2.0, 2.0, layers, (3, 3, 2)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={
            "oxide": gfem.Material(eps_r=3.9),
            "nitride": gfem.Material(eps_r=(7.5, 6.0, 5.0)),
        },
    )


ELECTROSTATIC = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


def _capacitor_objective(capacitor, solver, *, potential=False):
    morph = boundary_morph(capacitor, "oxide__nitride", grows="oxide")
    options = gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12)
    weights = np.random.default_rng(4).uniform(0.5, 1.5, 200)

    def objective(t):
        model = capacitor.replace(mesh=morph(t))
        result = gfem.solve(model, ELECTROSTATIC, options=options)
        value = result.capacitance[0, 0] * 1e12  # pF
        if potential:
            phi = result.potential["top"].coefficients
            value = value + jnp.sum(weights[: phi.shape[0]] * phi[:200])
        return value

    return objective, morph


@pytest.mark.parametrize("potential", [False, True], ids=["capacitance", "potential"])
@pytest.mark.parametrize("solver", ["direct", "bddc_gmres"])
def test_electrostatic_shape_gradient_matches_fd(capacitor, solver, potential):
    objective, morph = _capacitor_objective(capacitor, solver, potential=potential)
    t = jnp.zeros(morph.n_params)
    rng = np.random.default_rng(0)
    assert_gradient_matches(objective, t, unit_directions(rng, t.shape, 3), step=STEP)


def test_electrostatic_shape_gradient_taylor(capacitor):
    objective, morph = _capacitor_objective(capacitor, "bddc_gmres")
    t = jnp.full(morph.n_params, 1e-5)
    (d,) = unit_directions(np.random.default_rng(1), t.shape, 1)
    assert_taylor_second_order(objective, t, d, [4e-5, 2e-5, 1e-5, 5e-6])


def test_shape_gradient_under_jit_and_vmap(capacitor):
    objective, morph = _capacitor_objective(capacitor, "direct")
    rng = np.random.default_rng(6)
    batch = jnp.asarray(rng.uniform(-1e-5, 1e-5, (2, morph.n_params)))
    loop = np.stack([np.asarray(jax.grad(objective)(t)) for t in batch])
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(batch[0]), loop[0], rtol=1e-9
    )
    np.testing.assert_allclose(jax.vmap(jax.grad(objective))(batch), loop, rtol=1e-9)


LAYERS = (("sub", 0.5), ("cover", 0.5))


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1), layers=LAYERS))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    n = len(mesh.elements("sub"))
    eps = np.random.default_rng(3).uniform(2.0, 6.0, n) * (1 - 0.02j)
    return gfem.Model(
        mesh,
        materials={
            "sub": gfem.MaterialField(eps_r=eps, conductivity=np.full(n, 0.5)),
            "cover": gfem.Material(eps_r=(2.0, 2.5, 3.0), mu_r=1.3),
        },
        boundaries=[
            gfem.PEC("bottom"),
            gfem.Conductor("top", conductivity=5e7, thickness=2e-6),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 11))
WEIGHTS = np.array([[1.0, 0.4], [-0.6, 1.5]])


def _line_objective(line, problem, solver, rows=None):
    morph = boundary_morph(line, "sub__cover", grows="sub")
    options = gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12)

    def objective(t):
        model = line.replace(mesh=morph(t))
        s = gfem.solve(model, problem, options=options).s.values
        if rows is not None:
            s = s[jnp.asarray(rows)]
        return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) + jnp.sum(jnp.real(s[:, 1, 0]))

    return objective, morph


@pytest.mark.parametrize("solver", ["direct", "bddc_gmres"])
def test_driven_shape_gradient_matches_fd(line, solver):
    problem = gfem.Driven(frequency=(GRID[2], GRID[8]))
    objective, morph = _line_objective(line, problem, solver)
    t = jnp.zeros(morph.n_params)
    rng = np.random.default_rng(0)
    assert_gradient_matches(objective, t, unit_directions(rng, t.shape, 3), step=STEP)


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


def test_driven_adaptive_shape_gradient_matches_fd(line):
    samples = (GRID[0], GRID[10], GRID[5])
    problem = gfem.Driven(frequency=GRID, sweep="adaptive", samples=samples)
    # at the samples the full solution lies in the frozen basis, so the
    # frozen-basis gradient is the full-order one there
    rows = [GRID.index(f) for f in samples]
    objective, morph = _line_objective(line, problem, "direct", rows)
    t = jnp.zeros(morph.n_params)
    rng = np.random.default_rng(1)
    assert_gradient_matches(objective, t, unit_directions(rng, t.shape, 3), step=STEP)


def test_driven_shape_gradient_taylor(line):
    problem = gfem.Driven(frequency=(GRID[2], GRID[8]))
    objective, morph = _line_objective(line, problem, "bddc_gmres")
    t = jnp.full(morph.n_params, 1e-5)
    (d,) = unit_directions(np.random.default_rng(2), t.shape, 1)
    assert_taylor_second_order(objective, t, d, [4e-5, 2e-5, 1e-5, 5e-6])


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


# --- wave ports ---------------------------------------------------------------

WAVE_LAYERS = (("lo", 0.5), ("hi", 0.5))
WAVE_PORTS = {
    "lossless": (gfem.Material(eps_r=4.0), (0.0, 0.0)),
    "lossy": (gfem.Material(eps_r=4.0, tan_delta=0.05), (0.3e-3, 0.1e-3)),
}


@pytest.fixture(scope="module")
def wave_mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("wave") / "line.msh"
    sm.write_msh22(
        path,
        sm.strip_line(4.0, 2.0, 1.0, (8, 2, 2), layers=WAVE_LAYERS, core=(1.0, 3.0)),
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _wave_line(mesh, port):
    lower, (d1, d2) = WAVE_PORTS[port]
    return gfem.Model(
        mesh,
        materials={
            "lo": lower,
            "hi": gfem.Material(eps_r=2.0),
            "core": gfem.Material(eps_r=3.0),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.WavePort("p1", "port1", offset=d1),
            gfem.WavePort("p2", "port2", offset=d2),
        ],
    )


@pytest.mark.parametrize("port", ["lossless", "lossy"])
def test_wave_port_shape_gradient_matches_fd(wave_mesh, port):
    # the top plate moves; its nodes on the port faces are held, so the port
    # modes stay those of the imported faces
    line = _wave_line(wave_mesh, port)
    morph = boundary_morph(line, "top")
    problem = gfem.Driven(frequency=(12e9, 24e9))
    options = gfem.SolverOptions(order=2, linear_solver="direct")

    def objective(t):
        s = gfem.solve(line.replace(mesh=morph(t)), problem, options=options).s.values
        return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) + jnp.sum(jnp.real(s[:, 1, 0]))

    t = jnp.zeros(morph.n_params)
    rng = np.random.default_rng(8)
    assert_gradient_matches(objective, t, unit_directions(rng, t.shape, 2), step=STEP)


def test_moved_wave_port_nodes_raise_port_geometry_error(wave_mesh):
    line = _wave_line(wave_mesh, "lossless")
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
