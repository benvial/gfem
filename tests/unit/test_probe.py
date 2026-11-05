"""Point probes: `Field.at` evaluates a result field at points."""

import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm
from near_faces import interior_faces, into

import gfem
from gfem._engine import fields as engine_fields
from gfem._engine.probe import Located, locate
from gfem._engine.probe import _settle as settle
from gfem._engine.probe import _tets_of_nodes as tets_of_nodes
from gfem._probe import evaluate, layout

LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (2, 2, 1)))
    return gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)


@pytest.fixture(scope="module")
def cap(stack):
    problem = gfem.Electrostatic(
        terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
    )
    return gfem.solve(stack, problem, options=gfem.SolverOptions(order=2))


def _random_points(n, z_lo=0.0, z_hi=0.75, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform([0, 0, z_lo], [4.0, 3.0, z_hi], size=(n, 3))
    return p * 1e-6  # m


def test_probe_reads_the_exact_potential_at_points(cap):
    # the potentials are piecewise linear in z: exact in the discrete space
    p = _random_points(40)
    z = p[:, 2] * 1e6
    v = cap.potential["mid"].at(p)
    assert v.shape == (40,)
    expected = np.where(z <= 0.5, z / 0.5, (0.75 - z) / 0.25)
    np.testing.assert_allclose(v, expected, atol=1e-12)


def test_probe_of_one_point_is_a_scalar(cap):
    v = cap.potential["top"].at([2e-6, 1e-6, 0.625e-6])
    assert v.shape == ()
    np.testing.assert_allclose(v, 0.5, atol=1e-12)


# --- vector fields ---------------------------------------------------------------

L, W, H = 10.0, 2.0, 1.0  # mm
EPS = 4.0


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (20, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    )


@pytest.fixture(scope="module")
def mode(line):
    return gfem.solve(line, gfem.Eigenmode(target=8e9)).e_field(0)


def test_probe_of_a_mode_reads_its_cosine_profile_in_volts_per_metre(mode):
    # E_z = A cos(pi x / L), uniform across the line, 1 J of electric energy
    amplitude = np.sqrt(4 / (const.epsilon_0 * EPS * W * H * L * 1e-9))
    x = np.linspace(0.3, 9.7, 9)
    p = np.stack([x, np.full_like(x, 0.7), np.full_like(x, 0.4)], axis=1) * 1e-3
    e = np.asarray(mode.at(p))
    assert e.shape == (9, 3)
    np.testing.assert_allclose(e[:, :2], 0.0, atol=1e-2 * amplitude)
    sign = np.sign(e[0, 2].real)
    np.testing.assert_allclose(
        sign * e[:, 2], amplitude * np.cos(np.pi * x / L), atol=5e-3 * amplitude
    )
    # uniform across the line: same value at another (y, z)
    q = p.copy()
    q[:, 1:] = [1.9e-3, 0.95e-3]
    np.testing.assert_allclose(mode.at(q)[:, 2], e[:, 2], atol=5e-3 * amplitude)


def test_probe_of_b_reads_the_magnetic_flux_density(mode):
    b = mode.b().at([[2.5e-3, 1e-3, 0.5e-3]])
    assert b.shape == (1, 3)
    # B_y only: the curl of E_z(x) z-hat
    assert abs(b[0, 1]) > 1e3 * max(abs(b[0, 0]), abs(b[0, 2]))


# --- where a field is not defined ------------------------------------------------


def test_probe_outside_the_mesh_raises(cap):
    with pytest.raises(ValueError, match="outside the mesh"):
        cap.potential["top"].at([[2e-6, 1e-6, 2e-6]])


def test_probe_outside_the_support_names_the_region(cap):
    # the "mid" potential restricted to the oxide: the nitride is outside
    field = cap.potential["mid"]
    oxide = gfem.Field(**{**_kwargs(field), "support": ("oxide",)})
    with pytest.raises(ValueError, match="'nitride'"):
        oxide.at([[2e-6, 1e-6, 0.7e-6]])


def test_probe_inside_an_excluded_volume_raises(tmp_path):
    path = tmp_path / "plate.msh"
    layers = (("metal", 0.5), ("oxide", 0.5))
    sm.write_msh22(path, sm.plate(4.0, 3.0, layers, (1, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(
        mesh,
        materials={"oxide": gfem.Material(eps_r=3.9)},
        excluded=("metal",),
    )
    problem = gfem.Electrostatic(terminals={"t": "top"}, ground="metal__oxide")
    field = gfem.solve(model, problem).potential["t"]
    np.testing.assert_allclose(field.at([2e-6, 1e-6, 0.75e-6]), 0.5, atol=1e-12)
    with pytest.raises(ValueError, match="'metal'"):
        field.at([2e-6, 1e-6, 0.25e-6])


# --- which tetrahedron holds a point ----------------------------------------------


@pytest.fixture(scope="module")
def rough(tmp_path_factory):
    """A field whose normal component jumps at every face: random coefficients."""
    path = tmp_path_factory.mktemp("rough") / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (6, 4, 4)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    support = ("dielectric",)
    width, _ = engine_fields.dof_shape(mesh, support, (), "hcurl", 2)
    rng = np.random.default_rng(3)
    return gfem.Field(
        coefficients=jnp.asarray(rng.standard_normal(width) + 0j),
        mesh=mesh,
        quantity="E",
        unit="V/m",
        space="hcurl",
        order=2,
        support=support,
        omega=jnp.asarray(1.0),
    )


@pytest.mark.parametrize("delta", [1e-4, 7e-5, 1e-6, 1e-8])
def test_a_point_inside_a_tet_gets_that_tet_in_netgens_band(rough, delta):
    # netgen takes a tet up to 1e-4 outside the point, the neighbour half the time
    pairs, faces = interior_faces(rough.mesh)
    tets = np.concatenate([pairs[:, 0], pairs[:, 1]])
    points, bary = into(rough.mesh, tets, np.concatenate([faces, faces]), delta)
    np.testing.assert_array_equal(
        locate(rough.mesh, layout(rough), points).tetrahedra, tets
    )
    own = evaluate(
        rough.coefficients[None], rough.mesh, layout(rough), Located(tets, bary)
    )[0]
    np.testing.assert_allclose(rough.at(points), own, rtol=1e-10)


def test_a_point_on_a_face_gets_the_largest_tet_holding_it(rough):
    pairs, faces = interior_faces(rough.mesh)
    points, _ = into(rough.mesh, pairs[:, 0], faces, 0.0)
    found = locate(rough.mesh, layout(rough), points)
    np.testing.assert_array_equal(found.tetrahedra, pairs.max(axis=1))


@pytest.mark.parametrize("delta", [3e-11, -3e-11, 5e-10, -5e-10])
def test_a_point_at_round_off_from_a_face_counts_as_on_it(rough, delta):
    # round-off on a sliver reaches 1e-11: within 1e-9 of a face is on it
    pairs, faces = interior_faces(rough.mesh)
    points, _ = into(rough.mesh, pairs[:, 0], faces, delta)
    found = locate(rough.mesh, layout(rough), points)
    np.testing.assert_array_equal(found.tetrahedra, pairs.max(axis=1))


def test_the_node_to_tet_map_is_built_once_per_mesh_topology(rough):
    moved = gfem.Mesh(np.asarray(rough.mesh.nodes) * 1.0, rough.mesh._topology)
    assert tets_of_nodes(moved) is tets_of_nodes(rough.mesh)


class _Topology:
    """A stand-in for a mesh's topology: `_settle` reads its tetrahedra."""


def test_a_flat_tet_beside_a_point_is_skipped():
    # tets 0 and 1 share the face z = 0 under x + y <= 1; tet 2 is flat, in z = 0
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 0]], float
    )
    topology = _Topology()
    topology.tetrahedra = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [1, 2, 5, 0]])
    mesh = types.SimpleNamespace(
        nodes=nodes, tetrahedra=topology.tetrahedra, _topology=topology
    )
    point = np.array([[0.25, 0.25, 0.0]])
    support = np.ones(3, bool)
    tets, bary = settle(mesh, support, np.array([0]), point)
    np.testing.assert_array_equal(tets, [1])
    np.testing.assert_allclose(bary @ nodes[[0, 1, 2, 4]], point, atol=1e-15)


@pytest.fixture(scope="module")
def beside_metal(tmp_path_factory):
    """``V = (z - 0.5) / 0.5`` (z in um) above an excluded metal volume."""
    path = tmp_path_factory.mktemp("beside") / "plate.msh"
    sm.write_msh22(
        path, sm.plate(4.0, 3.0, (("metal", 0.5), ("oxide", 0.5)), (3, 3, 1))
    )
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(
        mesh, materials={"oxide": gfem.Material(eps_r=3.9)}, excluded=("metal",)
    )
    problem = gfem.Electrostatic(terminals={"t": "top"}, ground="metal__oxide")
    return gfem.solve(model, problem).potential["t"]


def _metal_faces(mesh):
    """``(n, 2)`` tets ``(metal, oxide)`` across the interface, and its faces."""
    pairs, faces = interior_faces(mesh)
    in_metal = np.isin(np.arange(len(mesh.tetrahedra)), mesh.elements("metal"))[pairs]
    across = in_metal[:, 0] != in_metal[:, 1]
    pairs, faces, in_metal = pairs[across], faces[across], in_metal[across]
    metal = np.where(in_metal[:, 0], pairs[:, 0], pairs[:, 1])
    oxide = np.where(in_metal[:, 0], pairs[:, 1], pairs[:, 0])
    return np.stack([metal, oxide], axis=1), faces


@pytest.mark.parametrize("delta", [-1e-10, 0.0, 1e-12, 1e-10, 1e-6, 7e-5, 1e-4])
def test_a_point_on_or_just_above_an_excluded_conductor_has_a_value(
    beside_metal, delta
):
    mesh = beside_metal.mesh
    pairs, faces = _metal_faces(mesh)
    points, _ = into(mesh, pairs[:, 1], faces, delta)
    np.testing.assert_allclose(
        beside_metal.at(points), (points[:, 2] * 1e6 - 0.5) / 0.5, atol=1e-10
    )


@pytest.mark.parametrize("delta", [1e-8, 1e-6, 1e-4])
def test_a_point_just_inside_an_excluded_conductor_raises(beside_metal, delta):
    pairs, faces = _metal_faces(beside_metal.mesh)
    points, _ = into(beside_metal.mesh, pairs[:, 0], faces, delta)
    for point in points:
        with pytest.raises(ValueError, match="'metal'"):
            beside_metal.at(point)


def test_netgens_band_still_accepts_a_point_just_outside_the_mesh(beside_metal):
    # a cell is 0.5 um high: 1e-6 of it above the top plate, where V = 1
    just_above = [1.7e-6, 1.1e-6, (1.0 + 0.5e-6) * 1e-6]
    np.testing.assert_allclose(beside_metal.at(just_above), 1.0, atol=1e-4)
    with pytest.raises(ValueError, match="outside the mesh"):
        beside_metal.at([1.7e-6, 1.1e-6, (1.0 + 0.5e-2) * 1e-6])


# --- JAX ---------------------------------------------------------------------------

IN_LINE = (
    np.array([[2.5, 1.0, 0.5], [7.5, 0.3, 0.2], [1.1, 1.7, 0.9], [5.2, 0.5, 0.6]])
    * 1e-3
)


def test_probe_backward_is_the_transpose_of_the_forward(mode):
    # the probe is linear in the coefficients: <vjp(w), c> = <w, at(c)>
    rng = np.random.default_rng(5)
    shape = mode.coefficients.shape
    c = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    w = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))

    def f(coefficients):
        return gfem.Field(**{**_kwargs(mode), "coefficients": coefficients}).at(IN_LINE)

    values, pullback = jax.vjp(f, jnp.asarray(c))
    (g,) = pullback(jnp.asarray(w))
    np.testing.assert_allclose(jnp.sum(g * c), jnp.sum(w * values), rtol=1e-12)


def test_probe_runs_under_jit_and_vmap(mode):
    p = jnp.asarray(IN_LINE)
    eager = mode.at(p)
    np.testing.assert_allclose(jax.jit(lambda f: f.at(p))(mode), eager, rtol=1e-14)
    # traced points: located inside the host callback
    np.testing.assert_allclose(jax.jit(mode.at)(p), eager, rtol=1e-14)
    stacked = jax.vmap(mode.at)(p[:, None, :])
    np.testing.assert_allclose(stacked[:, 0], eager, rtol=1e-14)


def test_probe_node_and_point_gradients_raise(mode):
    p = jnp.asarray(IN_LINE)
    topology = mode.mesh._topology

    def moved(nodes):
        field = gfem.Field(**{**_kwargs(mode), "mesh": gfem.Mesh(nodes, topology)})
        return jnp.abs(field.at(p)).sum()

    with pytest.raises(gfem.NonDifferentiableError, match="node coordinates"):
        jax.grad(moved)(mode.mesh.nodes)
    with pytest.raises(gfem.NonDifferentiableError, match="probe points"):
        jax.grad(lambda q: jnp.abs(mode.at(q)).sum())(p)
    # stop_gradient is the opt-out
    g = jax.grad(lambda q: jnp.abs(mode.at(jax.lax.stop_gradient(q))).sum())(p)
    assert not np.asarray(g).any()


def _kwargs(field):
    return {f.name: getattr(field, f.name) for f in dataclasses.fields(field)}
