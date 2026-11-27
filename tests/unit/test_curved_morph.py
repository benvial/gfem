"""Quality and `boundary_morph` on curved meshes.

Fixtures: the quarter coaxial line of ``tests/unit/data`` (radii 1 and 2 m,
length 1 m, mesh size 0.5; surfaces ``inner``, ``outer`` and ``sym``) at order
1 and 2, which have the same corners.
"""

import dataclasses
import math
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants

import gfem
from gfem import _pml
from gfem._curved import gated_triangles
from gfem.mesh import _TET_EDGES, _TRI_EDGES

DATA = Path(__file__).parent / "data"
EPS0 = scipy.constants.epsilon_0
EPS = np.finfo(float).eps
PROBLEM = gfem.Electrostatic(terminals={"in": "inner"}, ground="outer")


def _model(name, ports=()):
    mesh = gfem.Mesh.from_gmsh(DATA / f"{name}.msh", length_unit="m")
    return gfem.Model(mesh, materials={"fill": gfem.Material(eps_r=1.0)}, ports=ports)


@pytest.fixture(scope="module")
def curved():
    return _model("coax_order2")


@pytest.fixture(scope="module")
def straight():
    return _model("coax_order1")


@pytest.fixture(scope="module")
def morph(curved):
    return gfem.shape.boundary_morph(curved, "inner")


def _wall_nodes(mesh, name):
    """Corner and mid-edge nodes of the triangles of surface ``name``."""
    t = mesh._topology
    on = t.surface_tags == mesh.region(name).tag
    return np.unique(
        np.concatenate([t.surface_elements[on].ravel(), t.surface_midnodes[on].ravel()])
    )


# --- quality: a zero-length edge is bad, not NaN ------------------------------


def _one_straight_tetrahedron(mesh):
    t = mesh._topology
    empty = np.empty((0, 3), dtype=np.int64)
    alone = dataclasses.replace(
        t,
        volume_elements=t.volume_elements[:1],
        volume_tags=t.volume_tags[:1],
        volume_entities=t.volume_entities[:1],
        surface_elements=empty,
        surface_tags=empty[:, 0],
        surface_entities=empty[:, 0],
        surface_neighbours=np.empty((0, 2), dtype=np.int64),
        source=None,
    )
    return gfem.Mesh(mesh.nodes, alone)


def test_a_zero_length_edge_scores_zero_and_is_refused_by_with_nodes(straight):
    alone = _one_straight_tetrahedron(straight.mesh)
    nodes = np.array(alone.nodes)
    first, second = alone._topology.volume_elements[0, :2]
    nodes[second] = nodes[first]  # a collapsed edge: 0 / 0 in the scaled Jacobian
    q = np.asarray(gfem.shape.quality(gfem.Mesh(nodes, alone._topology)))
    assert not np.isnan(q).any()
    assert (q <= 0).all()
    with pytest.raises(gfem.MeshQualityError):
        alone.with_nodes(nodes)


# --- boundary_morph on a curved mesh -------------------------------------------


def test_a_curved_morph_has_the_corner_parameters_of_its_faceted_twin(straight, morph):
    # one parameter per corner node of the moving surface: the straight twin has
    # the same corners, so the same count and points
    twin = gfem.shape.boundary_morph(straight, "inner")
    assert morph.n_params == twin.n_params
    np.testing.assert_array_equal(morph.points, twin.points)
    assert morph.points.shape == (morph.n_params, 3)


def test_normals_of_a_curved_wall_are_radial_and_lie_in_the_end_caps(straight, morph):
    # The wall is a cylinder, so the exact normal at a node is radial and has
    # no z component. The faceted twin (size 0.5, wall arcs of 0.5 rad) tilts
    # its normals by up to 0.17 and out of the caps by 0.03: a chord facet is
    # tangent to the cylinder nowhere. The quadratic triangles follow the
    # circle to the error of a quadratic arc through three points; measured
    # 1.4e-3 (radial) and 6.4e-4 (z) at this size. The bounds are 5e-3 and
    # 2e-3: 3x the measure, and 35x and 15x below the faceted values.
    twin = gfem.shape.boundary_morph(straight, "inner")
    errors = []
    for m in (morph, twin):
        radial = np.stack([m.points[:, 0], m.points[:, 1], 0 * m.points[:, 2]], axis=1)
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        errors.append(
            (
                np.linalg.norm(m.normals - radial, axis=1).max(),
                np.abs(m.normals[:, 2]).max(),
            )
        )
    assert errors[0][0] < 5e-3
    assert errors[0][1] < 2e-3
    assert errors[1][0] > 0.1
    assert errors[1][1] > 2e-2


def test_a_uniform_offset_puts_corner_and_mid_edge_nodes_on_the_offset_wall(
    curved, morph
):
    # t = 0.05 m moves the inner wall (radius 1) to radius 1.05. A node moves by
    # t n with n a unit vector within delta of radial (delta < 5e-3 above), so
    # its new radius is sqrt((1 + t cos d)^2 + (t sin d)^2) = 1 + t + t d^2 / 2 +
    # ...: the error is second order, t delta^2 = 1.25e-6 (measured 5e-8). The
    # nodes start on the cylinder (gmsh places mid-edge nodes on it to 1e-16).
    # Along z a node moves by t n_z, |n_z| < 2e-3 above: the caps stay flat to 1e-4.
    t = 0.05
    nodes = np.asarray(morph(t).nodes)
    x0 = np.asarray(curved.mesh.imported_nodes)
    wall = _wall_nodes(curved.mesh, "inner")
    radius = np.hypot(nodes[wall, 0], nodes[wall, 1])
    np.testing.assert_allclose(radius, 1 + t, rtol=0, atol=t * 5e-3**2)
    assert np.abs(nodes[wall, 2] - x0[wall, 2]).max() < t * 2e-3


def test_unselected_and_interior_mid_edge_nodes_follow_their_chords(curved, morph):
    # Every mid-edge node off the moving wall moves by the mean of its corners'
    # displacements. Displacements are read from coordinates (x0 + u) - x0, an
    # error of eps |x| per coordinate: the tolerance is 8 eps X, X = max |x| = 2.
    t = np.linspace(0.0, 0.05, morph.n_params)
    moved = np.asarray(morph(t).nodes)
    u = moved - np.asarray(curved.mesh.imported_nodes)
    topology = curved.mesh._topology
    on_wall = np.zeros(len(u), dtype=bool)
    on_wall[_wall_nodes(curved.mesh, "inner")] = True
    for k, (a, b) in enumerate(_TET_EDGES):
        mid = topology.volume_midnodes[:, k]
        free = ~on_wall[mid]
        chord = (
            u[topology.volume_elements[:, a]] + u[topology.volume_elements[:, b]]
        ) / 2
        np.testing.assert_allclose(
            u[mid[free]], chord[free], rtol=0, atol=8 * EPS * 2.0
        )


def test_a_mid_edge_node_between_moving_corners_moves_along_its_own_normal(
    curved, morph
):
    # With a non-uniform t the mid node of a wall edge moves by exactly
    # (t_a + t_b) / 2 along a unit vector: its length is that mean to the
    # round-off of reading u from coordinates, 8 eps X (X = 2 m), and its
    # direction is radial to 5e-3 (the bound on the quadratic normals above).
    t = 0.01 + 0.04 * (morph.points[:, 2] > 0.5)
    u = np.asarray(morph(t).nodes) - np.asarray(curved.mesh.imported_nodes)
    topology = curved.mesh._topology
    on = topology.surface_tags == curved.mesh.region("inner").tag
    corners, mids = topology.surface_elements[on], topology.surface_midnodes[on]
    x0 = np.asarray(curved.mesh.imported_nodes)
    value = {tuple(np.round(p, 12)): v for p, v in zip(morph.points, t, strict=True)}

    def at(node):
        return value[tuple(np.round(x0[node], 12))]

    for k, (a, b) in enumerate(_TRI_EDGES):
        for ca, cb, mid in zip(corners[:, a], corners[:, b], mids[:, k], strict=True):
            mean = (at(ca) + at(cb)) / 2
            assert abs(np.linalg.norm(u[mid]) - mean) < 8 * EPS * 2.0
            radial = np.array([x0[mid, 0], x0[mid, 1], 0.0])
            radial /= np.linalg.norm(radial)
            assert np.linalg.norm(u[mid] / np.linalg.norm(u[mid]) - radial) < 5e-3


def test_the_gradient_of_a_uniform_offset_matches_differences_on_the_same_mesh(
    curved, morph
):
    # Capacitance of the offset wall. Central difference with step h = 1e-3 m:
    # evaluation noise ~ 1e-14 C / h = 1e-11 C, truncation h^2 |C'''| / 6 with
    # |C'''| / |C'| <= 10 for C ~ 1 / ln(2 / (1 + t)): 1.7e-6 C'. rtol 1e-5
    # leaves 6x on truncation.
    def capacitance(t):
        mesh = morph(t * jnp.ones(morph.n_params))
        return gfem.solve(curved.replace(mesh=mesh), PROBLEM).capacitance[0, 0]

    h = 1e-3
    fd = (capacitance(h) - capacitance(-h)) / (2 * h)
    g = jax.grad(capacitance)(0.0)
    np.testing.assert_allclose(float(g), float(fd), rtol=1e-5)


def test_the_curved_offset_gradient_follows_the_radius_derivative(
    curved, straight, morph
):
    # dC/da of C = pi eps0 L / (2 ln(b / a)) at a = 1, b = 2, L = 1 m is
    # C / (a ln(b / a)). The gradient differentiates the discrete capacitance
    # along the offset, so it carries the discretization error of a size 0.5
    # mesh at p = 2: measured 3e-4 on the curved twin, bounded here by 1e-3. The
    # faceted twin's wall has the wrong area: measured 3.4e-2, required above 2e-2.
    exact = EPS0 * (math.pi / 2) / math.log(2) / math.log(2)

    def error(model, m):
        def capacitance(t):
            mesh = m(t * jnp.ones(m.n_params))
            return gfem.solve(model.replace(mesh=mesh), PROBLEM).capacitance[0, 0]

        return abs(float(jax.grad(capacitance)(0.0)) - exact) / exact

    assert error(curved, morph) < 1e-3
    assert error(straight, gfem.shape.boundary_morph(straight, "inner")) > 2e-2


def test_jit_and_grad_through_a_curved_morph_agree_with_eager_and_differences(morph):
    # jit against eager: the same operations in another association order, so
    # an error of a few eps |x| per node: 64 eps X (X = 2) as for straight tests.
    t = jnp.linspace(0.0, 0.03, morph.n_params)
    eager = np.asarray(morph(t).nodes)
    jitted = np.asarray(jax.jit(lambda s: morph(s).nodes)(t))
    np.testing.assert_allclose(jitted, eager, rtol=0, atol=64 * EPS * 2.0)

    # A directional derivative of a node moment against a central difference of
    # step h = 1e-4 on the linear map: exact to round-off (nodes are linear in t),
    # so the error is the differencing noise 2 eps X / h = 9e-12.
    d = jnp.cos(jnp.arange(morph.n_params))

    def moment(s):
        return jnp.sum(morph(s).nodes[:, 0] ** 2)

    h = 1e-4
    fd = (moment(t + h * d) - moment(t - h * d)) / (2 * h)
    jvp = jax.jvp(moment, (t,), (d,))[1]
    np.testing.assert_allclose(float(jvp), float(fd), rtol=1e-6)
    np.testing.assert_allclose(
        float(jnp.vdot(jax.grad(moment)(t), d)), float(fd), rtol=1e-6
    )


def test_the_faces_of_a_sliding_wave_port_keep_their_mid_edge_nodes_on_their_chords():
    # The port face is the flat plane x = 0 of the coax; two of its edges lie on
    # the moving wall ``inner`` (the rim). Its mid-edge nodes follow their chords
    # even there, where the wall's own rule would move them along the wall's
    # normal. Tolerance 64 eps X, as above, X = 2 m the largest coordinate; the
    # displacement is O(0.02) so its chord mean carries a few eps of X as well.
    ports = [gfem.WavePort("p", "port")]
    model = _model("coax_split_order2", ports=ports)
    m = gfem.shape.boundary_morph(model, "inner", sliding_ports=["p"])
    t = 0.02 * (1 + np.sin(np.arange(m.n_params)))
    u = np.asarray(m(t).nodes) - np.asarray(model.mesh.imported_nodes)
    topology = model.mesh._topology
    on = topology.surface_tags == model.mesh.region("port").tag
    corners, mids = topology.surface_elements[on], topology.surface_midnodes[on]
    rim = np.isin(corners, _wall_nodes(model.mesh, "inner")).any()
    assert rim, "the port face must touch the moving wall for this check to bite"
    # the rim nodes really move, so the check compares nonzero displacements
    assert np.abs(u[corners]).max() > 1e-3
    for k, (a, b) in enumerate(_TRI_EDGES):
        chord = (u[corners[:, a]] + u[corners[:, b]]) / 2
        np.testing.assert_allclose(u[mids[:, k]], chord, rtol=0, atol=64 * EPS * 2.0)


def test_a_wave_port_on_a_face_with_curved_rim_arcs_refuses():
    # The ends of the coax are flat but their rims are circular arcs: the face's
    # mid-edge nodes leave their chords, so the gate refuses the port (the morph
    # rule above applies to straight faces only).
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        _model("coax_order2", ports=[gfem.WavePort("p", "sym")])
    assert caught.value.feature == "wave_port"


def test_the_mid_edge_nodes_of_a_held_surface_do_not_move(curved):
    m = gfem.shape.boundary_morph(curved, "inner", fixed=["outer"])
    u = np.asarray(m(0.03).nodes) - np.asarray(curved.mesh.imported_nodes)
    held = _wall_nodes(curved.mesh, "outer")
    assert np.abs(u[held]).max() == 0.0


# --- the layer check sees mid-edge nodes ----------------------------------------


def test_check_layer_nodes_counts_a_mid_edge_node_past_a_stretched_face(curved):
    # Corners all lie at x <= 2 (outer radius 2); the face is x = 2 and a mid-edge
    # node near it is pushed 0.1 m past it, its tet's corners untouched.
    mesh = curved.mesh
    topology = mesh._topology
    x = np.array(mesh.nodes)
    mids = topology.volume_midnodes.ravel()
    mid = int(mids[np.argmax(x[mids, 0])])
    x[mid, 0] = 2.1
    moved = gfem.Mesh(x, topology)
    layers = types.SimpleNamespace(
        elements=np.empty(0, dtype=np.int64),
        sides=(_pml.Side(axis=0, sign=1, face=2.0, thickness=1.0, regions=()),),
    )
    model = types.SimpleNamespace(
        _layers=layers,
        mesh=moved,
        _domain=types.SimpleNamespace(
            active=np.ones(len(topology.volume_elements), bool)
        ),
    )
    with pytest.raises(ValueError, match="past the inner face"):
        _pml.check_layer_nodes(model, gfem.Driven(np.array([1e9])))


def test_the_morph_keeps_straight_every_face_the_gate_reads_as_straight():
    # One list of gated faces: a port, a periodic pair, a sheet design and an
    # absorbing face beside an anisotropic material (flat faces of the cell).
    mesh = gfem.Mesh.from_gmsh(DATA / "cell_order2.msh", length_unit="m")
    air = gfem.Material()
    model = gfem.Model(
        mesh,
        materials={
            "ball": air,
            "air": air,
            "heater": air,
            "pml": gfem.Material(eps_r=(2.0, 2.0, 3.0)),
        },
        ports=[gfem.LumpedPort("l", "xlo", direction="+z")],
        boundaries=[
            gfem.Periodic(("ylo", "yhi")),
            gfem.Conductor(
                "sheet",
                5.8e7,
                fill=gfem.SheetField(fill=np.ones(len(mesh.elements("sheet")))),
            ),
            gfem.Absorbing("xhi"),
        ],
    )
    gated = gated_triangles(model)
    expected = np.zeros(len(mesh.triangles), bool)
    for name in ("xlo", "ylo", "yhi", "sheet", "xhi"):
        expected[mesh.elements(name)] = True
    np.testing.assert_array_equal(gated, expected)
    # faces no feature reads from corners are free to follow the wall
    assert not gated[mesh.elements("wall")].any()
    assert not gated[mesh.elements("zwall")].any()
