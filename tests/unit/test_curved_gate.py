"""The curved-mesh feature gate: which features run where their elements are straight.

Fixtures (tests/unit/data, made by make_curved_fixtures.py): ``cell_order2``, a
cube of side 3 m with a sphere of radius 0.3 m (curved wall), a core box, a
heater box and a shell ``pml``; only the elements touching the sphere are
curved, so every flat outer face, the heater and the shell are straight. Its
order-1 twin ``cell_order1`` shares the geometry. ``straight_box`` is a
second-order mesh of a straight-walled box whose mid nodes all sit on their
chord midpoints: a curved mesh with no curved element.
"""

import numpy as np
import pytest

import gfem
from gfem import _curved

DATA = __import__("pathlib").Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def cell():
    return gfem.Mesh.from_gmsh(DATA / "cell_order2.msh", length_unit="m")


@pytest.fixture(scope="module")
def cell_straight():
    return gfem.Mesh.from_gmsh(DATA / "cell_order1.msh", length_unit="m")


# --- the straight predicate -----------------------------------------------------


def test_only_the_elements_touching_the_sphere_are_curved(cell):
    tets = _curved.curved_tetrahedra(cell)
    tris = _curved.curved_triangles(cell)
    assert tets.shape == (len(cell.tetrahedra),)
    assert tris.shape == (len(cell.triangles),)
    in_region = {r: tets[cell.elements(r)] for r in cell.volumes}
    # the ball and the air around it hold the curved tetrahedra, the heater and
    # the shell none
    assert in_region["ball"].any()
    assert in_region["air"].any()
    assert not in_region["heater"].any()
    assert not in_region["pml"].any()
    # the sphere's triangles are curved, every flat face's are straight
    assert tris[cell.elements("wall")].all()
    for flat in ("xlo", "xhi", "ylo", "yhi", "zwall", "sheet"):
        assert not tris[cell.elements(flat)].any(), flat


def test_a_straight_mesh_has_no_curved_element(cell_straight):
    assert not _curved.curved_tetrahedra(cell_straight).any()
    assert not _curved.curved_triangles(cell_straight).any()


def test_the_predicate_is_scale_invariant(cell):
    # The tolerance is 64 eps times the largest coordinate, so reading the same
    # file in micrometres (nodes x 1e-6) leaves the curved set unchanged: both
    # the threshold and a real deviation (>= 1e-10 of an edge) scale together.
    small = gfem.Mesh.from_gmsh(DATA / "cell_order2.msh", length_unit="um")
    np.testing.assert_array_equal(
        _curved.curved_tetrahedra(small), _curved.curved_tetrahedra(cell)
    )


# --- the model's items ----------------------------------------------------------

AIR = gfem.Material()
UNIAXIAL = gfem.Material(eps_r=(2.0, 2.0, 3.0))
MATERIALS = dict.fromkeys(("ball", "air", "heater", "pml"), AIR)


def _model(mesh, **kw):
    kw.setdefault("materials", MATERIALS)
    return gfem.Model(mesh, **kw)


#: Items on the sphere (curved) and the feature each refuses with.
ON_THE_SPHERE = {
    "wave_port": {"ports": [gfem.WavePort("p", "wall")]},
    "lumped_port": {"ports": [gfem.LumpedPort("l", "wall", direction="+z")]},
    "floquet_port": {"ports": [gfem.FloquetPort("f", "wall", polarization="TE")]},
    "periodic": {"boundaries": [gfem.Periodic(("wall", "xhi"))]},
    "pml": {"pml": gfem.PML("ball")},
    "sheet_frame": {
        "boundaries": [
            gfem.Conductor("wall", 5.8e7, fill=gfem.SheetField(fill=np.ones(78)))
        ]
    },
}


@pytest.mark.parametrize("key", list(ON_THE_SPHERE))
def test_a_gated_item_on_a_curved_element_refuses_with_its_key(cell, key):
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        _model(cell, **ON_THE_SPHERE[key])
    assert caught.value.feature == key
    text = str(caught.value)
    assert "curved (second-order) mesh" in text
    assert "ElementOrder 1" in text
    assert "elements are curved" in text or "it is curved" in text
    assert " m." in text  # the first curved element is located, in metres


def test_the_refusal_is_a_not_implemented_error(cell):
    with pytest.raises(NotImplementedError):
        _model(cell, ports=[gfem.WavePort("p", "wall")])


def test_an_absorbing_face_beside_an_anisotropic_material_is_gated(cell):
    # Its admittance frame comes from chord normals. Only the curved face
    # beside the anisotropic material refuses; an isotropic neighbour, or a
    # flat face, does not.
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        _model(
            cell,
            materials={**MATERIALS, "ball": UNIAXIAL},
            boundaries=[gfem.Absorbing("wall")],
        )
    assert caught.value.feature == "sheet_frame"
    wall = [gfem.Absorbing("wall")]
    _curved.refuse_model(cell, wall, (), None, MATERIALS)
    _model(
        cell,
        materials={**MATERIALS, "pml": UNIAXIAL},
        boundaries=[gfem.Absorbing("xhi")],
    )


def test_a_message_says_when_every_tetrahedron_is_curved():
    mesh = gfem.Mesh.from_gmsh(DATA / "cell_order2.msh", length_unit="m")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_curved, "_all_curved", lambda _mesh: True)
        with pytest.raises(gfem.CurvedMeshUnsupportedError, match="HighOrderOptimize"):
            _model(mesh, ports=[gfem.WavePort("p", "wall")])


#: The same items on straight elements: flat outer faces and the shell.
ON_FLAT_FACES = {
    "wave_port": {"ports": [gfem.WavePort("p", "xlo")]},
    "lumped_port": {"ports": [gfem.LumpedPort("l", "xlo", direction="+z")]},
    "periodic": {"boundaries": [gfem.Periodic(("ylo", "yhi"))]},
    "pml": {"pml": gfem.PML("pml"), "background": AIR},
    "sheet_frame": {
        "boundaries": [
            gfem.Conductor("sheet", 5.8e7, fill=gfem.SheetField(fill=np.ones(66)))
        ]
    },
}


@pytest.mark.parametrize("key", list(ON_FLAT_FACES))
def test_a_gated_item_on_straight_elements_builds(cell, key):
    model = _model(cell, **ON_FLAT_FACES[key])
    assert model.mesh.curved


def test_a_floquet_port_on_a_flat_face_passes_the_gate(cell):
    # A Floquet port also needs a doubly periodic cell, which this fixture is
    # not (its z walls are one group), so the gate is called without the rest
    # of the model's checks.
    port = gfem.FloquetPort("f", "xlo", polarization="TE")
    _curved.refuse_model(cell, (), (port,), None, MATERIALS)


# --- a moved mesh -----------------------------------------------------------------


def _bend_port_face(mesh, shift=1e-3):
    """``mesh`` with one mid-edge node of an ``xlo`` triangle off its chord (m)."""
    nodes = np.array(mesh.nodes)
    tri = mesh.elements("xlo")[0]
    mid = mesh._topology.surface_midnodes[tri, 0]
    nodes[mid] += [shift, 0.0, 0.0]
    return mesh.with_nodes(nodes)


def test_a_morph_that_curves_a_port_face_raises_at_the_solve(cell):
    model = _model(cell, ports=[gfem.WavePort("p", "xlo")])
    moved = gfem.Model(
        _bend_port_face(cell), materials=MATERIALS, ports=[gfem.WavePort("p", "xlo")]
    )
    # the nodes moved after the item was gated on the imported nodes
    problem = gfem.Electrostatic(terminals={"w": "wall"}, ground="xhi")
    gfem.solve(model, problem)
    with pytest.raises(ValueError, match=r"moved mesh curves .*WavePort 'p'"):
        gfem.solve(moved, problem)


def test_a_straight_move_of_the_corners_and_their_mid_nodes_keeps_the_gate(cell):
    # an affine map keeps every mid node on its chord midpoint
    nodes = np.asarray(cell.nodes) * np.array([1.0, 1.0, 1.0]) + [0.01, 0.0, 0.0]
    moved = cell.with_nodes(nodes)
    model = gfem.Model(moved, materials=MATERIALS, ports=[gfem.WavePort("p", "xlo")])
    _curved.check_moved(model)


def _bend_heater(mesh, shift=1e-3):
    """``mesh`` with one mid-edge node of a ``heater`` tetrahedron off its chord."""
    nodes = np.array(mesh.nodes)
    tet = mesh.elements("heater")[0]
    nodes[mesh._topology.volume_midnodes[tet, 0]] += [shift, 0.0, 0.0]
    return mesh.with_nodes(nodes), tet


def test_a_morph_that_curves_a_heated_tetrahedron_refuses_the_power_input(cell):
    # the heater is straight as imported, so its power input passed the gate then
    moved, _ = _bend_heater(cell)
    _steady(cell, "heater", power=1e-3)
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        _steady(moved, "heater", power=1e-3)
    assert caught.value.feature == "heat_input"


def test_a_morph_that_curves_a_read_tetrahedron_refuses_the_point_read(cell):
    moved, tet = _bend_heater(cell)
    model = _model(moved)
    problem = gfem.Electrostatic(terminals={"w": "wall"}, ground="xhi")
    potential = gfem.solve(model, problem).potential["w"]
    point = _centroids(cell, [tet])
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        potential.at(point)
    assert caught.value.feature == "point_read"


# --- inputs of a solve -----------------------------------------------------------

THERMAL = gfem.Material(thermal_conductivity=1.0)
THERMAL_MATERIALS = dict.fromkeys(MATERIALS, THERMAL)
SINK = gfem.FixedTemperature("xlo", temperature=300.0)


def _steady(mesh, region, **heat):
    model = gfem.Model(mesh, materials=THERMAL_MATERIALS, boundaries=[SINK])
    source = gfem.HeatSource.uniform(region, **heat)
    return gfem.solve(model, gfem.SteadyThermal(source))


def test_a_power_heat_input_on_curved_elements_refuses(cell):
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        _steady(cell, "air", power=1e-3)
    assert caught.value.feature == "heat_input"
    assert "HeatSource" in str(caught.value)


def test_a_power_heat_input_on_straight_elements_runs(cell):
    result = _steady(cell, "heater", power=1e-3)
    assert np.isfinite(np.asarray(result.temperature.coefficients)).all()


def test_a_power_density_needs_no_numpy_volume_so_it_runs_on_curved_elements(cell):
    result = _steady(cell, "air", power_density=1.0)
    assert np.isfinite(np.asarray(result.temperature.coefficients)).all()


# --- point reads -------------------------------------------------------------------


@pytest.fixture(scope="module")
def potential(cell):
    model = _model(cell)
    problem = gfem.Electrostatic(terminals={"w": "wall"}, ground="xhi")
    return gfem.solve(model, problem).potential["w"]


def _centroids(mesh, tets):
    return mesh.imported_nodes[mesh.tetrahedra[tets]].mean(axis=1)


def test_a_point_in_a_curved_tetrahedron_refuses(cell, potential):
    curved = np.flatnonzero(_curved.curved_tetrahedra(cell))
    point = _centroids(cell, curved[:1])
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        potential.at(point)
    assert caught.value.feature == "point_read"
    assert "point" in str(caught.value)


def test_a_point_in_a_straight_tetrahedron_reads_the_field(cell, potential):
    straight = np.flatnonzero(~_curved.curved_tetrahedra(cell))
    point = _centroids(cell, straight[:1])
    assert np.isfinite(np.asarray(potential.at(point))).all()


# --- numpy-integrating entries ---------------------------------------------------

DIRECT = gfem.SolverOptions(order=1, linear_solver="direct")
#: In a straight air tetrahedron of ``cell``, 0.5 m east of the sphere's centre.
SOURCE = (1.7, 1.5, 1.5)
FREQUENCY = 1e8


@pytest.fixture(scope="module")
def radiating(cell):
    model = _model(cell, pml=gfem.PML("pml"), background=AIR)
    dipole = gfem.ElectricDipole("d", position=SOURCE, moment=(0.0, 0.0, 1e-9))
    problem = gfem.Driven(FREQUENCY, sources=dipole, keep_fields=True)
    return model, gfem.solve(model, problem, options=DIRECT)


def _around_source(radius):
    """A side: the elements within ``radius`` m of the source (centroids)."""

    def side(centroids):
        return ((centroids - np.array(SOURCE)) ** 2).sum(axis=1) < radius**2

    return side


THETA = np.linspace(0.3, 2.8, 3)
PHI = np.array([0.0, 1.0])[:, None]


def test_the_far_field_of_the_truncation_layer_runs_on_a_curved_mesh(radiating):
    # the layer touching the shell is straight (flat faces), whatever the ball does
    model, result = radiating
    ff = gfem.far_field(result, "d", THETA, PHI, model=model)
    assert np.isfinite(ff.e_theta).all()


def test_a_far_field_layer_with_a_curved_element_refuses(radiating):
    # radius 0.3 m: the layer reaches the sphere's neighbours
    model, result = radiating
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        gfem.far_field(result, "d", THETA, PHI, model=model, side=_around_source(0.3))
    assert caught.value.feature == "far_field"


BEAM = gfem.GaussianBeam(
    "fiber",
    waist=0.5,
    focus=(1.7, 1.5, 2.2),
    theta=np.pi - 0.2,
    phi=0.3,
    polarization="TM",
)


def test_beam_coupling_refuses_on_a_curved_layer_and_runs_on_a_straight_one(radiating):
    model, result = radiating
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        gfem.beam_coupling(result, "d", BEAM, model=model, side=_around_source(0.3))
    assert caught.value.feature == "beam_coupling"
    with pytest.warns(UserWarning, match="clips it"):  # the default side clips the beam
        coupling = gfem.beam_coupling(result, "d", BEAM, model=model)
    assert np.isfinite(np.asarray(coupling)).all()


def test_optical_force_refuses_any_curved_mesh(radiating):
    # even a body whose elements are all straight: the torque and the shape
    # pass read the whole shell
    model, result = radiating
    field = result.e_field(FREQUENCY, "d")
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        gfem.optical_force(field, "heater", model=model)
    assert caught.value.feature == "optical_force"
    assert "curved (second-order) mesh" in str(caught.value)


@pytest.fixture(scope="module")
def modes(cell):
    boundaries = [gfem.PEC(s) for s in ("xlo", "xhi", "ylo", "yhi", "zwall")]
    model = _model(cell, boundaries=boundaries)
    return gfem.solve(
        model, gfem.Eigenmode(target=FREQUENCY, n_modes=1), options=DIRECT
    )


def test_interface_participation_refuses_on_a_curved_owner_and_runs_on_a_straight_one(
    modes,
):
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        gfem.interface_participation(modes, "wall", 3e-9, 10.0)
    assert caught.value.feature == "interface_participation"
    on_flat = gfem.interface_participation(modes, "xlo", 3e-9, 10.0)
    assert np.isfinite(np.asarray(on_flat.participation)).all()


# --- incident fields -------------------------------------------------------------

PLANE = gfem.PlaneWave("te", theta=0.4, phi=0.3, polarization="TE")


def test_an_incident_field_on_a_curved_physical_element_refuses(cell):
    model = _model(cell, pml=gfem.PML("pml"), background=AIR)
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        gfem.solve(model, gfem.Driven(FREQUENCY, sources=PLANE), options=DIRECT)
    assert caught.value.feature == "incident_field"
    assert "'te'" in str(caught.value)


@pytest.fixture(scope="module")
def straight_box(tmp_path_factory):
    """A second-order mesh of a straight box: curved format, no curved element."""
    gmsh = pytest.importorskip("gmsh")
    import pml_cases

    tmp = tmp_path_factory.mktemp("straight_box")
    first = pml_cases.box(
        tmp / "first.msh", ("-x", "+x", "-y", "+y", "-z", "+z"), thickness=0.25, cells=4
    )
    out = tmp / "second.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(first))
        gmsh.model.mesh.setOrder(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(out))
    finally:
        gmsh.finalize()
    return gfem.Mesh.from_gmsh(out)


def test_an_incident_field_runs_when_every_physical_element_is_straight(straight_box):
    import pml_cases

    assert straight_box.curved
    assert not _curved.curved_tetrahedra(straight_box).any()
    model = gfem.Model(
        straight_box,
        materials={"air": AIR, "pml": AIR},
        pml=gfem.PML("pml"),
        background=AIR,
    )
    result = gfem.solve(model, gfem.Driven(pml_cases.F0, sources=PLANE), options=DIRECT)
    assert result is not None


def _second_order(first, out):
    """Re-write a first-order gmsh file at order 2: mid nodes on the chords."""
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(first))
        gmsh.model.mesh.setOrder(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(out))
    finally:
        gmsh.finalize()
    return out


@pytest.fixture(scope="module")
def cube_scattering(tmp_path_factory):
    """A solved plane-wave column on a dielectric cube in a layer, order 2.

    The mesh is straight (the incident gate lets the solve run) but curved in
    format; the cube is the scatterer of `current_moments`.
    """
    import multipole_cases as mc

    tmp = tmp_path_factory.mktemp("cube_scattering")
    first = mc.cube(tmp / "first.msh", centred=True)
    model = mc.portless_scatterer_model(_second_order(first, tmp / "second.msh"))
    assert model.mesh.curved
    problem = gfem.Driven(299792458.0 / 1e-6, sources=PLANE, keep_fields=True)
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    return model, gfem.solve(model, problem, options=options)


@pytest.mark.parametrize(
    ("key", "call"),
    [
        ("multipoles", lambda r, m: gfem.multipoles(r, "te", model=m)),
        ("current_moments", lambda r, m: gfem.current_moments(r, "te", model=m)),
        ("cross_sections", lambda r, m: gfem.cross_sections(r, "te", model=m)),
    ],
)
def test_the_incident_field_measurements_refuse_on_curved_elements(
    cube_scattering, monkeypatch, key, call
):
    # The incident gate stops a solve on a curved physical element, so a result
    # that reaches these entries sits on a straight mesh. Mark every element
    # curved after the solve (the predicate is the gate's only input) to reach
    # the measurement's own gate: the truncation layer or the scatterer.
    model, result = cube_scattering
    mesh = model.mesh
    n_tet, n_tri = len(mesh.tetrahedra), len(mesh.triangles)
    monkeypatch.setattr(
        _curved,
        "_bent_edges",
        lambda *_a, **_k: (np.ones((n_tet, 6), bool), np.ones((n_tri, 3), bool)),
    )
    with pytest.raises(gfem.CurvedMeshUnsupportedError) as caught:
        call(result, model)
    assert caught.value.feature == key


# --- the table --------------------------------------------------------------------


def _public_functions():
    import inspect

    return sorted(n for n in gfem.__all__ if inspect.isfunction(getattr(gfem, n)))


def test_every_public_function_is_classified():
    assert set(_public_functions()) == set(_curved.ENTRIES)


def test_every_classification_names_a_known_status_and_key():
    upstream = {"2d_mesh", "cross_section"}
    for name, (status, key) in _curved.ENTRIES.items():
        assert status in (_curved.ACCEPTED, _curved.GATED, _curved.REFUSED), name
        if status == _curved.ACCEPTED:
            assert key is None, name
        elif status == _curved.GATED:
            assert key in _curved.FEATURES, name
        else:
            assert key in upstream, name


def test_every_gated_feature_has_an_entry_or_an_item():
    used = {k for _, k in _curved.ENTRIES.values() if k} | set(_curved.ITEMS.values())
    assert set(_curved.FEATURES) <= used


def test_the_items_name_public_classes_and_features():
    for cls, key in _curved.ITEMS.items():
        assert cls in gfem.__all__
        assert key in _curved.FEATURES


def test_every_feature_has_a_distinct_docs_label_and_a_remedy():
    labels = [f.label for f in _curved.FEATURES.values()]
    assert all(labels)
    assert len(set(labels)) == len(labels)
    assert all(f.reason and f.remedy for f in _curved.FEATURES.values())


def test_every_key_raised_in_the_package_is_in_the_table():
    import re

    source = DATA.parents[2] / "src" / "gfem"
    raised = set()
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        raised |= set(re.findall(r"CurvedMeshUnsupportedError\(\s*\"(\w+)\"", text))
    assert raised <= set(_curved.FEATURES) | {"2d_mesh", "cross_section"}
