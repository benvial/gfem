"""The tutorials' committed meshes are what their generator writes.

docs/tutorials/data/*.msh are read by the tutorials, the get-started page and
several drift tests, and tools/make_tutorial_meshes.py is the only description
of how they were built. This regenerates them (no gmsh, about two seconds) and
compares bytes, so neither can change without the other. The transmon meshes
are gmsh-built by tools/transmon_mesh.py and pinned by test_transmon_mesh.py;
the metagrating meshes are gmsh-built by tools/metagrating_mesh.py
(`just metagrating-mesh`); the heater meshes (and the thermo-optic recipe's extruded
heater_straight.msh) are gmsh-built by tools/heater_section_mesh.py
(`just heater-section-mesh`); sphere.msh is netgen-built by tools/sphere_mesh.py
(`just sphere-mesh`) and nanosphere.msh by tools/nanosphere_mesh.py
(`just nanosphere-mesh`), meta_atom.msh by tools/meta_atom_mesh.py
(`just meta-atom-mesh`); grating.msh is gmsh-built by tools/grating_mesh.py
(`just grating-mesh`) and pinned by test_grating_mesh.py; the photonic-crystal
meshes are gmsh-built by tools/rods_mesh.py (`just rods-mesh`); wr90_guide.msh
is a copy of the gmsh-built tests/unit/data/waveguide.msh
(tools/make_port_fixtures.py), pinned below.
"""

from pathlib import Path

import grating_mesh
import heater_section_mesh
import make_tutorial_meshes
import meshio
import metagrating_mesh
import numpy as np
import periodic_cases as pc
import pytest
import rods_mesh
import transmon_mesh

import gfem

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
GMSH_BUILT = (
    transmon_mesh.MESHES
    | metagrating_mesh.MESHES
    | heater_section_mesh.MESHES
    | heater_section_mesh.STRAIGHT
    | grating_mesh.MESHES
    | rods_mesh.MESHES
    | {"sphere.msh": None, "nanosphere.msh": None, "meta_atom.msh": None}
    | {"wr90_guide.msh": None}
)
COMMITTED = sorted(
    path.name for path in DATA.glob("*.msh") if path.name not in GMSH_BUILT
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("tutorial-meshes")
    return {path.name: path for path in make_tutorial_meshes.write(out)}


def test_the_generator_writes_every_committed_mesh(generated):
    assert sorted(generated) == COMMITTED


@pytest.mark.parametrize("name", COMMITTED)
def test_a_committed_mesh_is_the_generators_output(generated, name):
    assert generated[name].read_bytes() == (DATA / name).read_bytes(), (
        f"docs/tutorials/data/{name} differs from what tools/make_tutorial_meshes.py "
        "writes: rerun `python tools/make_tutorial_meshes.py docs/tutorials/data`"
    )


def test_the_wr90_guide_is_the_port_fixture():
    fixture = Path(__file__).parent / "data" / "waveguide.msh"
    assert (DATA / "wr90_guide.msh").read_bytes() == fixture.read_bytes()


# each triangle-only mesh is the z = 0 face of a slab mesh (one cell thick),
# with its slab's volume names on the areas and its side-face names on the lines
SECTIONS = {
    "bend2d.msh": ("bend.msh", {"port1", "port2", "outer"}),
    "bend2d_7um.msh": ("bend_7um.msh", {"port1", "port2", "outer"}),
    "line2d.msh": ("line.msh", {"port1", "port2", "side"}),
}


def _named(mesh, kind):
    """Cells of ``kind`` and the physical name of each."""
    names = {int(tag): name for name, (tag, _) in mesh.field_data.items()}
    cells = [b.data for b in mesh.cells if b.type == kind]
    tags = [
        t
        for b, t in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True)
        if b.type == kind
    ]
    if not cells:
        return np.empty((0, 0), int), []
    return np.concatenate(cells), [names[int(t)] for t in np.concatenate(tags)]


def _slab_face(slab):
    """The slab's z = 0 nodes, and its triangles there named by their tetrahedra."""
    bottom = np.flatnonzero(slab.points[:, 2] == 0.0)
    tets, tet_names = _named(slab, "tetra")
    faces = {}
    for tet, name in zip(tets, tet_names, strict=True):
        on = [v for v in tet if slab.points[v, 2] == 0.0]
        if len(on) == 3:
            faces[frozenset(on)] = name
    return bottom, faces


def _boundary_edges(tris):
    count = {}
    for tri in tris:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = frozenset((int(tri[a]), int(tri[b])))
            count[key] = count.get(key, 0) + 1
    return {e for e, n in count.items() if n == 1}


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_a_triangle_mesh_is_written(generated, name):
    assert name in generated


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_a_triangle_mesh_is_its_slabs_z0_face_node_for_node(generated, name):
    slab_name, line_names = SECTIONS[name]
    flat = meshio.read(generated[name])
    slab = meshio.read(DATA / slab_name)
    bottom, faces = _slab_face(slab)
    assert {b.type for b in flat.cells} == {"triangle", "line"}
    np.testing.assert_array_equal(flat.points, slab.points[bottom])
    tris, tri_names = _named(flat, "triangle")
    as_slab = {
        frozenset(int(bottom[v]) for v in tri): n
        for tri, n in zip(tris, tri_names, strict=True)
    }
    assert len(as_slab) == len(tris)
    assert as_slab == faces
    # counterclockwise in the xy plane
    p = flat.points[tris][:, :, :2]
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    assert (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0] > 0).all()
    lines, seg_names = _named(flat, "line")
    assert {frozenset(map(int, s)) for s in lines} == _boundary_edges(tris)
    assert len(lines) == len(_boundary_edges(tris))
    assert set(seg_names) == line_names
    assert {n: d for n, (_, d) in flat.field_data.items()} == (
        dict.fromkeys(tri_names, 2) | dict.fromkeys(line_names, 1)
    )


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_a_lines_name_is_its_slab_side_face(generated, name):
    """Each segment carries the name of the slab side face standing on it."""
    slab_name, _ = SECTIONS[name]
    flat = meshio.read(generated[name])
    slab = meshio.read(DATA / slab_name)
    bottom = np.flatnonzero(slab.points[:, 2] == 0.0)
    sides = {}
    tris, tri_names = _named(slab, "triangle")
    for tri, n in zip(tris, tri_names, strict=True):
        on = frozenset(int(v) for v in tri if slab.points[v, 2] == 0.0)
        if len(on) == 2:
            sides[on] = n
    lines, seg_names = _named(flat, "line")
    for seg, n in zip(lines, seg_names, strict=True):
        assert sides[frozenset(int(bottom[v]) for v in seg)] == n


def test_the_bend_names():
    flat = meshio.read(DATA / "bend2d.msh")
    assert {n: d for n, (_, d) in flat.field_data.items()} == {
        "design": 2,
        "core": 2,
        "cladding": 2,
        "port1": 1,
        "port2": 1,
        "outer": 1,
    }


def test_the_line_names():
    flat = meshio.read(DATA / "line2d.msh")
    assert {n: d for n, (_, d) in flat.field_data.items()} == {
        "feed": 2,
        "design": 2,
        "port1": 1,
        "port2": 1,
        "side": 1,
    }


@pytest.mark.parametrize("name", sorted(rods_mesh.MESHES))
def test_the_rods_tutorial_mesh_has_the_disk_area(name):
    """The polygon rod has the disk's area, and the cell pairs on x and y."""
    mesh = pc.rods_tutorial_mesh(name)
    nodes = np.asarray(mesh.imported_nodes)
    p = nodes[mesh.volume_elements[mesh.elements("rod")]]
    volume = np.abs(np.linalg.det(p[:, 1:] - p[:, :1])).sum() / 6
    r = pc.ROD_R * pc.MM
    assert volume == pytest.approx(np.pi * r**2 * mesh.section_thickness, rel=1e-12)
    model = gfem.Model(
        mesh,
        materials={"rod": gfem.Material(eps_r=pc.ROD_EPS), "air": gfem.Material()},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
    )
    a = pc.ROD_A * pc.MM
    np.testing.assert_allclose(
        model.periodic_translations, [(a, 0.0, 0.0), (0.0, a, 0.0)], atol=1e-15
    )
