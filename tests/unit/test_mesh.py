"""`gfem.Mesh.from_gmsh`: regions, units, adjacency, pytree, export.

The fixtures under tests/unit/data were written by gmsh 4.15 with
tools/make_mesh_fixtures.py: a 2 x 2 x 1 um substrate "sub" (tag 1) holding a
1 x 1 x 0.4 um conductor box that is also tagged "sub" (gsim tags meshed
conductors with a dielectric name), under a 2 x 2 x 1 um "air" box (tag 2).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

import gfem

DATA = Path(__file__).parent / "data"
TWO_BOX = {
    "msh22": DATA / "two_box_msh22.msh",
    "msh41": DATA / "two_box_msh41.msh",
}
VOLUMES = {"sub": 1, "air": 2}
SURFACES = {"metal_xy": 4, "metal_z": 5, "air__sub": 7, "sub__None": 8, "air__None": 9}


@pytest.fixture(params=sorted(TWO_BOX))
def two_box(request):
    return gfem.Mesh.from_gmsh(TWO_BOX[request.param], length_unit="um")


def test_regions_keep_gmsh_physical_names_and_numbers(two_box):
    assert {r.name: r.tag for r in two_box.regions if r.dim == 3} == VOLUMES
    assert {r.name: r.tag for r in two_box.regions if r.dim == 2} == SURFACES
    assert two_box.volumes == ("sub", "air")
    assert two_box.surfaces == (
        "metal_xy",
        "metal_z",
        "air__sub",
        "sub__None",
        "air__None",
    )


def _tet_volumes(mesh):
    x = np.asarray(mesh.nodes)[mesh.tetrahedra]
    e = x[:, 1:] - x[:, :1]
    return np.abs(np.linalg.det(e)) / 6


def test_node_coordinates_are_si(two_box):
    nodes = np.asarray(two_box.nodes)
    np.testing.assert_allclose(nodes.min(axis=0), [0, 0, 0], atol=1e-18)
    np.testing.assert_allclose(nodes.max(axis=0), [2e-6, 2e-6, 2e-6], rtol=1e-12)
    vol = _tet_volumes(two_box)
    for name in ("sub", "air"):
        in_region = two_box.tetrahedron_tags == two_box.region(name).tag
        np.testing.assert_allclose(vol[in_region].sum(), 4e-18, rtol=1e-12)


@pytest.mark.parametrize(("unit", "scale"), [("m", 1.0), ("mm", 1e-3), ("nm", 1e-9)])
def test_length_unit_scales_the_file_coordinates(unit, scale):
    mesh = gfem.Mesh.from_gmsh(TWO_BOX["msh22"], length_unit=unit)
    assert mesh.length_unit == unit
    np.testing.assert_allclose(np.asarray(mesh.nodes).max(), 2 * scale, rtol=1e-12)


def test_unknown_length_unit_names_the_supported_ones():
    with pytest.raises(ValueError, match=r"'inch'.*'um'"):
        gfem.Mesh.from_gmsh(TWO_BOX["msh22"], length_unit="inch")


def test_elements_lists_a_regions_tetrahedra_or_triangles_in_mesh_order(two_box):
    sub = two_box.elements("sub")
    np.testing.assert_array_equal(
        sub, np.flatnonzero(two_box.tetrahedron_tags == VOLUMES["sub"])
    )
    assert not sub.flags.writeable
    np.testing.assert_allclose(_tet_volumes(two_box)[sub].sum(), 4e-18, rtol=1e-12)
    metal = two_box.elements("metal_xy")
    np.testing.assert_array_equal(
        metal, np.flatnonzero(two_box.triangle_tags == SURFACES["metal_xy"])
    )
    with pytest.raises(gfem.RegionNotFoundError, match="Did you mean 'air'"):
        two_box.elements("ari")


def test_unknown_region_name_suggests_a_close_match(two_box):
    with pytest.raises(gfem.RegionNotFoundError, match="air__sub"):
        two_box.region("air_sub")


def test_mesh_is_a_pytree_whose_only_leaf_is_the_node_coordinates(two_box):
    (path, leaf), *rest = jax.tree_util.tree_flatten_with_path(two_box)[0]
    assert rest == []
    assert jax.tree_util.keystr(path) == ".nodes"
    assert leaf is two_box.nodes
    assert leaf.shape == (len(np.asarray(two_box.nodes)), 3)
    assert leaf.dtype == jnp.float64


def test_jit_and_grad_see_node_coordinates_and_keep_regions(two_box):
    moved = jax.jit(lambda m: jax.tree_util.tree_map(lambda x: x + 1e-7, m))(two_box)
    assert isinstance(moved, gfem.Mesh)
    assert moved.regions == two_box.regions
    np.testing.assert_allclose(moved.nodes, two_box.nodes + 1e-7)

    cot = jax.grad(lambda m: jnp.sum(m.nodes**2))(two_box)
    np.testing.assert_allclose(cot.nodes, 2 * two_box.nodes)


def test_moved_mesh_keeps_the_imported_geometry(two_box):
    moved = jax.tree_util.tree_map(lambda x: 2 * x, two_box)
    np.testing.assert_array_equal(moved.imported_nodes, np.asarray(two_box.nodes))
    np.testing.assert_allclose(moved.nodes, 2 * moved.imported_nodes)


def test_adjacency_lists_the_volumes_on_each_side_of_a_surface(two_box):
    def sides(name):
        return {frozenset(pair) for pair in two_box.adjacency(name)}

    assert sides("air__sub") == {frozenset({"air", "sub"})}
    assert sides("sub__None") == {frozenset({"sub", None})}
    assert sides("air__None") == {frozenset({"air", None})}
    assert sides("metal_xy") == sides("metal_z") == {frozenset({"sub"})}
    assert two_box.adjacent_volumes("air__sub") == ("sub", "air")
    assert two_box.adjacent_volumes("metal_z") == ("sub",)
    assert two_box.adjacent_volumes("air__None") == ("air",)


def test_adjacency_follows_triangle_orientation_domin_behind_domout_in_front(two_box):
    x = np.asarray(two_box.nodes)
    tri = x[two_box.triangles]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centroid = x[two_box.tetrahedra].mean(axis=1)
    inside, outside = two_box.triangle_neighbours.T
    assert (inside >= 0).all()  # every fixture triangle has a tet behind it
    behind = np.einsum("ij,ij->i", normal, centroid[inside] - tri[:, 0])
    assert (behind < 0).all()
    has_out = outside >= 0
    front = np.einsum(
        "ij,ij->i", normal[has_out], centroid[outside[has_out]] - tri[has_out, 0]
    )
    assert (front > 0).all()
    exterior = two_box.triangle_tags == two_box.region("sub__None").tag
    assert (outside[exterior] == -1).all()


def test_entities_sharing_a_volume_name_stay_apart(two_box):
    vol = _tet_volumes(two_box)
    ent = two_box.tetrahedron_entities
    sub = two_box.tetrahedron_tags == two_box.region("sub").tag
    by_entity = {int(e): vol[sub & (ent == e)].sum() for e in np.unique(ent[sub])}
    assert len(by_entity) == 2
    conductor = min(by_entity, key=by_entity.get)
    np.testing.assert_allclose(by_entity[conductor], 0.4e-18, rtol=1e-12)

    metal = two_box.triangle_tags == two_box.region("metal_xy").tag
    pairs = ent[two_box.triangle_neighbours[metal]]
    assert {frozenset(p.tolist()) for p in pairs} == {frozenset(by_entity)}


@pytest.mark.parametrize("fmt", ["msh22", "msh41"])
def test_second_order_mesh_raises_with_the_element_order_1_remedy(fmt):
    with pytest.raises(ValueError, match=r"tetra10.*triangle6") as err:
        gfem.Mesh.from_gmsh(DATA / f"cube_order2_{fmt}.msh")
    assert "ElementOrder 1" in str(err.value)


@pytest.mark.parametrize("fmt", ["msh22", "msh41"])
def test_a_face_in_two_physical_groups_raises_naming_both(fmt):
    # gmsh fixture: every cube face in "wall" (2), one face also in "port" (3)
    with pytest.raises(ValueError, match=r"more than one physical group.*'port'"):
        gfem.Mesh.from_gmsh(DATA / f"cube_overlap_{fmt}.msh")


def test_region_ids_export_to_vtu(two_box, tmp_path):
    path = tmp_path / "regions.vtu"
    two_box.to_vtu(path)
    back = meshio.read(path)
    np.testing.assert_allclose(back.points, two_box.nodes)
    by_type = {
        block.type: (block.data, region, entity)
        for block, region, entity in zip(
            back.cells,
            back.cell_data["region"],
            back.cell_data["entity"],
            strict=True,
        )
    }
    tets, tet_region, tet_entity = by_type["tetra"]
    np.testing.assert_array_equal(tets, two_box.tetrahedra)
    np.testing.assert_array_equal(tet_region, two_box.tetrahedron_tags)
    np.testing.assert_array_equal(tet_entity, two_box.tetrahedron_entities)
    tris, tri_region, _ = by_type["triangle"]
    np.testing.assert_array_equal(tris, two_box.triangles)
    np.testing.assert_array_equal(tri_region, two_box.triangle_tags)
    assert set(np.unique(tet_region)) == {1, 2}


def test_an_unnamed_physical_group_raises(tmp_path):
    text = (DATA / "two_box_msh22.msh").read_text()
    path = tmp_path / "unnamed.msh"
    path.write_text(
        text.replace('2 7 "air__sub"\n', "").replace(
            "$PhysicalNames\n7", "$PhysicalNames\n6"
        )
    )
    with pytest.raises(ValueError, match=r"\[7\].*no name"):
        gfem.Mesh.from_gmsh(path)


def _retag_first(kind, tmp_path):
    """Copy the msh 2.2 fixture with the first element of `kind` put in group 0."""
    gmsh_type = {"triangle": "2", "tetra": "4"}[kind]
    lines = (DATA / "two_box_msh22.msh").read_text().splitlines()
    for i, line in enumerate(lines):
        f = line.split()
        if len(f) > 5 and f[1] == gmsh_type and f[2] == "2":
            lines[i] = " ".join([f[0], f[1], f[2], "0", *f[4:]])
            break
    path = tmp_path / f"untagged_{kind}.msh"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_triangles_outside_any_physical_group_are_ignored(tmp_path):
    full = gfem.Mesh.from_gmsh(TWO_BOX["msh22"])
    mesh = gfem.Mesh.from_gmsh(_retag_first("triangle", tmp_path))
    assert len(mesh.triangles) == len(full.triangles) - 1
    assert 0 not in mesh.triangle_tags


def test_tetrahedra_outside_any_physical_group_raise(tmp_path):
    with pytest.raises(ValueError, match="outside any volume physical group"):
        gfem.Mesh.from_gmsh(_retag_first("tetra", tmp_path))
