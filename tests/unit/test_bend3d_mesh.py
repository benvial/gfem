"""The 3D bend mesh builder (tools/bend3d_mesh.py): geometry only, no solve."""

import bend3d_mesh as bm
import numpy as np
import pytest

H = 0.4  # um: the coarsest in-plane cell, a few thousand tetrahedra


@pytest.fixture(scope="module")
def half():
    return bm.bend3d(H, half=True)


@pytest.fixture(scope="module")
def full():
    return bm.bend3d(H, half=False)


@pytest.fixture(scope="module")
def straight():
    return bm.bend3d(H, half=True, kind="straight")


def test_region_and_surface_names(half, full, straight):
    for mesh, surfaces in (
        (half, {"port1", "port2", "outer", "mirror"}),
        (full, {"port1", "port2", "outer"}),
        (straight, {"port1", "port2", "outer", "mirror"}),
    ):
        _, _, tet_names, _, tri_names = mesh
        assert set(tet_names) == {"cladding", "core", "design"}
        assert set(tri_names) == surfaces


def _faces(tris):
    return {tuple(f) for f in np.sort(tris, axis=1).tolist()}


def test_every_exterior_face_is_named(half, full, straight):
    for _, tets, _, tris, _ in (half, full, straight):
        faces = np.sort(tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]], axis=2)
        uniq, counts = np.unique(faces.reshape(-1, 3), axis=0, return_counts=True)
        assert _faces(uniq[counts == 1]) <= _faces(tris)


def _cells(nodes, cells, names, keep=None):
    """{(cell as a set of node coordinates, region name)}, comparable across meshes."""
    xyz = [tuple(p) for p in np.round(nodes, 9).tolist()]
    return {
        (frozenset(xyz[v] for v in cell), name)
        for cell, name in zip(cells.tolist(), names, strict=True)
        if keep is None or keep(nodes[cell])
    }


def test_full_mesh_is_symmetric_under_z_reflection(full):
    nodes, tets, tet_names, tris, tri_names = full
    flipped = nodes * [1.0, 1.0, -1.0]
    assert _cells(flipped, tets, tet_names) == _cells(nodes, tets, tet_names)
    assert _cells(flipped, tris, tri_names) == _cells(nodes, tris, tri_names)


def test_half_mesh_is_the_full_meshs_upper_part(half, full):
    nodes, tets, tet_names, tris, tri_names = half
    upper = full[0][:, 2] >= 0
    assert np.allclose(np.unique(nodes, axis=0), np.unique(full[0][upper], axis=0))

    def above(p):
        return bool((p[:, 2] >= -1e-12).all())

    assert _cells(nodes, tets, tet_names) == _cells(*full[:3], keep=above)
    named = [n != "mirror" for n in tri_names]
    assert _cells(nodes, tris[named], np.array(tri_names)[named].tolist()) == _cells(
        full[0], *full[3:], keep=above
    )


def test_straight_guide_reductions():
    nodes, _, _, _, tri_names = bm.bend3d(H, half=True, kind="straight", ymirror=True)
    assert "ywall" in tri_names
    assert nodes[:, 1].min() == pytest.approx(0.0)
    short = bm.bend3d(H, half=True, kind="straight", length=3.2)[0]
    assert np.ptp(short[:, 0]) == pytest.approx(3.2)
    with pytest.raises(ValueError, match="straight"):
        bm.bend3d(H, half=True, ymirror=True)
    assert (
        bm.mesh_name(0.1, half=True, kind="straight", ymirror=True, length=3.2)
        == "straight3d_h0.1_half_ymirror_L3.2.msh"
    )
    assert bm.mesh_name(0.07, half=False, box=3.5) == "bend3d_h0.07_full_box3.5.msh"


def test_larger_box_keeps_the_5um_mesh_inside(half):
    nodes, tets, tet_names, _, _ = bm.bend3d(H, half=True, box=3.5)
    assert np.ptp(nodes[:, 0]) == pytest.approx(7.0)

    def inside(p):
        return bool((np.abs(p[:, :2]) <= bm.BOX + 1e-12).all())

    assert _cells(nodes, tets, tet_names, keep=inside) == _cells(*half[:3])
