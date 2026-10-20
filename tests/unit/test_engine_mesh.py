"""The engine builds its NGSolve mesh from a `gfem.Mesh`, keeping names and numbers."""

from pathlib import Path

import ngsolve as ngs
import pytest

import gfem
from gfem._engine.mesh import to_ngsolve

DATA = Path(__file__).parent / "data"


@pytest.fixture(params=["msh22", "msh41"])
def two_box(request):
    return gfem.Mesh.from_gmsh(DATA / f"two_box_{request.param}.msh")


def test_domains_and_boundaries_carry_region_names(two_box):
    built = to_ngsolve(two_box)
    assert set(built.mesh.GetMaterials()) == {"sub", "air"}
    assert set(built.mesh.GetBoundaries()) == set(two_box.surfaces)
    # coordinates stay in the file's length unit (um): 2 x 2 x 1 um per volume
    for name in ("sub", "air"):
        vol = ngs.Integrate(1, built.mesh, definedon=built.mesh.Materials(name))
        assert vol == pytest.approx(4.0, rel=1e-12)
    area = ngs.Integrate(
        1, built.mesh, ngs.BND, definedon=built.mesh.Boundaries("metal_xy")
    )
    assert area == pytest.approx(2.0, rel=1e-12)


def test_face_descriptors_keep_gmsh_numbers_and_adjacency(two_box):
    built = to_ngsolve(two_box)
    ngmesh = built.mesh.ngmesh
    materials = built.mesh.GetMaterials()
    seen = set()
    for i in range(ngmesh.GetNFaceDescriptors()):
        fd = ngmesh.FaceDescriptor(i + 1)
        assert fd.bc == two_box.region(fd.bcname).tag
        din = materials[fd.domin - 1] if fd.domin else None
        dout = materials[fd.domout - 1] if fd.domout else None
        assert din is not None  # one-sided faces point out of the mesh
        assert (din, dout) in two_box.adjacency(fd.bcname) or (
            dout,
            din,
        ) in two_box.adjacency(fd.bcname)
        seen.add((fd.bcname, frozenset({din, dout})))
    assert ("sub__None", frozenset({"sub", None})) in seen
    assert ("air__sub", frozenset({"sub", "air"})) in seen


def test_each_gmsh_entity_is_its_own_domain(two_box):
    built = to_ngsolve(two_box)
    assert len(built.domains) == 3  # substrate, conductor (both "sub"), air
    volumes = []
    for k in range(len(built.domains)):
        bits = ngs.BitArray(len(built.domains))
        bits.Clear()
        bits.Set(k)
        region = ngs.Region(built.mesh, ngs.VOL, bits)
        volumes.append(ngs.Integrate(1, built.mesh, definedon=region))
    assert sorted(volumes) == pytest.approx([0.4, 3.6, 4.0], rel=1e-12)


def test_element_order_maps_engine_elements_to_mesh_elements(two_box):
    built = to_ngsolve(two_box)
    tets = two_box.tetrahedra[built.tetrahedron_order]
    for i, el in enumerate(built.mesh.Elements(ngs.VOL)):
        assert sorted(v.nr for v in el.vertices) == sorted(tets[i])
    tris = two_box.triangles[built.triangle_order]
    for i, el in enumerate(built.mesh.Elements(ngs.BND)):
        assert sorted(v.nr for v in el.vertices) == sorted(tris[i])


def test_exterior_triangles_facing_into_the_mesh_are_flipped(tmp_path):
    lines = (DATA / "two_box_msh22.msh").read_text().splitlines()
    for i, line in enumerate(lines):
        f = line.split()
        if len(f) == 8 and f[1] == "2" and f[3] == "8":  # a "sub__None" triangle
            lines[i] = " ".join([*f[:5], f[7], f[6], f[5]])
    path = tmp_path / "inward.msh"
    path.write_text("\n".join(lines) + "\n")
    mesh = gfem.Mesh.from_gmsh(path)
    assert mesh.adjacency("sub__None") == ((None, "sub"),)

    built = to_ngsolve(mesh)
    ngmesh = built.mesh.ngmesh
    fds = [ngmesh.FaceDescriptor(i + 1) for i in range(ngmesh.GetNFaceDescriptors())]
    exterior = [fd for fd in fds if fd.bcname == "sub__None"]
    assert [(fd.domin > 0, fd.domout) for fd in exterior] == [(True, 0)]
    area = ngs.Integrate(
        1, built.mesh, ngs.BND, definedon=built.mesh.Boundaries("sub__None")
    )
    assert area == pytest.approx(4 + 4 * 2, rel=1e-12)  # bottom + four 2 x 1 sides
