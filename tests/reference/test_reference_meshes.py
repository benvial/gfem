"""The reference meshes import, build and crack as their metadata and Palace say.

Every case directory of the reference data with a ``palace.msh`` carries
gmsh's ``metadata.json`` (group names, tags, element counts, bounding box,
topology); the driven cases also fix Palace's cracked free DOF count.
"""

import json

import ngsolve as ngs
import numpy as np
import pytest

import gfem
from gfem._engine.domain import hcurl_space
from gfem._engine.mesh import to_ngsolve

#: Palace's order-2 free DOF counts on the cracked driven domains.
DRIVEN_DOFS = {"r8": 162_640, "r4": 338_338, "r8_box150": 333_910}


def _gsim_model(mesh: gfem.Mesh) -> gfem.Model:
    return gfem.Model(
        mesh,
        materials={
            "sio2": gfem.Material(eps_r=4.1),
            "sin": gfem.Material(eps_r=7.5, tan_delta=1e-3),
            "air": gfem.Material(),
        },
        excluded=(gfem.enclosed_by("topmetal2_xy", "topmetal2_z"),),
    )


def test_reference_mesh_imports_with_its_metadata_names_and_numbers(palace_case):
    meta = json.loads((palace_case / "metadata.json").read_text())["mesh"]
    mesh = gfem.Mesh.from_gmsh(palace_case / "palace.msh", length_unit="um")

    for key, dim, tags in (
        ("volumes", 3, mesh.tetrahedron_tags),
        ("surfaces", 2, mesh.triangle_tags),
    ):
        expected = {g["name"]: (g["tag"], g["elements"]) for g in meta["groups"][key]}
        got = {
            r.name: (r.tag, int((tags == r.tag).sum()))
            for r in mesh.regions
            if r.dim == dim
        }
        assert got == expected
    assert len(mesh.tetrahedra) == meta["elements_by_type"]["Tetrahedron 4"]
    assert len(mesh.triangles) == meta["elements_by_type"]["Triangle 3"]
    box = meta["bbox"]
    np.testing.assert_allclose(
        np.asarray(mesh.nodes).min(axis=0),
        np.array([box["xmin"], box["ymin"], box["zmin"]]) * 1e-6,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        np.asarray(mesh.nodes).max(axis=0),
        np.array([box["xmax"], box["ymax"], box["zmax"]]) * 1e-6,
        atol=2e-13,
    )
    for surface in mesh.surfaces:
        assert mesh.adjacent_volumes(surface)


def test_reference_mesh_topology_matches_gmsh_metadata(palace_case):
    meta = json.loads((palace_case / "metadata.json").read_text())["mesh"]
    built = to_ngsolve(gfem.Mesh.from_gmsh(palace_case / "palace.msh"))
    assert built.mesh.ne == meta["tetrahedra"]
    assert built.mesh.nedge == meta["topology"]["edges"]
    assert built.mesh.nface == meta["topology"]["triangular_faces"]
    dofs = meta["field_dofs"]
    if dofs["estimated_field_dofs"] is not None:  # uncracked Nedelec count
        fes = ngs.HCurl(built.mesh, order=dofs["field_order"], type1=True)
        assert fes.ndof == dofs["estimated_field_dofs"]


@pytest.mark.parametrize("case", list(DRIVEN_DOFS))
def test_enclosed_by_finds_the_gsim_conductors_inside_sio2(reference_dir, case):
    mesh = gfem.Mesh.from_gmsh(reference_dir / "driven" / case / "palace.msh")
    model = _gsim_model(mesh)
    assert model.excluded_entities == (("sio2", 5), ("sio2", 6), ("sio2", 7))


@pytest.mark.parametrize(("case", "ndof"), DRIVEN_DOFS.items())
def test_order2_free_dofs_match_the_cracked_palace_domain(reference_dir, case, ndof):
    mesh = gfem.Mesh.from_gmsh(reference_dir / "driven" / case / "palace.msh")
    space = hcurl_space(_gsim_model(mesh), to_ngsolve(mesh), order=2)
    assert sum(space.FreeDofs()) == ndof
