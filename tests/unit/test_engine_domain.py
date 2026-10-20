"""The engine solves electromagnetic problems on the model's domain only."""

from pathlib import Path

import ngsolve as ngs
import numpy as np
import pytest

import gfem
from gfem._engine.cache import discretization
from gfem._engine.domain import em_region, hcurl_space
from gfem._engine.mesh import to_ngsolve

DATA = Path(__file__).parent / "data"
METAL = gfem.enclosed_by("metal_xy", "metal_z")


@pytest.fixture(params=["msh22", "msh41"])
def mesh(request):
    return gfem.Mesh.from_gmsh(DATA / f"two_box_{request.param}.msh")


def _model(mesh, excluded=()):
    return gfem.Model(
        mesh,
        materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()},
        excluded=excluded,
    )


def _nedelec2_count(tets):
    """Order-2 first-kind Nedelec DOFs on these tetrahedra: 2 per edge + 2 per face."""
    edges = np.sort(tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]], axis=2)
    faces = np.sort(tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]], axis=2)
    n_edges = len(np.unique(edges.reshape(-1, 2), axis=0))
    n_faces = len(np.unique(faces.reshape(-1, 3), axis=0))
    return 2 * n_edges + 2 * n_faces


def test_region_volume_leaves_out_the_excluded_conductor(mesh):
    built = to_ngsolve(mesh)
    full = ngs.Integrate(1, built.mesh, definedon=em_region(_model(mesh), built))
    cut = ngs.Integrate(
        1, built.mesh, definedon=em_region(_model(mesh, (METAL,)), built)
    )
    # file units are um: 2 x 2 x 2 box, 1 x 1 x 0.4 conductor
    assert full == pytest.approx(8.0, rel=1e-12)
    assert cut == pytest.approx(7.6, rel=1e-12)


def test_hcurl_space_has_dofs_only_on_the_solve_domain(mesh):
    built = to_ngsolve(mesh)
    whole = hcurl_space(_model(mesh), built, order=2)
    assert whole.ndof == _nedelec2_count(mesh.tetrahedra)
    model = _model(mesh, (METAL,))
    space = hcurl_space(model, built, order=2)
    active = mesh.tetrahedra[_active(mesh, model)]
    assert space.ndof == _nedelec2_count(active)
    assert sum(space.FreeDofs()) == space.ndof
    assert space.is_complex


def _active(mesh, model):
    out = np.ones(len(mesh.tetrahedra), bool)
    names = {r.name: r.tag for r in mesh.regions}
    for region, entity in model.excluded_entities:
        out &= ~(
            (mesh.tetrahedron_tags == names[region])
            & (mesh.tetrahedron_entities == entity)
        )
    return out


def test_solves_on_one_mesh_share_spaces_per_physics_domain():
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    problem = gfem.Electrostatic(
        terminals={"metal": ["metal_xy", "metal_z"]}, ground="sub__None"
    )
    full, cut = _model(mesh), _model(mesh, (METAL,))
    retuned = full.with_material("sub", gfem.Material(eps_r=3.9))
    for model in (full, cut, retuned):
        gfem.solve(model, problem)
    domains = {key[0] for key in discretization(mesh).spaces}
    # the excluded metal makes another physics domain; a material does not
    assert domains == {("em", ()), ("em", cut._domain.excluded)}
