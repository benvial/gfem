"""`gfem.Periodic` on a model: node pairing, build checks, wave vectors, job files.

The meshes are the periodic cells of tests/periodic_cases.py (mm), or
`structured_meshes.periodic_cell` variants that break one condition of a
periodic pair each.
"""

import numpy as np
import periodic_cases as pc
import pytest
import structured_meshes as sm

import gfem

MM = 1e-3


@pytest.fixture(scope="module")
def c2_mesh(tmp_path_factory):
    return pc.c2_mesh(tmp_path_factory.mktemp("c2"))


@pytest.fixture(scope="module")
def c4_mesh(tmp_path_factory):
    return pc.c4_mesh(tmp_path_factory.mktemp("c4"))


def _cell(tmp_path, cell, name="cell"):
    path = sm.write_msh22(tmp_path / f"{name}.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _box(size=(10.0, 7.0, 1.0), cells=(4, 3, 1), **kwargs):
    edges = tuple(np.linspace(0, s, n + 1) for s, n in zip(size, cells, strict=True))
    return sm.periodic_cell(edges, **kwargs)


def _model(mesh, boundaries, **kwargs):
    materials = kwargs.pop("materials", {"cell": gfem.Material()})
    return gfem.Model(mesh, materials=materials, boundaries=boundaries, **kwargs)


def _x_pair():
    return gfem.Periodic(("x_min", "x_max"))


# --- pairing on the C2 and C4 cells ------------------------------------------------


def test_two_pairs_measure_their_translations_in_metres(c2_mesh):
    model = pc.c2(c2_mesh)
    np.testing.assert_allclose(
        model.periodic_translations,
        [(pc.C2_AX * MM, 0, 0), (0, pc.C2_AY * MM, 0)],
        rtol=0,
        atol=1e-15,
    )
    assert all(isinstance(v, float) for a in model.periodic_translations for v in a)


def test_three_pairs_follow_boundary_order(c4_mesh):
    model = gfem.Model(
        c4_mesh,
        materials={"cell": gfem.Material()},
        boundaries=[
            gfem.Periodic(("z_max", "z_min")),
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
    )
    L = pc.C4_L * MM
    np.testing.assert_allclose(
        model.periodic_translations,
        [(0, 0, -L), (L, 0, 0), (0, L, 0)],
        rtol=0,
        atol=1e-15,
    )


def test_a_model_without_pairs_has_no_translations(c4_mesh):
    model = gfem.Model(c4_mesh, materials={"cell": gfem.Material()})
    assert model.periodic_translations == ()


def test_paired_surfaces_are_not_default_pmc(c2_mesh):
    model = pc.c2(c2_mesh)
    assert model.default_pmc == ()
    one = _model(c2_mesh, [_x_pair()], materials={"cell": gfem.Material(eps_r=2)})
    assert set(one.default_pmc) == {"y_min", "y_max", "z_min", "z_max"}
