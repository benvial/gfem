"""`gfem.shape.boundary_morph` on a periodic model: tied motion of paired nodes.

The cell is a 10 mm cube periodic along x, y and z, split at x = 5 mm into
the volumes "lo" and "hi" by the interior surface "mid", which crosses the
y and z seams.
"""

import collections

import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._periodic import tie_classes
from gfem.shape import boundary_morph

MM = 1e-3
L = 10.0  # mm


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    edges = (np.linspace(0, L, 5),) * 3
    cell = sm.periodic_cell(
        edges,
        lambda c: "lo" if c[0] < L / 2 else "hi",
        sheets=(("mid", L / 2),),
    )
    path = sm.write_msh22(tmp_path_factory.mktemp("cube") / "cube.msh", cell)
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"lo": gfem.Material(eps_r=2.0), "hi": gfem.Material()},
        boundaries=[gfem.Periodic((f"{a}_min", f"{a}_max")) for a in "xyz"],
    )


def _displacement(morph, model, t):
    return np.asarray(morph(t).nodes) - model.mesh.imported_nodes


def test_corners_edges_and_faces_of_a_three_pair_cell_form_tie_classes(model):
    x = model.mesh.imported_nodes
    ties = tie_classes(len(x), model._periodic)
    sizes = collections.Counter(collections.Counter(ties.tolist()).values())
    # 5 x 5 x 5 nodes: the 8 corners, 4 parallel copies of each of the 9
    # inner cell-edge nodes, 2 copies of each of the 27 inner face nodes
    assert sizes == {1: 27, 2: 27, 4: 9, 8: 1}
    corner = ties[np.flatnonzero((x == 0).all(axis=1))[0]]
    np.testing.assert_array_equal(x[corner], [0.0, 0.0, 0.0])
    # a representative is on the first surface of each of its pairs
    assert not np.isclose(x[ties], L * MM).any()


def test_twins_move_together_and_keep_every_pair(model):
    morph = boundary_morph(model, "mid", grows="lo")
    # 9 inner nodes, 3 + 3 seam-edge classes of 2, 1 corner class of 4
    assert morph.n_params == 16
    np.testing.assert_allclose(morph.normals, np.tile([1.0, 0, 0], (16, 1)))
    t = np.random.default_rng(0).uniform(-1, 1, 16) * 1e-4
    u = _displacement(morph, model, t)
    for p in model._periodic:
        np.testing.assert_allclose(
            u[p.second_nodes], u[p.first_nodes], rtol=0, atol=1e-18
        )
    # the moved mesh still pairs (checked at the build)
    model.replace(mesh=morph(t))


def test_a_selection_of_one_side_of_the_seam_moves_both_twins(model):
    morph = boundary_morph(
        model, "mid", grows="lo", where=lambda p: p[:, 1] < L * MM / 2
    )
    # rows y = 0 and 2.5 mm: the corner and 3 edge classes, 1 + 3 classes
    assert morph.n_params == 8
    u = _displacement(morph, model, np.full(8, 1e-4))
    x = model.mesh.imported_nodes
    on = np.isclose(x[:, 0], L * MM / 2)
    first = np.flatnonzero(on & np.isclose(x[:, 1], 0) & np.isclose(x[:, 2], 0.005))
    second = np.flatnonzero(
        on & np.isclose(x[:, 1], L * MM) & np.isclose(x[:, 2], 0.005)
    )
    np.testing.assert_allclose(u[second], u[first], rtol=0, atol=1e-18)
    np.testing.assert_allclose(u[first], [[1e-4, 0, 0]], rtol=1e-12)


def test_a_held_member_holds_its_class(model):
    morph = boundary_morph(model, "mid", grows="lo", fixed=["y_min"])
    # the classes on y = 0, and so on y = L, are held
    assert morph.n_params == 16 - 4
    u = _displacement(morph, model, np.full(morph.n_params, 1e-4))
    x = model.mesh.imported_nodes
    on_y = np.isclose(x[:, 1], 0) | np.isclose(x[:, 1], L * MM)
    assert np.abs(u).max() > 0
    np.testing.assert_array_equal(u[on_y], 0.0)


@pytest.mark.parametrize("surface", ["x_min", "x_max"])
def test_a_paired_surface_cannot_be_the_moving_surface(model, surface):
    with pytest.raises(ValueError, match="cannot be the moving surface") as info:
        boundary_morph(model, surface)
    assert "slide tied to" in str(info.value)
