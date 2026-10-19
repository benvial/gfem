"""Mesh quality: the scaled Jacobian, and the guards on moved meshes."""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

import gfem
from gfem.shape import boundary_morph, quality

DATA = Path(__file__).parent / "data"
UM = 1e-6
# A regular tetrahedron, its vertices listed with negative orientation.
REGULAR = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


def _one_tet(tmp_path, order):
    path = tmp_path / "tet.msh"
    meshio.write(
        path,
        meshio.Mesh(
            REGULAR,
            [("tetra", np.array([order]))],
            cell_data={"gmsh:physical": [np.array([1])], "gmsh:geometrical": [[1]]},
            field_data={"box": np.array([1, 3])},
        ),
        file_format="gmsh22",
        binary=False,
    )
    return gfem.Mesh.from_gmsh(path)


def _with_nodes(mesh, nodes):
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [nodes])


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [1, 0, 2, 3]])
def test_a_regular_tet_scores_one_whatever_its_orientation(tmp_path, order):
    mesh = _one_tet(tmp_path, order)
    np.testing.assert_allclose(quality(mesh), [1.0], rtol=1e-14)


def test_flat_and_inverted_tets_score_zero_and_below(tmp_path):
    mesh = _one_tet(tmp_path, [0, 1, 2, 3])
    x = np.asarray(mesh.nodes)
    centre = x[1:].mean(axis=0)
    flat = x.copy()
    flat[0] = centre  # node 0 into the plane of the other three
    through = x.copy()
    through[0] = 2 * centre - x[0]  # node 0 reflected through that plane
    assert float(quality(_with_nodes(mesh, jnp.asarray(flat)))[0]) == pytest.approx(
        0, abs=1e-14
    )
    assert float(quality(_with_nodes(mesh, jnp.asarray(through)))[0]) < 0


def test_quality_is_differentiable(tmp_path):
    mesh = _one_tet(tmp_path, [0, 1, 2, 3])
    rng = np.random.default_rng(0)
    x = mesh.nodes + 0.2 * UM * rng.standard_normal((4, 3))  # a distorted tet

    def q(x):
        return quality(_with_nodes(mesh, x))[0]

    g = np.asarray(jax.grad(q)(x))
    dx = 1e-4 * UM * rng.standard_normal((4, 3))
    fd = (q(x + dx) - q(x - dx)) / 2
    assert float(np.sum(g * dx)) == pytest.approx(float(fd), rel=1e-6)


@pytest.fixture(scope="module")
def mesh():
    return gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")


@pytest.fixture(scope="module")
def model(mesh):
    return gfem.Model(
        mesh,
        materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()},
        excluded=(gfem.enclosed_by("metal_xy", "metal_z"),),
    )


def test_imported_gmsh_meshes_have_positive_quality(mesh):
    q = np.asarray(quality(mesh))
    assert q.shape == (len(mesh.tetrahedra),)
    assert (q > 0).all()
    assert (q <= 1 + 1e-12).all()


def _active(mesh, model):
    out = np.ones(len(mesh.tetrahedra), bool)
    tag = {r.name: r.tag for r in mesh.regions}
    for region, entity in model.excluded_entities:
        out &= ~(
            (mesh.tetrahedron_tags == tag[region])
            & (mesh.tetrahedron_entities == entity)
        )
    return out


def test_a_morph_warns_then_raises_as_elements_degrade(model, mesh):
    # the conductor's +x wall (x = 1.5 um) moves into the 0.5 um gap to x = 2 um
    morph = boundary_morph(
        model, "metal_z", where=lambda p: np.abs(p[:, 0] - 1.5 * UM) < 1e-12
    )
    active = _active(mesh, model)
    q0 = np.asarray(quality(mesh))
    # under jit the morph does not check, so q can be computed for any step
    moved_nodes = jax.jit(lambda t: morph(t).nodes)
    seen = set()
    for a in np.arange(0.025, 0.7, 0.025) * UM:
        t = a * np.ones(morph.n_params)
        q = np.asarray(quality(_with_nodes(mesh, moved_nodes(t))))
        if (q[active] > 0).all():
            expect_warning = (q / q0)[active].min() < 0.25
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                morph(t)
            hits = [w for w in caught if w.category is gfem.MeshQualityWarning]
            assert bool(hits) == expect_warning, a
            seen.add("warning" if hits else "ok")
            if hits:
                message = str(hits[0].message)
                assert "region 'sub'" in message
                assert "remesh" in message
        else:
            with pytest.raises(gfem.MeshQualityError, match="inverted"):
                morph(t)
            seen.add("error")
    assert seen == {"ok", "warning", "error"}


def test_quality_errors_are_value_errors_and_warnings_user_warnings():
    assert issubclass(gfem.MeshQualityError, ValueError)
    assert issubclass(gfem.MeshQualityWarning, UserWarning)
