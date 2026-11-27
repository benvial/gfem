"""`gfem._refine.bisect` on a curved mesh: the quadratic geometry is kept.

A child's mid-edge nodes lie on the parent's quadratic element map, so the
refined mesh describes the same curved surface and volume, not a faceted one.
"""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest

import gfem
from gfem._engine.mesh import to_ngsolve
from gfem._refine import bisect
from gfem.mesh import _TET_EDGES, _TRI_EDGES

DATA = Path(__file__).parent / "data"
TET_EDGES = np.array(_TET_EDGES)
# A curved triangle's area element is the square root of a polynomial, so its
# quadrature is not exact and its error differs between a mesh and its
# refinement (1.4e-12 relative at the default order). A high order makes that
# error negligible against the 1e-12 bound.
AREA_ORDER = 16


@pytest.fixture(scope="module")
def octant():
    return gfem.Mesh.from_gmsh(DATA / "octant_order2.msh", length_unit="m")


def _marked(mesh, seed):
    rng = np.random.default_rng(seed)
    return rng.random(len(mesh.tetrahedra)) < 0.3


def _volume(mesh):
    built = to_ngsolve(mesh)
    return ngs.Integrate(1, built.mesh) * built.scale**3


def _area(mesh):
    built = to_ngsolve(mesh)
    return (
        ngs.Integrate(1, built.mesh.Boundaries("wall"), order=AREA_ORDER)
        * built.scale**2
    )


@pytest.fixture(scope="module")
def cycles(octant):
    """The octant bisected three times, with a different random mark each time."""
    meshes, refinements = [octant], []
    for seed in range(3):
        fine, ref = bisect(meshes[-1], _marked(meshes[-1], seed))
        meshes.append(fine)
        refinements.append(ref)
    return meshes, refinements


def test_a_refined_curved_mesh_is_curved(cycles):
    meshes, _ = cycles
    assert all(m.curved for m in meshes)
    t = meshes[-1]._topology
    assert t.volume_midnodes.shape == (len(t.volume_elements), 6)
    assert t.surface_midnodes.shape == (len(t.surface_elements), 3)


def test_the_curved_volume_and_area_are_kept_over_three_cycles(cycles):
    # Each child's map is the parent's quadratic map restricted to a
    # sub-simplex, again quadratic, so the integrals are those of the same
    # polynomial pieces: they agree to round-off. Each is a sum of n element
    # integrals of relative round-off a few eps (eps = 2.2e-16); n <= ~1e4 and
    # a random-walk accumulation gives sqrt(n) * few * eps ~ 1e-13, one order
    # under the 1e-12 bound (a faceted refinement loses ~1e-3).
    meshes, _ = cycles
    v0, a0 = _volume(meshes[0]), _area(meshes[0])
    for m in meshes[1:]:
        assert abs(_volume(m) / v0 - 1) < 1e-12
        assert abs(_area(m) / a0 - 1) < 1e-12


def test_the_quadratic_geometry_is_not_the_faceted_one(octant):
    # Guard for the check above: dropping the mid nodes changes the volume by
    # the sagitta effect, ~1e-3 of the octant's volume at size 0.2.
    fine, _ = bisect(octant, _marked(octant, 0))
    t = fine._topology
    straight = gfem.Mesh(
        jnp.asarray(fine.imported_nodes),
        dataclasses.replace(t, volume_midnodes=None, surface_midnodes=None),
    )
    assert abs(_volume(straight) / _volume(fine) - 1) > 1e-5


def test_every_edge_has_one_mid_node_and_no_node_is_orphaned(cycles):
    meshes, _ = cycles
    for m in meshes[1:]:
        t = m._topology
        tets, tmid = t.volume_elements, t.volume_midnodes
        tris, smid = t.surface_elements, t.surface_midnodes
        ends = np.concatenate(
            [
                tets[:, TET_EDGES].reshape(-1, 2),
                tris[:, np.array(_TRI_EDGES)].reshape(-1, 2),
            ]
        )
        mids = np.concatenate([tmid.ravel(), smid.ravel()])
        edge = np.sort(ends, axis=1)
        # one mid node per edge ...
        pairs = np.unique(np.column_stack([edge, mids]), axis=0)
        assert len(pairs) == len(np.unique(edge, axis=0))
        # ... and a different edge never shares it
        assert len(np.unique(pairs[:, 2])) == len(pairs)
        # the mid nodes are not corners
        assert not np.isin(mids, tets).any()
        # every node of the mesh is a corner or a mid node of a tetrahedron
        used = np.zeros(len(m.imported_nodes), bool)
        used[tets] = True
        used[tmid] = True
        assert used.all()


def test_children_keep_tags_entities_and_parents_as_on_a_straight_mesh(cycles):
    meshes, refinements = cycles
    for coarse, fine, ref in zip(meshes[:-1], meshes[1:], refinements, strict=True):
        tc, tf = coarse._topology, fine._topology
        np.testing.assert_array_equal(
            tf.volume_tags, tc.volume_tags[ref.tetrahedron_parents]
        )
        np.testing.assert_array_equal(
            tf.surface_entities, tc.surface_entities[ref.triangle_parents]
        )
        assert len(tf.volume_elements) > len(tc.volume_elements)


def test_each_parent_keeps_its_own_curved_volume(octant):
    # Not only the total: the children of every parent fill that parent's
    # curved volume. Element volumes are ~1e-3 of the total; the same
    # round-off argument as above holds per element, relative 1e-12.
    fine, ref = bisect(octant, _marked(octant, 5))
    coarse_v = np.asarray(ngs.Integrate(1, to_ngsolve(octant).mesh, element_wise=True))
    fine_v = np.asarray(ngs.Integrate(1, to_ngsolve(fine).mesh, element_wise=True))
    summed = np.bincount(
        ref.tetrahedron_parents, weights=fine_v, minlength=len(coarse_v)
    )
    np.testing.assert_allclose(summed, coarse_v, rtol=1e-12, atol=0)


def test_the_new_nodes_are_not_chord_midpoints(octant):
    # A mid node on the parent's map differs from the chord midpoint by about
    # the parent's sagitta; a straight rule would put it on the chord.
    fine, _ = bisect(octant, _marked(octant, 0))
    t = fine._topology
    x = np.asarray(fine.imported_nodes)
    new = np.arange(len(octant.imported_nodes), len(x))
    pts = x[t.volume_elements[:, TET_EDGES]]  # (n, 6, 2, 3)
    chord = pts.mean(axis=2)
    off = np.linalg.norm(x[t.volume_midnodes] - chord, axis=-1).max()
    assert len(new) > 0
    assert off > 1e-3
