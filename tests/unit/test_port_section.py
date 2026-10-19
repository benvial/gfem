"""The planar cross-section the port mode problem solves on."""

from pathlib import Path

import numpy as np
import pytest

import gfem
from gfem._port_section import port_section

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def slab_model():
    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    return gfem.Model(
        mesh,
        materials={
            "slab": gfem.Material(eps_r=4.0, tan_delta=0.01, conductivity=2.0),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


@pytest.mark.parametrize(("name", "inward"), [("p1", 1.0), ("p2", -1.0)])
def test_frame_is_right_handed_with_the_normal_into_the_domain(
    slab_model, name, inward
):
    port = next(p for p in slab_model.ports if p.name == name)
    s = port_section(slab_model, port)
    np.testing.assert_allclose(s.frame @ s.frame.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(s.frame[0], s.frame[1]), s.frame[2], atol=1e-12)
    np.testing.assert_allclose(s.frame[2], [0.0, 0.0, inward], atol=1e-12)


def test_triangles_are_counter_clockwise_and_cover_the_port(slab_model):
    s = port_section(slab_model, slab_model.ports[0])
    p = s.points[s.triangles]
    area2 = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
        p[:, 1, 1] - p[:, 0, 1]
    ) * (p[:, 2, 0] - p[:, 0, 0])
    assert (area2 > 0).all()
    np.testing.assert_allclose(area2.sum() / 2, 22.86 * 10.16, rtol=1e-12)
    np.testing.assert_allclose(s.to_global(s.points)[:, 2], 0.0, atol=1e-15)


def test_materials_come_from_the_volume_behind_each_triangle(slab_model):
    s = port_section(slab_model, slab_model.ports[0])
    centroid = s.to_global(s.points[s.triangles].mean(axis=1))
    in_slab = centroid[:, 0] < 8e-3
    assert in_slab.any()
    assert (~in_slab).any()
    np.testing.assert_allclose(s.eps_nn[in_slab], 4.0 * (1 - 0.01j))
    np.testing.assert_allclose(s.eps_nn[~in_slab], 1.0)
    np.testing.assert_allclose(s.sigma[in_slab], 2.0)
    np.testing.assert_allclose(s.sigma[~in_slab], 0.0)
    np.testing.assert_allclose(
        s.nu_tt, np.broadcast_to(np.eye(2), s.nu_tt.shape), atol=1e-15
    )


def test_every_outer_edge_is_on_the_pec_walls(slab_model):
    s = port_section(slab_model, slab_model.ports[0])
    edges = np.sort(s.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    outer = uniq[counts == 1]
    pec = np.unique(np.sort(s.pec_edges, axis=1), axis=0)
    np.testing.assert_array_equal(pec, outer)
    assert len(s.absorbing_edges) == 0
