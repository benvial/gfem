"""The planar cross-section the port mode problem solves on."""

import math
from pathlib import Path

import anisotropy
import numpy as np
import pytest

import gfem
from gfem._port_section import edge_admittance, port_section

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
    np.testing.assert_allclose(s.sigma_nn[in_slab], 2.0)
    np.testing.assert_allclose(s.sigma_nn[~in_slab], 0.0)
    np.testing.assert_allclose(
        s.sigma_tt[in_slab],
        np.broadcast_to(2 * np.eye(2), (in_slab.sum(), 2, 2)),
        atol=1e-15,
    )
    np.testing.assert_allclose(s.sigma_tt[~in_slab], 0.0)
    np.testing.assert_allclose(
        s.nu_tt, np.broadcast_to(np.eye(2), s.nu_tt.shape), atol=1e-15
    )


def test_per_axis_loss_tangent_is_split_along_the_port_frame(slab_model):
    # port normal = z: eps_nn holds the z value, the tangential block the x
    # and y ones seen along the frame's in-plane axes u, v
    slab = gfem.Material(eps_r=4.0, tan_delta=(0.01, 0.02, 0.03))
    model = slab_model.replace(materials={**slab_model.materials, "slab": slab})
    s = port_section(model, model.ports[0])
    centroid = s.to_global(s.points[s.triangles].mean(axis=1))
    in_slab = centroid[:, 0] < 8e-3
    np.testing.assert_allclose(s.eps_nn[in_slab], 4.0 * (1 - 0.03j))
    uv = s.frame[:2, :2]  # rows: u, v in (x, y)
    expected = uv @ np.diag(4.0 * (1 - 1j * np.array([0.01, 0.02]))) @ uv.T
    tangential = s.eps_tt[in_slab]
    np.testing.assert_allclose(tangential, np.broadcast_to(expected, tangential.shape))


def test_every_outer_edge_is_on_the_pec_walls(slab_model):
    s = port_section(slab_model, slab_model.ports[0])
    edges = np.sort(s.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    outer = uniq[counts == 1]
    pec = np.unique(np.sort(s.pec_edges, axis=1), axis=0)
    np.testing.assert_array_equal(pec, outer)
    assert len(s.absorbing_edges) == 0


# --- anisotropic sections --------------------------------------------------------

# Principal axes turned 45 degrees about z, the port normal.
_C = math.sqrt(0.5)
DIAGONAL = ((_C, _C, 0.0), (-_C, _C, 0.0), (0.0, 0.0, 1.0))


def test_index_bounds_use_the_principal_values_of_turned_tensors(slab_model):
    # eps_tt = diag(1, 4) at 45 degrees has the diagonal (2.5, 2.5); a mode
    # can see up to 4. sigma_tt = diag(0.1, 0.5) likewise has 0.3 on its
    # diagonal and sigma_nn = 0.2.
    turned = gfem.Material(
        eps_r=(1.0, 4.0, 1.0), conductivity=(0.1, 0.5, 0.2), axes=DIAGONAL
    )
    model = slab_model.replace(materials={"slab": turned, "air": turned})
    s = port_section(model, model.ports[0])
    n2_shift, n2_max = s.index_squared()
    assert n2_shift >= 4.0 * (1 - 1e-12)
    assert n2_max >= 4.0 * (1 - 1e-12)
    np.testing.assert_allclose(s.max_conductivity, 0.5, rtol=1e-12)


def _transverse(rng, lo, hi):
    """A random tensor with z principal, turned about z."""
    t = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(t), np.sin(t)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return r @ np.diag(rng.uniform(lo, hi, 3)) @ r.T


def _in_section(eps, mu):
    """``eps_tt, eps_nn, nu_tt, nu_nn`` of 3x3 tensors, frame (u, v, n) = (x, y, z)."""
    rot = np.array([[0.0, 1.0], [-1.0, 0.0]])  # R a = a x n
    nu_tt = rot.T @ np.linalg.inv(mu[:2, :2]) @ rot
    return eps[:2, :2], eps[2, 2], nu_tt, 1 / mu[2, 2]


def test_edge_admittance_is_the_3d_face_admittance_on_the_edge():
    rng = np.random.default_rng(0)
    z = np.array([0.0, 0.0, 1.0])
    for _ in range(20):
        eps, mu = _transverse(rng, 1.0, 12.0), _transverse(rng, 0.5, 3.0)
        phi = rng.uniform(0, 2 * np.pi)
        m = np.array([np.cos(phi), np.sin(phi), 0.0])  # in-plane edge normal
        tau = np.cross(z, m)
        y3 = anisotropy.admittance(eps, mu, m)
        et, en, nt, nn = _in_section(eps, mu)
        got = edge_admittance(
            et[None], np.array([en]), nt[None], np.array([nn]), m[None, :2], where="x"
        )[0]
        want = np.array([tau @ y3 @ tau, z @ y3 @ z])
        # measured 1e-15 / 2e-16 (round-off)
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert abs(tau @ y3 @ z) <= 1e-12 * np.abs(want).max()


def test_edge_admittance_of_diagonal_tensors_pairs_the_plane_wave_components():
    eps, mu = np.diag([2.0, 3.0, 5.0]), np.diag([1.5, 1.2, 1.1])
    et, en, nt, nn = _in_section(eps, mu)
    got = edge_admittance(
        et[None],
        np.array([en]),
        nt[None],
        np.array([nn]),
        np.array([[1.0, 0.0]]),
        where="x",
    )[0]
    # edge normal x, tangent y: E along y with H along z; E along z with H along y
    np.testing.assert_allclose(
        got, [np.sqrt(3.0 / 1.1), np.sqrt(5.0 / 1.2)], rtol=1e-15
    )


def test_edge_admittance_refuses_a_medium_with_no_real_admittance():
    eps, mu = np.diag([2.0, -3.0, 5.0]), np.eye(3)
    et, en, nt, nn = _in_section(eps, mu)
    with pytest.raises(ValueError, match="Absorbing 'side'"):
        edge_admittance(
            et[None],
            np.array([en]),
            nt[None],
            np.array([nn]),
            np.array([[1.0, 0.0]]),
            where="Absorbing 'side'",
        )


def _guide_with_absorbing_walls(materials):
    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.Absorbing("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def test_isotropic_absorbing_edges_keep_their_scalar_admittance():
    model = _guide_with_absorbing_walls(
        {"slab": gfem.Material(eps_r=4.0, mu_r=2.0), "air": gfem.Material()}
    )
    s = port_section(model, model.ports[0])
    y = s.absorbing_admittance
    assert y.shape == (len(s.absorbing_edges), 2)
    # bit for bit the mean-diagonal value of the isotropic medium beside the edge
    owner = _edge_owners(s)
    eps = np.trace(s.eps_tt[owner], axis1=1, axis2=2).real / 2
    mu = 2 / np.trace(s.nu_tt[owner], axis1=1, axis2=2).real
    np.testing.assert_array_equal(y[:, 0], np.sqrt(eps / mu))
    np.testing.assert_array_equal(y[:, 1], y[:, 0])
    assert set(np.round(y[:, 0], 12)) <= {1.0, round(np.sqrt(2.0), 12)}


def test_anisotropic_absorbing_edges_take_the_admittance_along_each_edge():
    # the walls are at x = const and y = const; eps and mu diagonal
    eps, mu = (2.0, 3.0, 5.0), (1.5, 1.2, 1.1)
    model = _guide_with_absorbing_walls(
        {
            "slab": gfem.Material(eps_r=eps, mu_r=mu),
            "air": gfem.Material(eps_r=eps, mu_r=mu),
        }
    )
    s = port_section(model, model.ports[0])
    p = s.to_global(s.points[s.absorbing_edges])
    along_y = np.abs(p[:, 0, 0] - p[:, 1, 0]) < 1e-12  # edge normal x
    assert along_y.any()
    assert (~along_y).any()
    y = s.absorbing_admittance
    np.testing.assert_allclose(
        y[along_y],
        np.broadcast_to([np.sqrt(3.0 / 1.1), np.sqrt(5.0 / 1.2)], y[along_y].shape),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        y[~along_y],
        np.broadcast_to([np.sqrt(2.0 / 1.1), np.sqrt(5.0 / 1.5)], y[~along_y].shape),
        rtol=1e-12,
    )


def _edge_owners(s):
    """The triangle beside each absorbing edge."""
    keys = np.sort(s.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    want = np.sort(s.absorbing_edges, axis=1)
    return np.array([np.flatnonzero((keys == e).all(axis=1))[0] // 3 for e in want])
