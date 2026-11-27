"""`cad_morph(curved=True)`: gmsh order-2 meshes, mid-edge design velocities and
the mid-edge motion of the extension, the refusal of a build-set element order,
and `remesh` keeping the order."""

import cad_cases as cc
import jax
import numpy as np
import pytest

import gfem
from gfem import _cad
from gfem.mesh import _TET_EDGES, LENGTH_UNITS

UM = LENGTH_UNITS["um"]
EPS = np.finfo(float).eps

#: The disk's mesh size, um: about 9 elements round the 0.4 um circle.
LC = 0.15


@pytest.fixture(scope="module")
def gmsh():
    # Inside a fixture, not at module level: collecting this file must not
    # import gmsh in jobs that have no system libraries for it.
    pytest.importorskip("gmsh")
    return _cad.import_gmsh()


@pytest.fixture(scope="module")
def cad(gmsh):  # noqa: ARG001
    return _cad.build_cad(cc.disk_build(LC), {"radius": cc.R0}, "um", curved=True)


@pytest.fixture(scope="module")
def morph(gmsh):  # noqa: ARG001
    return gfem.shape.cad_morph(
        cc.disk_build(LC), {"radius": cc.R0}, cc.make_model, curved=True
    )


def _mid_nodes(mesh):
    return np.unique(mesh._topology.volume_midnodes)


def test_the_curved_build_reads_a_second_order_mesh(cad):
    assert cad.mesh.curved
    assert cad.slide_midnodes.shape == (len(cad.slide_triangles), 3)
    assert len(_mid_nodes(cad.mesh)) > 0


def test_the_straight_build_is_unchanged(gmsh):  # noqa: ARG001
    cad = _cad.build_cad(cc.disk_build(LC), {"radius": cc.R0}, "um")
    assert not cad.mesh.curved
    assert cad.slide_midnodes is None


def test_mid_edge_nodes_of_the_wall_lie_on_the_cylinder(cad):
    x = np.asarray(cad.mesh.imported_nodes) / UM
    mids = _mid_nodes(cad.mesh)
    r = np.linalg.norm(x[:, :2] - 1, axis=1)
    # surface mid nodes strictly between the caps, within 0.01 um of the radius
    z = x[:, 2]
    between = (z > cc.Z0 + 0.01) & (z < cc.Z0 + cc.T0 - 0.01)
    wall = mids[(cad.dim[mids] == 2) & between[mids] & (np.abs(r[mids] - cc.R0) < 0.01)]
    assert len(wall) > 0
    # gmsh puts a boundary mid-edge node on the CAD to round-off of the
    # coordinates (a few ulp of 2 um, 4.4e-16 each) plus the kernel's
    # evaluation of the surface, measured below 1e-15; the match tolerance
    # 1e-9 of the shared cases is far above that and below a node's 1e-2 sag.
    np.testing.assert_allclose(r[wall], cc.R0, rtol=0, atol=cc.MATCH)


def test_mid_edge_design_velocities_are_the_analytic_ones(cad):
    # The closest point of a cylinder node moves radially by one unit per unit
    # and a cap node not at all, exact up to the closest-point search: the
    # tolerance of the straight-mesh velocity test (1e-9) holds for mid nodes.
    x = np.asarray(cad.mesh.imported_nodes) / UM
    v = cad.velocity.reshape(-1, 3, 1)[:, :, 0]
    np.testing.assert_allclose(v, cc.disk_velocity(x, cc.R0), rtol=0, atol=1e-9)


def test_interior_mid_edge_nodes_follow_their_chord(morph):
    mesh = morph.model.mesh
    t = mesh._topology
    w = np.asarray(morph._w).reshape(-1, 3)
    cad = _cad.build_cad(cc.disk_build(LC), {"radius": cc.R0}, "um", curved=True)
    inside = np.unique(t.volume_midnodes)
    inside = inside[cad.dim[inside] == 3]
    assert len(inside) > 100
    ends = {}
    for k, (a, b) in enumerate(_TET_EDGES):
        for m, i, j in zip(
            t.volume_midnodes[:, k],
            t.volume_elements[:, a],
            t.volume_elements[:, b],
            strict=True,
        ):
            ends[int(m)] = (int(i), int(j))
    a = np.array([ends[int(m)][0] for m in inside])
    b = np.array([ends[int(m)][1] for m in inside])
    # one addition and one halving of two displacements: at most one ulp of
    # their sum, 2 eps |W| (the 0.5 is exact); 4 eps is a margin of two.
    atol = 4 * EPS * np.abs(w).max()
    np.testing.assert_allclose(w[inside], 0.5 * (w[a] + w[b]), rtol=0, atol=atol)


def test_a_wall_mid_edge_node_keeps_the_cad_normal_velocity(morph):
    # P V_m + (I - P) chord has the normal part of V_m, one unit, along the
    # quadratic normal n_m; the true radial direction differs from n_m by at
    # most delta = 1e-3 rad (measured 4.9e-4 at this size), and the tangential
    # part of the chord mean is below l / (2 R) = 0.19 (neighbouring corners'
    # radial velocities differ by their chord angle), so the radial component
    # is 1 within delta^2 / 2 + 0.19 delta < 2e-4: 1e-3 with a margin of five.
    x = np.asarray(morph.model.mesh.imported_nodes)
    w = np.asarray(morph._w).reshape(-1, 3) / UM
    mids = _mid_nodes(morph.model.mesh)
    r = x[:, :2] / UM - 1
    rho = np.linalg.norm(r, axis=1)
    wall = mids[np.abs(rho[mids] - cc.R0) < cc.MATCH]
    z = x[wall, 2] / UM
    wall = wall[(z > cc.Z0 + 0.01) & (z < cc.Z0 + cc.T0 - 0.01)]
    assert len(wall) > 20
    radial = np.einsum("ni,ni->n", w[wall, :2], r[wall] / rho[wall, None])
    np.testing.assert_allclose(radial, 1.0, rtol=0, atol=1e-3)


@pytest.mark.parametrize("curved", [False, True])
def test_a_build_that_sets_the_element_order_is_refused(gmsh, curved):
    inner = cc.disk_build(LC)

    def build(radius):
        inner(radius)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)

    # gfem's own session is finalized on the refusal: the option dies with it
    with pytest.raises(ValueError, match=r"Mesh\.ElementOrder") as info:
        gfem.shape.cad_morph(build, {"radius": cc.R0}, cc.make_model, curved=curved)
    assert "curved=True" in str(info.value)


def test_the_options_are_set_around_the_mesh_and_restored(gmsh, monkeypatch):
    seen = []
    generate = gmsh.model.mesh.generate

    def spy(dim=-1):
        seen.append(
            (
                gmsh.option.getNumber("Mesh.ElementOrder"),
                gmsh.option.getNumber("Mesh.HighOrderOptimize"),
            )
        )
        return generate(dim)

    monkeypatch.setattr(gmsh.model.mesh, "generate", spy)
    # the user's session is open: options are left as found
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        _cad.build_cad(cc.disk_build(LC), {"radius": cc.R0}, "um", curved=True)
        assert seen[-1] == (2, 1)
        assert gmsh.option.getNumber("Mesh.ElementOrder") == 1
        assert gmsh.option.getNumber("Mesh.HighOrderOptimize") == 0
        # a build's own optimization level (2 to 4) is kept
        inner = cc.disk_build(LC)

        def build(radius):
            inner(radius)
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 3)

        _cad.build_cad(build, {"radius": cc.R0}, "um", curved=True)
        assert seen[-1] == (2, 3)
        _cad.build_cad(cc.disk_build(LC), {"radius": cc.R0}, "um")
        assert seen[-1][0] == 1
    finally:
        gmsh.finalize()


def test_curved_is_an_attribute_and_remesh_keeps_it(morph):
    assert morph.curved is True
    assert "curved=True" in repr(morph)
    again = morph.remesh({"radius": cc.R0 + 0.02})
    assert again.curved is True
    assert again.model.mesh.curved


def test_the_default_morph_is_straight(gmsh):  # noqa: ARG001
    straight = gfem.shape.cad_morph(
        cc.disk_build(0.25), {"radius": cc.R0}, cc.make_model
    )
    assert straight.curved is False
    assert not straight.model.mesh.curved
    assert "curved" not in repr(straight)


def test_curved_must_be_a_bool(gmsh):  # noqa: ARG001
    with pytest.raises(TypeError, match="curved"):
        gfem.shape.cad_morph(
            cc.disk_build(LC), {"radius": cc.R0}, cc.make_model, curved="yes"
        )


def test_the_gradient_through_a_curved_morph_matches_a_difference_on_the_mesh(morph):
    # The same mesh at x0 +- eps W: only the chain through the morph differs.
    # Round-off of J (direct solve, ~1e-13 relative) over 2 eps, and the cubic
    # truncation eps^2 J'''/6 with J'''/J' ~ 1 um^-2: with eps = 2^-9 um the
    # sum is 1e-13/2e-3 + 4e-6/6 < 1e-6; rtol 1e-5 leaves a margin of ten.
    eps = 2.0**-9

    def value(r):
        return cc.capacitance(morph.model.replace(mesh=morph({"radius": r})))

    grad = float(jax.grad(value)(cc.R0))
    difference = (value(cc.R0 + eps) - value(cc.R0 - eps)) / (2 * eps)
    assert abs(grad - float(difference)) < 1e-5 * abs(float(difference))
