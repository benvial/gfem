"""The r8 morph scan: how far a CPW sidewall travels before the mesh degrades.

Mesh: driven/r8, the IHP TopMetal2 coplanar line (signal 20 um wide at
|y| < 10 um, 15 um gaps, metal 11.03-14.03 um in z). The +y sidewall of the
signal conductor moves by a * cos^2(pi x / 600 um) over |x| < 300 um, into
the gap (a > 0, the conductor grows) or into the metal (a < 0). Ports are
held, every other surface slides. Expected (default stiffened elastic
extension): the quality warning first fires at 11 +/- 0.5 um, the first
inverted element appears at 14 +/- 0.5 um, nothing inverts down to -15 um,
and every planar surface keeps its nodes on its plane.
"""

import os
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest

import gfem
from gfem.shape import boundary_morph, quality

UM = 1e-6
AMPLITUDES = np.round(np.arange(0.5, 15.01, 0.5), 2)


@pytest.fixture(scope="module")
def r8():
    ref = os.environ.get("GFEM_REFERENCE_DIR")
    if not ref:
        pytest.skip("GFEM_REFERENCE_DIR not set (needs the driven/r8 reference mesh)")
    mesh = gfem.Mesh.from_gmsh(Path(ref) / "driven" / "r8" / "palace.msh")
    return gfem.Model(
        mesh,
        materials={
            "sio2": gfem.Material(eps_r=4.1),
            "sin": gfem.Material(eps_r=7.5, tan_delta=1e-3),
            "air": gfem.Material(),
        },
        excluded=(gfem.enclosed_by("topmetal2_xy", "topmetal2_z"),),
        boundaries=[
            gfem.Absorbing(s, order=2) for s in ("sio2__None", "sin__None", "air__None")
        ],
        ports=[
            gfem.LumpedPort("P1", ("P1_E0", "P1_E1"), ("+y", "-y")),
            gfem.LumpedPort("P2", ("P2_E0", "P2_E1"), ("-y", "+y")),
        ],
    )


def _signal_wall(p):
    """The signal conductor's +y sidewall: y = 10 um, z in the metal layer."""
    return (
        (np.abs(p[:, 1] - 10 * UM) < 1e-12)
        & (p[:, 2] > 11 * UM)
        & (p[:, 2] < 14.1 * UM)
    )


@pytest.fixture(scope="module")
def morph(r8):
    return boundary_morph(r8, "topmetal2_z", where=_signal_wall)


def _bump(morph):
    x = morph.points[:, 0]
    return np.where(np.abs(x) < 300 * UM, np.cos(np.pi * x / (600 * UM)) ** 2, 0.0)


def _outcome(morph, a):
    """'error', 'warning' or 'ok' for amplitude a (um)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            morph(a * UM * _bump(morph))
        except gfem.MeshQualityError:
            return "error"
    return (
        "warning"
        if any(w.category is gfem.MeshQualityWarning for w in caught)
        else "ok"
    )


def test_the_signal_wall_moves_along_plus_y(morph):
    assert morph.n_params > 100
    np.testing.assert_allclose(morph.normals[:, 1], 1.0, atol=1e-12)


def test_warning_at_11um_and_error_at_14um_into_the_gap(morph):
    outcomes = {a: _outcome(morph, a) for a in AMPLITUDES}
    first_warning = min(a for a, o in outcomes.items() if o != "ok")
    first_error = min(a for a, o in outcomes.items() if o == "error")
    assert first_warning == pytest.approx(11.0, abs=0.5)
    assert first_error == pytest.approx(14.0, abs=0.5)


def test_nothing_inverts_moving_into_the_metal(morph, r8):
    moved = jax.jit(lambda t: morph(t).nodes)  # no check under jit
    mesh = r8.mesh
    treedef = jax.tree_util.tree_structure(mesh)
    for a in -AMPLITUDES:
        nodes = moved(a * UM * _bump(morph))
        q = np.asarray(quality(jax.tree_util.tree_unflatten(treedef, [nodes])))
        assert (q > 0).all(), a


def test_slides_keep_every_planar_surface_on_its_plane(morph, r8):
    mesh = r8.mesh
    x0 = mesh.imported_nodes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.MeshQualityWarning)
        moved = np.asarray(morph(13 * UM * _bump(morph)).nodes)
    shift = moved - x0
    scale = np.abs(shift).max()
    params = np.flatnonzero(_signal_wall(x0))
    checked = 0
    for surface in mesh.surfaces:
        tris = mesh.triangles[mesh.elements(surface)]
        n = np.cross(x0[tris[:, 1]] - x0[tris[:, 0]], x0[tris[:, 2]] - x0[tris[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        # group the surface's triangles by plane: normal (up to sign) and offset
        n *= np.where(n[np.arange(len(n)), np.abs(n).argmax(axis=1)] < 0, -1, 1)[
            :, None
        ]
        offset = np.einsum("ij,ij->i", n, x0[tris[:, 0]])
        key = np.round(np.column_stack([n, offset / UM]), 6)
        for plane in np.unique(key, axis=0):
            members = (key == plane).all(axis=1)
            nodes = np.setdiff1d(np.unique(tris[members]), params)
            off = np.abs(shift[nodes] @ plane[:3])
            assert off.max(initial=0) <= 1e-12 * scale, (surface, plane)
            checked += 1
    assert checked > 20
