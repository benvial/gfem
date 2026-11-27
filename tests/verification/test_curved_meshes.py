r"""Curved walls: a mesh with mid-edge nodes on the wall converges as $h^4$.

Each case solves one gmsh mesh written at order 1 (straight facets) and at
order 2 (the same corners, mid-edge nodes on the CAD surface) against a closed
form. The curved error is bounded above by $K h^r$ and falls at a slope above
a floor; the same bound rejects the faceted twin, so a curved solve that read
straight facets would fail. $h$ is gmsh's ``MeshSizeMax`` (um for the coax, mm
for the sphere).

Tolerance derivation, common to every case. Each bound is 3 times the largest
measured $|e| / h^r$ over the three sizes: the constant scatters by at most
1.7x across sizes (the pre-asymptotic rate and the mesh-to-mesh variation of
gmsh), which stands in for the variation across gmsh versions and platforms.
Slope floors are 0.5 to 0.7 below the measured least-squares slope. Contrast
factors are 3 to 5 times inside the measured ratios. Round-off is far below
all of it: direct solves reach about $10^{-12}$ relative and the eigen
tolerance is $10^{-10}$, four decades under the smallest bound ($6 \times
10^{-6}$ at the coax's finest size), so no comparison is near a bit.
"""

import math

import curved_cases as cc
import ngsolve as ngs
import numpy as np
import pytest

import gfem
from gfem._engine.mesh import to_ngsolve

SIZES = (0.2, 0.14, 0.1)


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """``twins(case, h)``: the (order 1, order 2) files, meshed once per call."""
    pytest.importorskip("gmsh")
    directory = tmp_path_factory.mktemp("curved")
    cache = {}

    def get(case, h):
        if (case, h) not in cache:
            cache[case, h] = cc.write_twins(case, h, directory)
        return cache[case, h]

    return get


def _slope(h, e):
    """Least-squares slope of log|e| against log h."""
    return np.polyfit(np.log(h), np.log(np.abs(e)), 1)[0]


def _coax_run(twins, order, sizes=SIZES):
    straight, curved = [], []
    for h in sizes:
        p1, p2 = twins("coax", h)
        straight.append(cc.coax_error(p1, order))
        curved.append(cc.coax_error(p2, order))
    return np.abs(straight), np.abs(curved)


def test_t1_coax_capacitance_at_order_2_is_h_cubed_curved_and_h_squared_straight(
    twins,
):
    # Measured curved |e| / h^3 = 9.6e-4, 9.5e-4, 9.3e-4 (slope 3.03, rising to
    # 4 at finer sizes: pre-asymptotic), faceted slope 1.93 and a curved/faceted
    # ratio of 310 to 680. Bounds: 3e-3 h^3 (3.1x), slope floor 2.5, faceted
    # ceiling 2.5, contrast 100 (3x inside 310).
    straight, curved = _coax_run(twins, 2)
    h = np.array(SIZES)
    assert (curved <= 3e-3 * h**3).all(), curved / h**3
    assert _slope(h, curved) >= 2.5
    assert _slope(h, straight) <= 2.5
    assert (curved <= straight / 100).all(), straight / curved
    # the same bound rejects the faceted twin
    assert (straight > 3e-3 * h**3).all()


def test_t2_coax_capacitance_at_order_3_is_held_to_h_to_the_fourth_by_the_wall(twins):
    # Measured curved |e| / h^4 = 7.8e-4, 6.0e-4, 7.7e-4 (the quadratic wall
    # caps order 3 at h^4: the field would give h^6), slope 4.04, ratio 2000 to
    # 8400. Bounds: 2.5e-3 h^4 (3.2x), slope floor 3.3, contrast 500.
    straight, curved = _coax_run(twins, 3)
    h = np.array(SIZES)
    assert (curved <= 2.5e-3 * h**4).all(), curved / h**4
    assert _slope(h, curved) >= 3.3
    assert (curved <= straight / 500).all(), straight / curved
    assert (straight > 2.5e-3 * h**4).all()


def test_t3_sphere_tm11_resonance_at_order_2_converges_as_h_to_the_fourth(twins):
    # Measured curved |e| / h^4 = 0.0175, 0.0158, 0.0187, slope 3.91; faceted
    # slope 1.98 and 0.14 mm ratio 335. Bounds: 0.06 h^4 (3.2x), slope floor
    # 3.3, contrast 100 at 0.14 mm.
    curved, straight = [], {}
    for h in SIZES:
        p1, p2 = twins("sphere", h)
        curved.append(abs(cc.sphere_errors(p2, 2, cc.TM_PEC, [cc.K_TM11], 1)[0]))
        if h == 0.14:
            straight[h] = abs(cc.sphere_errors(p1, 2, cc.TM_PEC, [cc.K_TM11], 1)[0])
    h = np.array(SIZES)
    curved = np.array(curved)
    assert (curved <= 0.06 * h**4).all(), curved / h**4
    assert _slope(h, curved) >= 3.3
    assert curved[1] <= straight[0.14] / 100, straight[0.14] / curved[1]
    assert straight[0.14] > 0.06 * 0.14**4


def test_t4_sphere_tm21_and_te11_resonances_at_order_2_beat_the_faceted_twin(twins):
    # At 0.14 mm measured curved errors 9.0e-6 (TM21) and 4.5e-5 (TE11), ratios
    # to faceted 213 and 47. Bounds: 3e-5 and 1.5e-4 (3.3x), contrast 10 (4.7x
    # inside the smaller ratio).
    p1, p2 = twins("sphere", 0.14)
    ks = [cc.K_TM21, cc.K_TE11]
    curved = np.abs(cc.sphere_errors(p2, 2, cc.TE_PEC, ks, 2))
    straight = np.abs(cc.sphere_errors(p1, 2, cc.TE_PEC, ks, 2))
    assert curved[0] <= 3e-5, curved
    assert curved[1] <= 1.5e-4, curved
    assert (curved <= straight / 10).all(), straight / curved
    assert straight[0] > 3e-5
    assert straight[1] > 1.5e-4


# --- the geometry floor of order 3 ---------------------------------------------


def _deficit(path):
    """$(V - V_\\text{mesh}) / (3 V)$ of the octant read from ``path``.

    A wall error acts on the sphere as a change of radius, so a resonance moves
    by that much (the volume integral is NGSolve's, on the elements the solve
    uses).
    """
    # read as metres: the ratio is scale free and the unit radius is 1
    built = to_ngsolve(gfem.Mesh.from_gmsh(path, length_unit="m"))
    volume = ngs.Integrate(1, built.mesh) * built.scale**3
    exact = math.pi / 6
    return (exact - volume) / (3 * exact)


@pytest.mark.reference
def test_t5_order_3_is_set_by_the_wall_not_the_field(twins):
    # Sphere TM11 at h = 0.14, 0.1, 0.07 mm and coax at 0.1, 0.07, 0.05 um, order
    # 3, up to 220k dofs. Measured: curved slope 4.1 to 4.4 (sphere) and
    # 4.0 to 4.5 (coax), inside [3.3, 4.8] where the field's h^6 would be
    # above 5; sphere error over its mesh's deficit 0.94 to 1.05 (bound 15 %,
    # 5x the 3 % measured); faceted order 3 within 2.7 % of faceted order 2
    # (bound 5 %), because both are the same facet error.
    sphere_h = (0.14, 0.1, 0.07)
    errors = []
    for h in sphere_h:
        p1, p2 = twins("sphere", h)
        e3 = cc.sphere_errors(p2, 3, cc.TM_PEC, [cc.K_TM11], 1)[0]
        errors.append(abs(e3))
        deficit = _deficit(p2)
        assert abs(e3 - deficit) <= 0.15 * abs(deficit), (h, e3, deficit)
        f3 = cc.sphere_errors(p1, 3, cc.TM_PEC, [cc.K_TM11], 1)[0]
        f2 = cc.sphere_errors(p1, 2, cc.TM_PEC, [cc.K_TM11], 1)[0]
        assert abs(f3 - f2) <= 0.05 * abs(f2), (h, f3, f2)
    assert 3.3 <= _slope(np.array(sphere_h), errors) <= 4.8

    coax_h = (0.1, 0.07, 0.05)
    straight3, curved3 = _coax_run(twins, 3, coax_h)
    straight2, _ = _coax_run(twins, 2, coax_h)
    assert 3.3 <= _slope(np.array(coax_h), curved3) <= 4.8
    assert (abs(straight3 - straight2) <= 0.05 * straight2).all()
