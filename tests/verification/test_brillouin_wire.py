r"""The backward Brillouin gain of the suspended silicon wire of Qiu et al. (2013).

The 315 x 283.5 nm silicon wire of the forward test, pump and Stokes in the
fundamental mode at 1.552 um, backward intramodal scattering at
:math:`q = 2\beta` (Qiu et al., Opt. Express 21, 31402 (2013), Sec. 3.2,
Fig. 3(b)). The modes E1, E2, E3, E4 are the ones even under both mirrors
(:math:`s_x = s_y = +1`, :math:`u_z` allowed), found by symmetry and ordered by
frequency. The published parts are the "electrostriction only" bar (the
photoelastic part here) and the "radiation pressure only" bar (the moving
boundary), in 1/(W m) at :math:`Q_m = 1000`; the text gives the totals of E1 and
E2 and the figure the total of E4. In E1 and E2 the two parts are destructive.
"""

import numpy as np
import optomechanics_cases as oc
import pytest
import scipy.constants as sc

import gfem

pytestmark = pytest.mark.verification

FREQUENCY = sc.c / oc.WIRE_WAVELENGTH
#: Wire 0.02 um, air 0.15 um, mechanical 0.01 um at order 3: twice as fine as
#: the forward test's mesh, because E1's photoelastic part (a small term of a
#: destructive pair) is 7 % high on the coarser one (579 against 538 here) and
#: is converged to 1.7 % here (529 on the wire 0.014, air 0.1, mechanical 0.007
#: mesh; every gated part and total moves at most 1.7 % between the two).
H_WIRE, H_AIR, H_MECH = 0.02, 0.15, 0.01
WIDTH, HEIGHT = 0.315, 0.2835
#: Published frequencies of E1, E2, E4, GHz (the figure's axis, in units of
#: 2 pi V_L / a, read by pixel).
PUBLISHED_F = {"E1": 13.99, "E2": 15.54, "E4": 22.70}
#: Published parts, 1/(W m): (photoelastic, moving boundary), then totals.
PUBLISHED_PARTS = {
    "E1": (530.0, 2750.0),
    "E2": (3530.0, 930.0),
    "E4": (1850.0, None),
}
PUBLISHED_TOTAL = {"E1": 890.0, "E2": 860.0, "E4": 1400.0}
#: Gate on the frequencies, 2 %: the measured -0.7 %, -0.8 %, -0.7 % are the
#: figure axis' digitization offset (-0.6 %) and the -0.2 % that gfem's own q
#: (0.456 % below the published 1.5 pi / a) gives; 2 % is 2.5x the largest.
#: Measured (GHz): E1 13.890, E2 15.414, E4 22.540 (-0.7 %, -0.8 %, -0.7 %).
FREQUENCY_GATE = 0.02
#: Gate on the parts, 10 %: the pixel read of a bar is +-30 1/(W m), 5.7 % of
#: the smallest gated one (530), the digitization bias is +1.7 %, the mesh error
#: is at most 3 % (forward test); their root sum of squares is 6.7 %.
#: Measured (photoelastic / moving boundary, 1/(W m)): E1 538 (+1.5 %) / 2757
#: (+0.2 %); E2 3583 (+1.5 %) / 868 (-6.6 %); E4 1797 (-2.9 %) / 29. E2's moving
#: boundary is 863-870 on every mesh and q tried (Qiu's own q included), 7 % under
#: the published bar: the +-30 figure read explains 3.2 %, the rest is unattributed.
PART_GATE = 0.10
#: Gate on the totals, 15 %: the total is the square of the sum of the part
#: amplitudes, so an error e of a part amplitude becomes 2 e (a + b) / (a - b),
#: 5.1 e for E1 and 6.2 e for E2; 15 % allows e of about 2.5 %, the mesh error
#: of the forward test, plus the digitization of the text values' 0.5 % rounding.
#: Measured totals: E1 859 (-3.5 % of the text's 890), E2 924 (+7.4 % of 860),
#: E4 1366 (-2.4 % of the digitized 1400); E3 (not gated) 281.
TOTAL_GATE = 0.15
#: E4's moving boundary is not read off the figure (a bar under 50 cannot be told
#: from zero); it must stay a tenth of the photoelastic part.
#: Measured: 29 / 1797 = 0.016. The destructive sign Re(dn_MB conj(dn_PE)) is
#: -90, -131, -17 for E1, E2, E4 (-10 for E3), in the units of the squared overlaps.
E4_RATIO = 0.1


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wire")
    optical = oc.wire_optical(
        oc.wire_section(tmp / "o.msh", WIDTH, HEIGHT, H_WIRE, H_AIR)
    )
    mech = oc.wire_mechanical(tmp / "m.msh", WIDTH, HEIGHT, H_MECH)
    mode = gfem.solve(optical, gfem.PortMode("p", FREQUENCY))
    q = 2 * float(np.imag(mode.gamma[0]))
    result = gfem.solve(
        mech,
        gfem.MechanicalEigenmode(18e9, n_modes=24, wavenumber=q),
        options=gfem.SolverOptions(order=3),
    )
    gain = gfem.brillouin_gain(optical, result, port="p", frequency=FREQUENCY)
    return q, result, gain


def _e_class(result):
    """Numbers of the modes even under both mirrors (u_z allowed), by frequency."""
    found = []
    for k in range(result.n_modes):
        sx, sy, _ = oc.parities(result.displacement(k), WIDTH, HEIGHT)
        if sx > 0.99 and sy > 0.99:
            found.append(k)
    return sorted(found, key=lambda k: float(result.frequency[k]))


def test_qiu_backward(case):
    q, result, gain = case
    assert gain.overlap.direction == "backward"
    # gfem's own beta: 1.4892e7 rad/m, 0.456 % below the published 1.5 pi / a
    assert q == pytest.approx(1.4892e7, rel=1e-3)
    e_class = _e_class(result)
    assert len(e_class) >= 4, e_class
    names = dict(zip(("E1", "E2", "E3", "E4"), e_class[:4], strict=True))
    row = {int(m): i for i, m in enumerate(gain.modes)}
    g_pe = np.asarray(gain.photoelastic)
    g_mb = np.asarray(gain.moving_boundary)
    g_tot = np.asarray(gain.total)
    dn_pe = np.asarray(gain.overlap.photoelastic)
    dn_mb = np.asarray(gain.overlap.moving_boundary)
    got = {}
    for name, k in names.items():
        i = row[k]
        got[name] = {
            "f": float(result.frequency[k]) / 1e9,
            "pe": float(g_pe[i]),
            "mb": float(g_mb[i]),
            "total": float(g_tot[i]),
            "sign": float(np.real(dn_mb[i] * np.conj(dn_pe[i]))),
        }
    for name, f in PUBLISHED_F.items():
        assert got[name]["f"] == pytest.approx(f, rel=FREQUENCY_GATE), (name, got)
    for name, (pe, mb) in PUBLISHED_PARTS.items():
        assert got[name]["pe"] == pytest.approx(pe, rel=PART_GATE), (name, got)
        if mb is not None:
            assert got[name]["mb"] == pytest.approx(mb, rel=PART_GATE), (name, got)
    for name, total in PUBLISHED_TOTAL.items():
        assert got[name]["total"] == pytest.approx(total, rel=TOTAL_GATE), (name, got)
    assert got["E4"]["mb"] < E4_RATIO * got["E4"]["pe"], got
    for name in ("E1", "E2", "E4"):
        assert got[name]["sign"] < 0, (name, got)
