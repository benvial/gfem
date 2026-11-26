r"""The suspended silicon wire of Qiu et al. (2013) and Rakich et al. (2012).

A 315 x 283.5 nm silicon wire in air (n = 3.5, TE-like fundamental mode at
1.552 um, :math:`E = 170` GPa, :math:`\nu = 0.28`, :math:`\rho = 2329`,
:math:`Q_m = 1000`, photoelastic constants :math:`(-0.09, 0.017, -0.051)` on the
cubic axes along x and y). Its mode E2, the breathing mode at
:math:`q = 0`, has a forward Brillouin gain of 0.44e4 (radiation pressure,
here the moving-boundary part), 0.42e4 (electrostriction, the photoelastic part)
and 1.72e4 1/(W m) in total. The 300 x 280 nm wire of Rakich et al. has its E2 at
13.0 GHz.

E2 is found by symmetry, not by index: the one mode, among those nearest
12.5 GHz, even under both mirrors (:math:`s_x = s_y = +1`) with no :math:`u_z`.
The gain follows from the coupling as
:math:`G_B = 4 Q_m \tilde g_0^2/(\hbar\omega v_g^2\,\mathrm{Re}\,\Omega)`,
:math:`v_g = c/n_g`, per part by squaring that part's :math:`\tilde g_0`.
"""

import numpy as np
import optomechanics_cases as oc
import pytest
import scipy.constants as sc

import gfem

pytestmark = pytest.mark.verification

FREQUENCY = sc.c / oc.WIRE_WAVELENGTH
QM = 1000.0
#: Published gains, 1/(W m): moving boundary (radiation pressure), photoelastic
#: (electrostriction), total.
PUBLISHED = {"moving_boundary": 0.44e4, "photoelastic": 0.42e4, "total": 1.72e4}
#: Mesh B of the probe: wire 0.04 um, air 0.25 um, mechanical 0.02 um at order 3.
H_WIRE, H_AIR, H_MECH = 0.04, 0.25, 0.02
#: Gate on the gains, 6 %: twice the 3.0 % measured on this mesh (parts and
#: total: -1.6 %, -3.0 %, -2.3 %; the finer mesh gives +0.4 %, -0.6 %, -0.1 %, so
#: the error is discretization, from below), and above the +-1.2 % of the
#: paper's two printed digits (0.42e4 +- 0.005e4).
GAIN_GATE = 0.06
#: Gate on the 300 x 280 nm frequency, 1 %: the paper's three digits (13.0 +-
#: 0.05 GHz, 0.4 %) with a margin; measured 12.995 GHz, 0.04 %.
FREQUENCY_GATE = 0.01
PUBLISHED_RAKICH = 13.0e9


def _e2(result, width, height):
    """The number of the one mode even under both mirrors without u_z."""
    found = []
    for k in range(result.n_modes):
        sx, sy, uz = oc.parities(result.displacement(k), width, height)
        if sx > 0.99 and sy > 0.99 and uz < 1e-6:
            found.append(k)
    assert len(found) == 1, found
    return found[0]


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("wire")


def test_qiu_gain(tmp):
    width, height = 0.315, 0.2835
    optical = oc.wire_optical(oc.wire_section(tmp / "o.msh", width, height, H_WIRE, H_AIR))
    mech = oc.wire_mechanical(tmp / "m.msh", width, height, H_MECH)
    result = gfem.solve(
        mech,
        gfem.MechanicalEigenmode(12.5e9, n_modes=6, wavenumber=0.0),
        options=gfem.SolverOptions(order=3),
    )
    k = _e2(result, width, height)
    # the breathing mode lies in the band the paper's modes do (12.56 GHz here)
    assert 12.0e9 < float(result.frequency[k]) < 13.1e9
    c = gfem.optomechanical_coupling(
        optical, result, port="p", frequency=FREQUENCY, mechanical_modes=k
    )
    n_g = float(np.real(c.overlap.n_g))
    omega = 2 * np.pi * FREQUENCY
    big_omega = float(np.real(c.omega_m[0]))
    v_g = sc.c / n_g

    def gain(g0):
        return 4 * QM * abs(complex(g0)) ** 2 / (sc.hbar * omega * v_g**2 * big_omega)

    got = {
        "moving_boundary": gain(c.moving_boundary[0]),
        "photoelastic": gain(c.photoelastic[0]),
        "total": gain(c.total[0]),
    }
    for name, value in got.items():
        assert value == pytest.approx(PUBLISHED[name], rel=GAIN_GATE), (name, got)


def test_rakich_frequency(tmp):
    width, height = 0.300, 0.280
    mech = oc.wire_mechanical(tmp / "r.msh", width, height, H_MECH)
    result = gfem.solve(
        mech,
        gfem.MechanicalEigenmode(13.0e9, n_modes=6, wavenumber=0.0),
        options=gfem.SolverOptions(order=3),
    )
    k = _e2(result, width, height)
    assert float(result.frequency[k]) == pytest.approx(
        PUBLISHED_RAKICH, rel=FREQUENCY_GATE
    )
