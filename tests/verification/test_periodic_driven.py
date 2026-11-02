"""Bloch driven solves: the C1 line two-port against its closed form.

C1 is a lossy parallel-plate TEM line periodic along x, with two 50 ohm
lumped ports on interior sheets. Its open-circuit impedance matrix at Bloch
wave number k is `periodic_cases.bloch_line_z`, so S = (Z - R)(Z + R)^-1.
At k != 0 S is not symmetric, and S(+k) differs from S(-k) by far more than
the discretization error, which pins the sign of the Bloch phase.
"""

import numpy as np
import periodic_cases as pc
import pytest

import gfem

#: k (rad/m) of each C1 solve: +k, -k and 0.
WAVE_NUMBERS = (pc.C1_K[0], -pc.C1_K[0], 0.0)


@pytest.fixture(scope="module")
def c1_mesh(tmp_path_factory):
    return pc.c1_mesh(tmp_path_factory.mktemp("c1"))


def _exact(k):
    eps = pc.C1_EPS * (1 - 1j * pc.C1_TAN_DELTA)
    z = pc.bloch_line_z(pc.C1_FREQUENCY, k, eps)
    eye = 50.0 * np.eye(2)
    return (z - eye) @ np.linalg.inv(z + eye)


@pytest.fixture(scope="module", params=["direct", "bddc_gmres"])
def c1_s(request, c1_mesh):
    """S of C1 at each of `WAVE_NUMBERS`, by one linear solver."""
    options = gfem.SolverOptions(order=pc.C1_ORDER, linear_solver=request.param)
    out = {}
    for k in WAVE_NUMBERS:
        result = gfem.solve(
            pc.c1(c1_mesh),
            gfem.Driven(pc.C1_FREQUENCY, wave_vector=(k, 0.0, 0.0)),
            options=options,
        )
        (out[k],) = np.asarray(result.s.values)
    return out


@pytest.mark.parametrize("k", WAVE_NUMBERS, ids=["+k", "-k", "0"])
def test_bloch_line_s_matches_the_closed_form(c1_s, k):
    assert np.abs(c1_s[k] - _exact(k)).max() <= pc.C1_S_TOL


def test_bloch_line_s_pins_the_sign_of_the_phase(c1_s):
    k = pc.C1_K[0]
    s = c1_s[k]
    # the field E(x + a) = exp(-i k a) E(x): -k would be another line
    assert abs(s[1, 0] - _exact(-k)[1, 0]) > 0.5
    assert abs(s[1, 0] - s[0, 1]) > 0.5
