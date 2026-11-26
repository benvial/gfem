r"""Mechanical eigenmodes of an isotropic section against closed forms, refined.

All sections are 2D, solved through the public surface
(`gfem.MechanicalEigenmode`) in an isotropic glass-like solid at order 4, with
the closed forms of `tests/mechanical_cases.py`. Relative errors of the
frequency, measured on the maxh = 0.125 um mesh (0.25 um in parentheses), and
the gates (each at least 25 times the measured error, to leave room for
another platform's mesh):

- anti-plane standing wave :math:`(1, 1)` of a free 1.7 x 1 um rectangle:
  :math:`1.1\times10^{-11}` (:math:`4.2\times10^{-9}`), gate :math:`10^{-9}`;
- shear horizontal modes SH0 and SH1 of the strip with sliding walls at
  :math:`q = 2\times10^6` rad/m: SH0 is exact in the discrete space
  (:math:`2\times10^{-15}`), SH1 :math:`3\times10^{-12}`
  (:math:`1.2\times10^{-9}`), gate :math:`10^{-9}`; the group velocity of SH0 is
  :math:`v_T` within :math:`2\times10^{-15}`, gate :math:`10^{-9}`, and SH1's is
  :math:`v_T^2 q/\omega` within :math:`10^{-9}`;
- Lamb modes S0 and A0 of a 1 um plate at the same :math:`q`:
  :math:`1.4\times10^{-13}` (:math:`1.8\times10^{-10}`) and
  :math:`1.0\times10^{-12}` (:math:`1.2\times10^{-9}`), gate :math:`10^{-9}`;
- the Rayleigh wave of a 4 um deep block at :math:`q = 2\pi/1` um, maxh = 0.1
  um: :math:`3.6\times10^{-9}` in frequency and :math:`6.6\times10^{-8}` in
  group velocity (:math:`v_R`, no dispersion); gates :math:`10^{-7}` and
  :math:`10^{-6}`. The block's depth contributes :math:`3\times10^{-9}` (5 um
  deep: :math:`1.3\times10^{-10}`), the rest is the mesh's.

The error of the four strip and plate modes falls at least 100 times from
maxh = 0.25 to 0.125 um (370 to 1300 measured), the rate of order 4
(:math:`h^{8}` for an eigenvalue, 256 in theory, less before the asymptote).

Runtime about 40 s on one thread.
"""

import functools

import mechanical_cases as mc
import numpy as np
import pytest

import gfem

Q = 2e6
SLIDING = {"left": "yz", "right": "yz"}
PLANE = {"left": "x", "right": "x"}
OPTIONS = gfem.SolverOptions(order=4, linear_solver="direct")


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """``solved(name, width, height, maxh, target, **problem)`` -> the result."""
    directory = tmp_path_factory.mktemp("mechanical")

    @functools.cache
    def solve(width, height, maxh, target, *, q, clamped):
        mesh = mc.rectangle(
            directory / f"{width}_{height}_{maxh}.msh", width, height, maxh
        )
        problem = gfem.MechanicalEigenmode(target, wavenumber=q, clamped=dict(clamped))
        return gfem.solve(mc.model(mesh), problem, options=OPTIONS)

    def run(width, height, maxh, target, *, q=0.0, clamped=None):
        return solve(
            width, height, maxh, target, q=q, clamped=tuple((clamped or {}).items())
        )

    return run


def _error(result, exact):
    return abs(float(result.frequency[0]) - exact) / exact


def test_the_standing_shear_wave_of_a_free_rectangle(solved):
    exact = mc.sh_standing(1, 1, 1.7, 1.0)
    assert _error(solved(1.7, 1.0, 0.125, exact), exact) < 1e-9


def test_the_shear_horizontal_modes_of_a_strip(solved):
    exact = mc.sh_strip(0, Q, 1.0)
    result = solved(1.7, 1.0, 0.125, exact, q=Q, clamped=SLIDING)
    assert _error(result, exact) < 1e-9
    assert float(result.group_velocity[0]) == pytest.approx(mc.V_T, rel=1e-9)
    exact = mc.sh_strip(1, Q, 1.0)
    result = solved(1.7, 1.0, 0.125, exact, q=Q, clamped=SLIDING)
    assert _error(result, exact) < 1e-9
    omega = 2 * np.pi * exact
    assert float(result.group_velocity[0]) == pytest.approx(
        mc.V_T**2 * Q / omega, rel=1e-9
    )


@pytest.mark.parametrize("kind", ["S", "A"])
def test_the_lamb_modes_of_a_plate(solved, kind):
    (exact,) = mc.lamb_roots(Q, 1.0, kind)
    assert _error(solved(0.5, 1.0, 0.125, exact, q=Q, clamped=PLANE), exact) < 1e-9


def test_the_rayleigh_wave_of_a_deep_block(solved):
    q = 2 * np.pi / (1.0 * mc.UM)
    v_r = mc.rayleigh_ratio() * mc.V_T
    exact = v_r * q / (2 * np.pi)
    clamped = {**PLANE, "bottom": "xyz"}
    result = solved(0.3, 4.0, 0.1, exact, q=q, clamped=clamped)
    assert _error(result, exact) < 1e-7
    assert float(result.group_velocity[0]) == pytest.approx(v_r, rel=1e-6)


@pytest.mark.parametrize(
    "case",
    [
        ("SH1", 1.7, 1.0, mc.sh_strip(1, Q, 1.0), SLIDING),
        ("S0", 0.5, 1.0, mc.lamb_roots(Q, 1.0, "S")[0], PLANE),
        ("A0", 0.5, 1.0, mc.lamb_roots(Q, 1.0, "A")[0], PLANE),
    ],
)
def test_the_error_falls_at_the_rate_of_order_four(solved, case):
    label, width, height, exact, clamped = case
    coarse = _error(solved(width, height, 0.25, exact, q=Q, clamped=clamped), exact)
    fine = _error(solved(width, height, 0.125, exact, q=Q, clamped=clamped), exact)
    assert coarse / fine >= 100, (label, coarse, fine)
