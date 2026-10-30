"""Nonlinear eigenmode problems: a conductor or a second-order absorbing wall.

A `Conductor`'s surface impedance grows as the square root of the
frequency and the second-order absorbing term as its inverse, so the
operator is no longer a polynomial in the frequency. The solve seeds the
modes with those factors frozen at the target and refines each one on the
true operator: the refined frequency is a root of the model's operator, so
it does not depend on the target, whereas a frozen seed's Q is off by the
detuning.

- Conductor: the parallel-plate line of the eigenmode unit tests (PEC
  bottom plate, PMC side walls, open ends) with a conductor as its top
  plate keeps its TEM field to first order in the loss, with Q = 2 h /
  delta for one lossy plate of skin depth delta (h the plate spacing).
- Second-order absorbing: a TE10 guide (PEC walls, 10 mm wide) shorted at
  x = 0, filled with eps_r = 10 up to x = a and with vacuum beyond, ended
  by an absorbing wall at x = L. On a TE10 wave the textbook second-order
  condition is the first-order one with k replaced by k - k_t^2 / (2 k)
  (k_t = pi / width), so its reflection is R = (beta - k_eff) / (beta +
  k_eff), and the guide's modes are the roots of the transmission-line
  condition at the dielectric step.
"""

import cmath
import math

import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm
from eigen_cases import (
    EPS,
    F1,
    SIGMA,
    H,
    conductor_line,
    line_mesh,
    terminated_line,
)

import gfem

#: Targets 25 % below and 30 % above the mode: a seed frozen there has a Q
#: off by about 13 %.
TARGETS = (0.75 * F1, 1.3 * F1)
TIGHT = gfem.SolverOptions(eigen_tolerance=1e-10)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return line_mesh(tmp_path_factory.mktemp("line"), (20, 1, 2))


def _skin_depth(omega):
    return math.sqrt(2 / (const.mu_0 * SIGMA * omega))


@pytest.fixture(scope="module")
def conductor_modes(mesh):
    model = conductor_line(mesh)
    return [gfem.solve(model, gfem.Eigenmode(target=f), options=TIGHT) for f in TARGETS]


@pytest.mark.parametrize("thickness", [None, 2e-6, 1e-8])
def test_nonpolynomial_terms_supply_their_analytic_derivative(mesh, thickness):
    # dtheta/domega against central differences along the real and the
    # imaginary axis at a complex frequency: equal only for an analytic theta
    from gfem._engine.driven import driven_operator

    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.Conductor("top", SIGMA, thickness),
            gfem.Absorbing("port1", order=2),
        ],
    )
    pieces = [
        p
        for p in driven_operator(model, 1).pieces
        if p.frequency_class == "nonpolynomial"
    ]
    names = sorted(p.name for p in pieces)
    assert names[0] == "absorbing order 2"
    assert names[1].startswith("conductor top")
    omega = 2 * math.pi * F1 * (1 + 0.05j)
    h = 1e-5 * abs(omega)
    for p in pieces:
        along_real = (p.theta(omega + h) - p.theta(omega - h)) / (2 * h)
        along_imag = (p.theta(omega + 1j * h) - p.theta(omega - 1j * h)) / (2j * h)
        exact = p.d_theta(omega)
        assert abs(along_real - exact) <= 1e-8 * abs(exact)
        assert abs(along_imag - exact) <= 1e-8 * abs(exact)


def test_conductor_plate_gives_the_transmission_line_q(conductor_modes):
    for result in conductor_modes:
        omega = complex(result.omega[0])
        assert float(result.error[0]) <= 1e-10
        np.testing.assert_allclose(omega.real / (2 * math.pi), F1, rtol=1e-2)
        q = 2 * H * 1e-3 / _skin_depth(omega.real)
        np.testing.assert_allclose(float(result.q[0]), q, rtol=2e-2)


def test_refined_conductor_mode_does_not_depend_on_the_target(conductor_modes):
    low, high = (complex(r.omega[0]) for r in conductor_modes)
    assert abs(low - high) <= 1e-12 * abs(high)


# --- second-order absorbing wall ---------------------------------------------------

CORE, TAIL, WIDTH, HEIGHT = 3.0, 3.0, 10.0, 1.0  # mm
EPS_CORE = 10.0
F_GUIDE = 24.15e9  # the mode near 24 GHz, Q about 10 (12 at first order)


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    length = CORE + TAIL
    edges = (
        np.concatenate([np.linspace(0, CORE, 7), np.linspace(CORE, length, 7)[1:]]),
        np.linspace(0, WIDTH, 5),
        np.linspace(0, HEIGHT, 2),
    )

    def surface(c):
        on = [abs(c[0]), abs(c[0] - length)]
        if min(on) < 1e-9:
            return "short" if on[0] < 1e-9 else "end"
        if min(abs(c[1]), abs(c[1] - WIDTH), abs(c[2]), abs(c[2] - HEIGHT)) < 1e-9:
            return "wall"
        return ""

    path = tmp_path_factory.mktemp("guide") / "guide.msh"
    sm.write_msh22(
        path,
        sm.structured(edges, lambda c: "core" if c[0] < CORE else "tail", surface),
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _guide(mesh, order):
    return gfem.Model(
        mesh,
        materials={
            "core": gfem.Material(eps_r=EPS_CORE),
            "tail": gfem.Material(eps_r=1.0),
        },
        boundaries=[
            gfem.PEC("short"),
            gfem.PEC("wall"),
            gfem.Absorbing("end", order=order),
        ],
    )


def _guide_modes(mesh, order, targets):
    model = _guide(mesh, order)
    return [
        complex(gfem.solve(model, gfem.Eigenmode(target=f), options=TIGHT).omega[0])
        for f in targets
    ]


def _guide_root(order, omega):
    """The guide's closed-form mode nearest ``omega`` (Newton in k)."""
    kt = math.pi / (WIDTH * 1e-3)
    a, d = CORE * 1e-3, TAIL * 1e-3

    def residual(k):
        b1 = cmath.sqrt(k * k * EPS_CORE - kt * kt)
        b2 = cmath.sqrt(k * k - kt * kt)
        k_eff = k - (kt * kt / (2 * k) if order == 2 else 0)
        rho = (b2 - k_eff) / (b2 + k_eff) * cmath.exp(-2j * b2 * d)
        return b1 * cmath.cos(b1 * a) * (1 + rho) + 1j * b2 * cmath.sin(b1 * a) * (
            1 - rho
        )

    k = omega / const.c
    for _ in range(50):
        h = 1e-7 * abs(k)
        step = residual(k) / ((residual(k + h) - residual(k - h)) / (2 * h))
        k -= step
        if abs(step) < 1e-14 * abs(k):
            break
    return k * const.c


@pytest.mark.parametrize("order", [1, 2])
def test_absorbing_guide_modes_are_the_closed_form_roots(guide, order):
    (omega,) = _guide_modes(guide, order, (F_GUIDE,))
    exact = _guide_root(order, omega)
    np.testing.assert_allclose(omega.real, exact.real, rtol=1e-3)
    q, q_exact = (abs(w) / (2 * w.imag) for w in (omega, exact))
    np.testing.assert_allclose(q, q_exact, rtol=1e-2)


def test_second_order_absorbing_mode_does_not_depend_on_the_target(guide):
    low, high = _guide_modes(guide, 2, (0.98 * F_GUIDE, 1.02 * F_GUIDE))
    assert abs(low - high) <= 1e-12 * abs(high)
    # the curl term moves the mode: Q 9.8 against 12.1 at first order
    (first,) = _guide_modes(guide, 1, (F_GUIDE,))
    assert abs(first - high) > 1e-2 * abs(high)


# --- the seed's request ------------------------------------------------------------


@pytest.fixture
def seed_solves(monkeypatch):
    """What each seed solve asks for and spends.

    One ``(nev, solves)`` per Arnoldi attempt: the pairs the rule asks the
    eigen solver for, and the ``T(sigma)^-1`` solves the seed made.
    """
    import gfem._eigen.rule as eigen_rule
    from gfem._eigen.operator import EigenOperator

    solves = [0]
    solve = EigenOperator.solve

    def counted(self, b):
        solves[0] += 1
        return solve(self, b)

    shift_invert = eigen_rule.shift_invert
    seen = []

    def spy(op, project, nev, **kwargs):
        before = solves[0]
        out = shift_invert(op, project, nev, **kwargs)
        seen.append((nev, solves[0] - before))
        return out

    monkeypatch.setattr(EigenOperator, "solve", counted)
    monkeypatch.setattr(eigen_rule, "shift_invert", spy)
    return seen


def test_a_nonlinear_seed_asks_for_one_pair_beyond_n_modes(mesh, guide, seed_solves):
    # behind its modes a frozen seed has the linearization cluster: the one
    # pair beyond n_modes is the guard's slot; a damped linear seed asks for
    # twice its mirror-root allowance, as before
    for f in TARGETS:
        gfem.solve(conductor_line(mesh), gfem.Eigenmode(target=f), options=TIGHT)
    problem = gfem.Eigenmode(target=F_GUIDE, n_modes=2)
    gfem.solve(_guide(guide, 2), problem, options=TIGHT)
    gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(target=8e9))
    assert [nev for nev, _ in seed_solves] == [2, 2, 3, 6]


def test_a_nonlinear_seed_spends_few_shift_solves(mesh, seed_solves):
    # 38 and 21 measured; 115 and 151 when the seed asked for 6 pairs
    for f in TARGETS:
        gfem.solve(conductor_line(mesh), gfem.Eigenmode(target=f), options=TIGHT)
    assert len(seed_solves) == 2
    for _, solves in seed_solves:
        assert solves <= 50
