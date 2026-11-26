r"""Physical fields and Kerr overlaps of a bent port mode, and what it refuses."""

from __future__ import annotations

import copy
import dataclasses

import bend_cases as bc
import jax
import numpy as np
import pytest

import gfem

N2 = 1.0  # m^2/W: $\gamma_\mathrm{NL}$ is linear in it, so the value is a scale
A = bc.WR90_A * 1e-3  # m


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = bc.wr90_mesh(tmp_path_factory.mktemp("bend_kerr") / "wr90.msh")
    return bc.wr90_model(path, N2)


@pytest.fixture(scope="module")
def straight(model):
    return bc.solve_mode(model, bc.WR90_F, None, order=3)


@pytest.fixture(scope="module")
def bent(model):
    return bc.solve_mode(model, bc.WR90_F, bc.wr90_bend_m(30e-3), order=2)


def _gamma(model, result):
    return np.asarray(gfem.kerr_nonlinearity(model, result).gamma_nl)[0]


# --- the straight limit and the straight path -----------------------------------


# The bent and the straight problem differ by $h - 1 \le 10^{-6}$ at
# $R = 10^6 a$, and by its first order only through the odd part of the mode
# about the centre line, which a symmetric guide has none of: the difference
# of $\gamma_\mathrm{NL}$ is $O(10^{-12})$. What remains is two eigen-solves
# (ARPACK tolerance $10^{-10}$ on $\gamma$ and the BLAS of the platform, as
# for the other straight-limit gates), which moves $\gamma_\mathrm{NL}$, a
# quadratic form of the eigenvector, by about the same: gate $10^{-9}$.
STRAIGHT_LIMIT_RTOL = 1e-9


def test_a_very_large_radius_gives_the_straight_nonlinear_coefficient(model, straight):
    bent = bc.solve_mode(model, bc.WR90_F, bc.wr90_bend_m(1e6 * A), order=3)
    np.testing.assert_allclose(
        _gamma(model, bent), _gamma(model, straight), rtol=STRAIGHT_LIMIT_RTOL
    )


# The literals were computed before the bent path existed (WR-90 section of
# `bend_cases.wr90_mesh`, $n_2 = 1$, order 3, 10 GHz): $\gamma_\mathrm{NL}$ and
# $E_y$ at two points. The same solve on another platform differs by the
# eigen-solve's tolerance ($10^{-10}$ on $\gamma$, as in the straight-limit
# gate) times the BLAS differences of a 2000-DOF direct factorization, so the
# gate is $10^{-9}$, not the $10^{-13}$ of one machine (where it holds).
STRAIGHT_GOLDEN = {
    "gamma_nl": 2374519.690127559,
    "e_y": (2371.4630305468604, 2371.6330406564953),
}
GOLDEN_RTOL = 1e-9


def test_a_straight_mode_keeps_its_field_and_coefficient(model, straight):
    result = straight
    points = np.array([[0.3 * A, 2e-3, 0.0], [0.7 * A, 5e-3, 0.0]])
    e_y = np.asarray(result.e_field.at(points))[0, :, 1]
    np.testing.assert_allclose(e_y.real, STRAIGHT_GOLDEN["e_y"], rtol=GOLDEN_RTOL)
    np.testing.assert_allclose(
        _gamma(model, result).real, STRAIGHT_GOLDEN["gamma_nl"], rtol=GOLDEN_RTOL
    )


# --- the physical field ------------------------------------------------------------


def test_a_bent_field_divides_the_normal_component_by_h(bent):
    radius = 30e-3
    field = bent.e_field
    solved = copy.copy(field)
    solved._bend = None  # the transformed field, as solved
    x = np.array([0.2, 0.5, 0.9]) * A
    points = np.stack([x, np.full(3, 4e-3), np.zeros(3)], axis=1)
    physical = np.asarray(field.at(points))[0]
    raw = np.asarray(solved.at(points))[0]
    h = 1 + (x - A / 2) / radius
    # the frame of the port is the identity on this mesh (normal along z):
    # $E_x, E_y$ unchanged, $E_z / h$; one division, so within rounding
    np.testing.assert_allclose(physical[:, :2], raw[:, :2], rtol=0, atol=0)
    np.testing.assert_allclose(
        physical[:, 2] * h,
        raw[:, 2],
        rtol=1e-14,
        atol=1e-14 * np.abs(raw).max(),
    )


# --- the coupling tensor -----------------------------------------------------------


def test_the_coupling_diagonal_equals_the_coefficient_on_a_bent_mode(model, bent):
    result = bent
    tensor = np.asarray(gfem.kerr_coupling(model, result).gamma_nl)
    # the same sum over the points in two orders of summation (an einsum over
    # all points against a per-element sum): rounding of $10^3$ terms
    np.testing.assert_allclose(tensor[0, 0, 0, 0], _gamma(model, result), rtol=1e-12)


def test_the_coupling_refuses_a_straight_and_a_bent_mode(model, bent):
    straight = bc.solve_mode(model, bc.WR90_F, None, order=2)
    with pytest.raises(ValueError, match="different guides"):
        gfem.kerr_coupling(model, [straight, bent])


def test_the_coupling_refuses_two_radii(model, bent):
    a = bent
    b = bc.solve_mode(model, bc.WR90_F, bc.wr90_bend_m(40e-3), order=2)
    with pytest.raises(ValueError, match="different guides"):
        gfem.kerr_coupling(model, [a, b])


# --- the overlaps without a bend metric ------------------------------------------


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("three_wave_coupling", gfem.three_wave_coupling),
        ("second_harmonic", lambda m, r: gfem.second_harmonic(m, r, r)),
        ("spdc", lambda m, r: gfem.spdc(m, r, r, r)),
        ("sfwm", lambda m, r: gfem.sfwm(m, r, r, r)),
    ],
)
def test_the_overlaps_without_a_bend_metric_refuse_a_bent_mode(model, bent, name, call):
    with pytest.raises(NotImplementedError, match=f"gfem.{name}.*bent"):
        call(model, bent)


def test_the_characteristic_impedance_of_a_bent_mode_runs(bent):
    z = gfem.characteristic_impedance(bent, ((0.5 * A, 0, 0), (0.5 * A, 10.16e-3, 0)))
    assert np.isfinite(np.asarray(z)).all()


# --- transformations ---------------------------------------------------------------


def _gamma_of_radius(model):
    def gamma(radius):
        result = bc.solve_mode(model, bc.WR90_F, bc.wr90_bend_m(radius), order=2)
        # a jit-staged frequency is not readable on the host (as for a straight
        # mode); the grid is the one solved
        result = dataclasses.replace(result, frequency=np.array([bc.WR90_F]))
        return gfem.kerr_nonlinearity(model, result).gamma_nl[0].real

    return gamma


# Eager and staged runs do the same operations except that XLA may fuse and
# reorder the sums of the overlap (relative $10^{-15}$ per term, $\sqrt{n}$
# over the points) and the eigen-solve is the same host call; the gate
# $10^{-10}$ is the one of the solve's own jit-against-eager check.
JIT_RTOL = 1e-10


def test_jit_agrees_with_eager_for_the_coefficient_of_a_bent_mode(model):
    gamma = _gamma_of_radius(model)
    np.testing.assert_allclose(jax.jit(gamma)(30e-3), gamma(30e-3), rtol=JIT_RTOL)


def test_jit_agrees_with_eager_for_its_radius_gradient(model):
    gamma = _gamma_of_radius(model)
    np.testing.assert_allclose(
        jax.jit(jax.grad(gamma))(30e-3), jax.grad(gamma)(30e-3), rtol=JIT_RTOL
    )
