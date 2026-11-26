r"""Kerr overlaps and physical fields of a bent port mode against exact references.

- $\gamma_\mathrm{NL}$ of the H-plane bend of a WR-90 guide and of the bent
  parallel-plate line ($E$ in the bend plane, so both field components of the
  bent mode count), against the quadrature of the Bessel mode with the
  physical volume element $h$ (`bend_cases.KERR_H`, `KERR_E`);
- the physical field: $|E_x/E_s|$ of the bent slab against the Hankel ratio;
- the derivative of $\gamma_\mathrm{NL}$ by the radius against differences.
"""

from __future__ import annotations

import bend_cases as bc
import finite_differences as fd
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

N2 = 1.0  # m^2/W: the references are per unit $n_2$


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("bend_kerr_verification")


def _gamma_nl(model, result):
    return complex(np.asarray(gfem.kerr_nonlinearity(model, result).gamma_nl)[0])


# Measured at $p = 5$ on the coarse structured mesh: 4.6e-10, 1e-11, 4e-12 for
# $R$ = 20, 30, 60 mm (H) and 8e-11, 2e-12, 2e-12 for $R$ = 10, 20, 40 mm (E);
# the gates are the repository's for this effort (1e-7, 2e-8), 200 times the
# worst error. Without the volume element $h$ they are off by 0.5-4 % (H) and
# 0.25-4 % (E), and with the transformed $E_s$ by up to 0.2 %. The degree-$4p$ rule
# is not exact for $h$ and $e_n/h$, whose error is the part of the above.
@pytest.mark.parametrize("radius", [20.0, 30.0, 60.0])
def test_h_plane_bend_nonlinear_coefficient(workdir, radius):
    model = bc.wr90_model(bc.wr90_mesh(workdir / "wr90.msh"), N2)
    result = bc.solve_mode(model, bc.WR90_F, bc.wr90_bend(radius), order=5)
    gamma = _gamma_nl(model, result)
    ref = bc.KERR_H[radius]
    assert abs(gamma.imag) <= 1e-12 * abs(gamma.real)
    assert abs(gamma.real / ref - 1) <= 1e-7


@pytest.mark.parametrize("radius", [10.0, 20.0, 40.0])
def test_e_in_plane_line_bend_nonlinear_coefficient(workdir, radius):
    model = bc.line_model(bc.line_mesh(workdir / "line.msh"), N2)
    result = bc.solve_mode(model, bc.LINE_F, bc.line_bend(radius), order=5)
    gamma = _gamma_nl(model, result)
    ref = bc.KERR_E[radius]
    assert abs(gamma.imag) <= 1e-12 * abs(gamma.real)
    assert abs(gamma.real / ref - 1) <= 2e-8


# --- the physical field -------------------------------------------------------


def test_the_physical_field_follows_the_hankel_ratio(workdir):
    model = bc.slab_model(bc.slab_mesh(workdir / "slab_Hy.msh", "Hy", 0.05), 1e-10)
    result = bc.solve_mode(model, bc.SLAB_F, bc.slab_bend(5.0), order=4)
    xs = np.array(sorted(bc.SLAB_FIELD_RATIO))
    points = np.stack([xs * 1e-6, np.full(len(xs), 0.05e-6), np.zeros(len(xs))], 1)
    e = np.asarray(result.e_field.at(points))[0]
    ratio = np.abs(e[:, 0] / e[:, 2])
    ref = np.array([bc.SLAB_FIELD_RATIO[x] for x in xs])
    # measured 3e-7 at most (the field's discretization error at p = 4 on
    # h = 0.05 um; without the division by $h$ the ratio is off by 5-30 %)
    np.testing.assert_allclose(ratio, ref, rtol=1e-6)


# --- the radius ----------------------------------------------------------------


# As for the gradients of the mode itself: step $10^{-3}\rho$ ($\rho = R - a/2$,
# the distance of the nearest wall from the axis), truncation of the fourth-order
# difference $(\delta/\rho)^4 \approx 10^{-12}$, round-off $1.5\,\epsilon\,\rho/\delta
# \approx 1.5\times10^{-10}$ for a solve accurate to $10^{-13}$: both far below
# the repository's gate 1e-6.
def test_the_radius_derivative_of_the_nonlinear_coefficient_matches_differences(
    workdir,
):
    model = bc.wr90_model(bc.wr90_mesh(workdir / "wr90_fd.msh"), N2)
    radius = 30e-3
    rho = radius - bc.WR90_A / 2 * 1e-3
    options = gfem.SolverOptions(order=3, threads=1)

    def gamma(r):
        problem = gfem.PortMode("p1", (bc.WR90_F,), bend=bc.wr90_bend_m(r))
        result = gfem.solve(model, problem, options=options)
        return jnp.real(gfem.kerr_nonlinearity(model, result).gamma_nl[0])

    fd.assert_gradient_matches(
        gamma, jnp.asarray(radius), [1.0], step=1e-3 * rho, rel=1e-6
    )
