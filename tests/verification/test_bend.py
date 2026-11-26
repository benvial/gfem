r"""Bent port modes against exact Bessel/Hankel roots, in order $p = 2..5$.

Three guides whose bend has a closed-form mode (`bend_cases`):

- the H-plane bend of a WR-90 guide ($R$ = 20, 30, 60 mm; real $\nu$, with
  $n_g$ and $\beta_2$);
- a bent parallel-plate line with $E$ in the bend plane ($R$ = 10, 20, 40 mm;
  real $\nu$);
- the leaky bent dielectric slab, Ey and Hy ($R$ = 3, 5, 8 um; complex $\nu$,
  so the bend loss), with the group index and dispersion at $R = 5$ um.

Each case's error falls by at least $10^3$ from $p = 2$ to $p = 5$ on the
coarse structured mesh (a $p$-refinement: the mesh is not refined), and the
final order meets the gates below. The gates are about twice the error
measured at $p = 5$ (the floor of the discretization on that mesh).
"""

from __future__ import annotations

import math

import bend_cases as bc
import numpy as np
import pytest

import gfem

DB90 = 10 * math.pi / math.log(10)
ORDERS = (2, 3, 4, 5)
#: The required fall of the error of $\nu$ from $p = 2$ to $p = 5$.
DROP = 1e3


def _solve(model, frequency, bend, order):
    problem = gfem.PortMode("p1", (frequency,), bend=bend)
    return gfem.solve(
        model, problem, options=gfem.SolverOptions(order=order, threads=1)
    )


def _ladder(errors):
    """The fall of the error from the first order to the last, floored at round-off."""
    """The fall of the error from the first order to the last (round-off floor)."""
    floor = 1e-15
    return errors[0] / max(errors[-1], floor)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("bend_verification")


@pytest.mark.parametrize("radius", [20.0, 30.0, 60.0])
def test_wr90_h_plane_bend(workdir, radius):
    model = bc.wr90_model(bc.wr90_mesh(workdir / "wr90.msh"))
    nu0, ng0, b20 = bc.WR90[radius]
    errors = []
    for order in ORDERS:
        r = _solve(model, bc.WR90_F, bc.wr90_bend(radius), order)
        nu = bc.nu_of(r, radius * 1e-3)
        errors.append(abs(nu / nu0 - 1))
    assert _ladder(errors) >= DROP
    # p = 5 on the 6x2 mesh: nu to 2.6e-10 (R = 20 mm) and below; n_g and beta_2
    # follow it; the closed lossless guide has no loss, |Im nu| is round-off
    assert errors[-1] <= 1e-9
    assert abs(nu.imag) <= 1e-10 * abs(nu)
    assert abs(complex(np.asarray(r.n_g)[0]) / ng0 - 1) <= 1e-9
    assert abs(complex(np.asarray(r.beta_2)[0]) / b20 - 1) <= 1e-8


@pytest.mark.parametrize("radius", [10.0, 20.0, 40.0])
def test_e_in_plane_line_bend(workdir, radius):
    model = bc.line_model(bc.line_mesh(workdir / "line.msh"))
    errors = []
    for order in ORDERS:
        r = _solve(model, bc.LINE_F, bc.line_bend(radius), order)
        errors.append(abs(bc.nu_of(r, radius * 1e-3) / bc.LINE[radius] - 1))
    assert _ladder(errors) >= DROP
    # measured 7.8e-13 (R = 10 mm), 7e-16, 7e-16 at p = 5
    assert errors[-1] <= 2e-12


@pytest.mark.parametrize("radius", [3, 5, 8])
@pytest.mark.parametrize("pol", ["Ey", "Hy"])
def test_leaky_slab_bend(workdir, pol, radius):
    # a radiating Ey bend at R = 8 um loses 1e-4 per radian: a layer reflection of
    # R0 = 1e-10 would show in Im nu, so that case runs at R0 = 1e-12
    r0 = 1e-12 if (pol, radius) == ("Ey", 8) else 1e-10
    model = bc.slab_model(bc.slab_mesh(workdir / f"slab_{pol}.msh", pol), r0)
    ref = bc.SLAB_NU[(radius, pol)]
    errors = []
    for order in ORDERS:
        r = _solve(model, bc.SLAB_F, bc.slab_bend(radius), order)
        nu = bc.nu_of(r, radius * 1e-6)
        errors.append(abs(nu - ref) / abs(ref))
    assert _ladder(errors) >= DROP
    assert abs(nu.real - ref.real) <= 1e-12 * abs(ref)
    assert abs(nu.imag - ref.imag) <= 1e-9
    # bend_loss is L_90 = -(10 pi/ln 10) Im nu: the literal's, to the Im nu gate
    np.testing.assert_allclose(
        np.asarray(r.bend_loss)[0], -DB90 * ref.imag, atol=DB90 * 1e-9, rtol=0
    )


@pytest.mark.parametrize("pol", ["Ey", "Hy"])
def test_leaky_slab_dispersion(workdir, pol):
    model = bc.slab_model(bc.slab_mesh(workdir / f"disp_{pol}.msh", pol))
    r = _solve(model, bc.SLAB_F, bc.slab_bend(5), 5)
    ng, b2 = bc.SLAB_DISPERSION[pol]
    got_ng = complex(np.asarray(r.n_g)[0])
    got_b2 = complex(np.asarray(r.beta_2)[0]) * 1e24  # ps^2/m
    # measured at p = 5: n_g 4.4e-13 (Ey), 1.1e-12 (Hy); beta_2 2.6e-10, 2.8e-10
    assert abs(got_ng / ng - 1) <= 3e-12
    assert abs(got_b2 / b2 - 1) <= 6e-10
