"""Driven wave ports on anisotropic fills: a 10 mm WR-28 guide, two wave ports.

A uniform guide does not reflect, and its S21 is ``exp(-gamma L)`` with the
propagation constant of its own port mode. For a fill with principal axes
along the guide (x across, y up, z along), TE10 has E along y and the
closed form ``beta^2 = mu_x (eps_y k0^2 - (pi / a)^2 / mu_z)`` (lossless) or
``eps_y (1 - i sigma_y / (omega eps0))`` in place of ``eps_y`` (lossy); the
references are in the anisotropic-materials verification notes. A fill
turned about the guide axis has a hybrid mode with no closed form, but the
guide is still uniform.

The port's Robin term and its modal coupling must read the port's
inverse-permeability tensor ``A = N mu^-1 N^T``: with the rotated
permeability in the coupling instead, the mu = (1, 2, 1.5) guide reflects
about 0.33 (ADR 0054).
"""

import anisotropy
import numpy as np
import pytest

import gfem

LENGTH = 10.0  # mm
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")

EPS = (2.0, 4.0, 3.0)
MU = (1.0, 2.0, 1.5)
F_MU = 16e9  # TE10 alone propagates in the eps and mu fill (TE20 from 17.2 GHz)
BETA_MU = 565.4329872556062  # rad/m, TE10 of that fill at 16 GHz
F_SIGMA = 20e9
SIGMA = (0.2, 0.05, 0.1)  # S/m
GAMMA_SIGMA = 5.540510958349465 + 712.541098151911j  # 1/m, TE10 at 20 GHz


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return anisotropy.wr28(
        tmp_path_factory.mktemp("wr28"), length=LENGTH, cells=(8, 4, 10)
    )


def _solve(mesh, material, frequency):
    model = gfem.Model(
        mesh,
        materials={"fill": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    result = gfem.solve(model, gfem.Driven(frequency), options=OPTIONS)
    s = np.asarray(result.s.values)[0]
    gamma = complex(np.asarray(result.port_modes["p1"].gamma)[0])
    return s, gamma


def test_a_guide_with_anisotropic_permeability_transmits_its_te10_mode(mesh):
    s, gamma = _solve(mesh, gfem.Material(eps_r=EPS, mu_r=MU), F_MU)
    np.testing.assert_allclose(gamma.imag, BETA_MU, rtol=1e-4)
    assert abs(s[0, 0]) <= 1e-2
    assert abs(s[1, 1]) <= 1e-2
    assert abs(np.angle(s[1, 0] * np.exp(1j * BETA_MU * LENGTH * 1e-3))) <= 1e-2
    assert abs(s[0, 1] - s[1, 0]) <= 1e-8  # reciprocal


def test_a_guide_with_per_axis_conductivity_attenuates_along_its_field(mesh):
    material = gfem.Material(eps_r=EPS, conductivity=SIGMA)
    s, _ = _solve(mesh, material, F_SIGMA)
    assert abs(s[1, 0] - np.exp(-GAMMA_SIGMA * LENGTH * 1e-3)) <= 1e-3
    assert abs(s[0, 0]) <= 1e-2


@pytest.mark.parametrize(
    ("values", "frequency"),
    [({"eps_r": EPS}, 20e9), ({"eps_r": EPS, "mu_r": MU}, F_MU)],
    ids=["eps", "eps_mu"],
)
def test_a_fill_turned_about_the_guide_axis_does_not_reflect(mesh, values, frequency):
    # Palace issue #835 (eps only) reflected |S11| ~ 2.5 here
    material = gfem.Material(**values, axes=anisotropy.turned(2, 30.0))
    s, gamma = _solve(mesh, material, frequency)
    assert abs(gamma.real) <= 1e-8 * abs(gamma)  # a propagating mode
    assert abs(s[0, 0]) <= 1e-2
    assert abs(s[1, 0]) >= 0.99
