"""Driven operator: manufactured solutions converge at the Nédélec rate.

On the unit cube with PEC walls, E = (sin πy sin πz, sin πx sin πz,
sin πx sin πy) is divergence-free with n x E = 0, and solves
curl(mu_r^-1 curl E) - k0² eps E = f with f = (2π² / mu_r - k0² eps) E.
The driven operator (its affine pieces at one frequency, as the solve
assembles them) with that source converges in L2 at rate p for order p.

A lumped two-port parallel-plate line then checks the S-parameters
themselves: they converge to the transmission-line closed form, and stay
symmetric and passive with a lossy dielectric.
"""

import math

import ngsolve as ngs
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem
from gfem._engine import quadrature
from gfem._engine.driven import driven_operator
from gfem._linalg import Operator, prepare

EPS, TAN_DELTA, MU = 2.0, 0.1, 1.5
K0 = 2.0  # 1/m: the cube is 1 m across, so omega = K0 c
CELLS = {1: (2, 4, 8), 2: (2, 4, 8), 3: (2, 4)}


def _error(tmp_path, order, n):
    path = sm.write_msh22(
        tmp_path / f"cube{n}.msh", sm.plate(1.0, 1.0, (("box", 1.0),), (n, n, n))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="m"),
        materials={"box": gfem.Material(eps_r=EPS, tan_delta=TAN_DELTA, mu_r=MU)},
        boundaries=[gfem.PEC(s) for s in ("bottom", "top", "side")],
    )
    op = driven_operator(model, order)
    omega = K0 * const.c
    assert op.wavenumber(omega) == pytest.approx(K0)
    x, y, z = ngs.x, ngs.y, ngs.z
    pi = math.pi
    exact = ngs.CF(
        (
            ngs.sin(pi * y) * ngs.sin(pi * z),
            ngs.sin(pi * x) * ngs.sin(pi * z),
            ngs.sin(pi * x) * ngs.sin(pi * y),
        )
    )
    factor = 2 * pi**2 / MU - K0**2 * EPS * (1 - 1j * TAN_DELTA)
    space = op.space
    rhs = ngs.LinearForm(space)
    rhs += factor * exact * space.TestFunction() * quadrature.dx(order + 2)
    rhs.Assemble()
    handle = prepare(
        Operator(op.form(omega), space.FreeDofs()),
        symmetry="complex_symmetric",
        kind="direct",
    )
    gf = ngs.GridFunction(space)
    gf.vec.FV().NumPy()[:] = handle.solve(rhs.vec.FV().NumPy())
    handle.free()
    mesh = op.built.mesh
    diff = gf - exact
    error = ngs.Integrate(ngs.InnerProduct(diff, diff), mesh, order=2 * order + 4)
    return math.sqrt(abs(error))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_manufactured_solution_converges_at_the_nedelec_rate(tmp_path, order):
    errors = [_error(tmp_path, order, n) for n in CELLS[order]]
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert rates[-1] > order - 0.3, (errors, rates)


L, W, H = 10.0, 2.0, 1.0  # mm
ETA0 = const.mu_0 * const.c


def _line_s(tmp_path, cells, order, tan_delta=0.0):
    path = sm.write_msh22(
        tmp_path / f"line{cells[0]}.msh", sm.strip_line(L, W, H, cells)
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0, tan_delta=tan_delta)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z", resistance=30.0),
        ],
    )
    result = gfem.solve(
        model,
        gfem.Driven(frequency=[3e9, 8e9]),
        options=gfem.SolverOptions(order=order, linear_solver="direct"),
    )
    return np.asarray(result.s.values)


def _line_exact(f):
    beta = 2 * math.pi * f / const.c * 2.0
    z0 = ETA0 * H / (W * 2.0)
    c, s = math.cos(beta * L * 1e-3), math.sin(beta * L * 1e-3)
    a, b, cc, d = c, 1j * z0 * s, 1j * s / z0, c
    r1, r2 = 50.0, 30.0
    # power-wave S of an ABCD two-port between real references r1, r2
    den = a * r2 + b + cc * r1 * r2 + d * r1
    k = math.sqrt(r1 * r2)
    return np.array(
        [
            [
                (a * r2 + b - cc * r1 * r2 - d * r1) / den,
                2 * k * (a * d - b * cc) / den,
            ],
            [2 * k / den, (-a * r2 + b - cc * r1 * r2 + d * r1) / den],
        ]
    )


@pytest.mark.parametrize(
    ("order", "cells"),
    [
        (1, ((10, 2, 2), (20, 4, 4), (40, 8, 8))),
        (2, ((2, 1, 1), (4, 1, 1), (8, 1, 1))),
    ],
)
def test_line_s_parameters_converge_to_the_transmission_line(tmp_path, order, cells):
    exact = np.array([_line_exact(f) for f in (3e9, 8e9)])
    errors = [np.abs(_line_s(tmp_path, c, order) - exact).max() for c in cells]
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert rates[-1] > 2 * order - 0.3, (errors, rates)


def test_lossy_line_is_symmetric_and_passive(tmp_path):
    for m in _line_s(tmp_path, (10, 1, 1), 2, tan_delta=0.05):
        np.testing.assert_allclose(m, m.T, atol=1e-12)
        assert np.linalg.svd(m, compute_uv=False).max() < 1
