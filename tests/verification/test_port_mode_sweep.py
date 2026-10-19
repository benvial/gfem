"""The port mode sweep against exact port mode solves over 300-point grids.

Cases:
- a coplanar waveguide on silicon (signal 10 um, gaps 6 um, closed 300 x 300
  um box) over 1-100 GHz: quasi-TEM, with the E-field formulation's
  low-frequency accuracy floor at the bottom of the band;
- WR-90 with an off-centre eps_r = 6 block on its bottom wall over 9-20 GHz: a
  hybrid mode whose effective index climbs from 1.15 to 2.13.

Targets at the default ``port_mode_tolerance`` (1e-6): relative error of the
propagation constant <= 1e-8, of the mode field (1 W, same phase) <= 1e-5,
above the low-frequency floor, and at most 12 exact solves per port.
"""

from pathlib import Path

import numpy as np
import pytest

import gfem

DATA = Path(__file__).parents[1] / "unit" / "data"
N_GRID = 300


def _cpw():
    mesh = gfem.Mesh.from_gmsh(DATA / "cpw_section.msh", length_unit="um")
    model = gfem.Model(
        mesh,
        materials={"si": gfem.Material(eps_r=11.9), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls"), gfem.PEC("metal")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    # points in the slots and around the signal, m
    x = np.linspace(132e-6, 168e-6, 19)
    y = np.linspace(140e-6, 160e-6, 11)
    return model, np.linspace(1e9, 100e9, N_GRID), (x, y)


def _hybrid():
    mesh = gfem.Mesh.from_gmsh(DATA / "hybrid_waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"block": gfem.Material(eps_r=6.0), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    x = np.linspace(0.5e-3, 22.36e-3, 16)
    y = np.linspace(0.5e-3, 9.66e-3, 8)
    return model, np.linspace(9e9, 20e9, N_GRID), (x, y)


@pytest.fixture(scope="module", params=["cpw", "hybrid"])
def case(request):
    model, grid, (x, y) = {"cpw": _cpw, "hybrid": _hybrid}[request.param]()
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    swept = gfem.solve(model, gfem.PortMode("p1", grid))
    return request.param, model, grid, points, swept


def test_at_most_twelve_exact_solves_cover_the_band(case):
    _, _, grid, _, swept = case
    solved = np.asarray(swept.solved)
    assert solved[0]
    assert solved[-1]
    assert solved.sum() <= 12, grid[solved]


def test_the_sweep_matches_exact_solves(case):
    name, model, grid, points, swept = case
    gamma = np.asarray(swept.gamma)
    field = np.asarray(swept.e_field.at(points))
    residual = np.asarray(swept.residual)
    # unsampled frequencies spread over the band
    for i in np.linspace(1, N_GRID - 2, 7).astype(int):
        exact = gfem.solve(model, gfem.PortMode("p1", grid[i]))
        k_err = abs(gamma[i] / np.asarray(exact.gamma)[0] - 1)
        e_ref = np.asarray(exact.e_field.at(points))[0]
        e_err = np.linalg.norm(field[i] - e_ref) / np.linalg.norm(e_ref)
        # Below a few GHz the CPW sits on the low-frequency floor: there the
        # guard accepts 10x the exact solve's own backward error (reported as
        # a residual above the tolerance), and the exact solve is no sharper
        # than that, so only the looser bound applies.
        on_floor = residual[i] > 1e-6
        k_tol, e_tol = (1e-6, 1e-4) if on_floor else (1e-8, 1e-5)
        assert k_err <= k_tol, (name, grid[i], k_err, on_floor)
        assert e_err <= e_tol, (name, grid[i], e_err, on_floor)
