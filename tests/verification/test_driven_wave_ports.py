"""Driven wave ports: convergence, and how port mode errors reach S.

- An empty WR-90 guide (6 mm, TE10 at 8-12 GHz): S21 converges with the
  order to ``exp(-gamma L)`` with the analytic propagation constant, and S11
  to zero.
- WR-90 with an eps_r = 6 block along its length (a strongly dispersive
  hybrid mode, 9-20 GHz): the ports impose the mode's full n x H (Robin
  term plus the rank-two modal correction), so the guide is matched: S11
  converges to zero and S21 to ``exp(-gamma L)`` with the order (measured
  max |S21 - exp(-gamma L)| 5.6e-3 / 1.9e-4 / 3.0e-6 at orders 2 / 3 / 4;
  without the correction S11 was about 0.4 at 9 GHz whatever the order).
  The adaptive sweep, which projects the correction as a low-rank term,
  matches the uniform one.
- The same guide: S's error is at least first order in the port mode's
  error as ``SolverOptions.port_mode_tolerance`` tightens from 1e-4 to
  1e-8 (measured log-log slope 1.18), and below 1e-5 at the default
  tolerance.
"""

from pathlib import Path

import numpy as np
import pytest

import gfem

DATA = Path(__file__).parents[1] / "unit" / "data"
C0 = 299792458.0
A, LENGTH = 22.86e-3, 6e-3
HYBRID_LENGTH = 2e-3  # m


def _guide(name, materials):
    mesh = gfem.Mesh.from_gmsh(DATA / name, length_unit="mm")
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def test_te10_transmission_converges_with_order():
    model = _guide("waveguide.msh", {"air": gfem.Material()})
    f = np.array([8e9, 10e9, 12e9])
    beta = np.sqrt((2 * np.pi * f / C0) ** 2 - (np.pi / A) ** 2)
    errors = []
    for order in (2, 3, 4):
        options = gfem.SolverOptions(order=order, linear_solver="direct")
        s = np.asarray(
            gfem.solve(model, gfem.Driven(tuple(f)), options=options).s.values
        )
        errors.append(
            max(
                np.abs(s[:, 1, 0] - np.exp(-1j * beta * LENGTH)).max(),
                np.abs(s[:, 0, 0]).max(),
            )
        )
    assert errors[0] > 10 * errors[1] > 100 * errors[2]
    assert errors[2] < 1e-6


GRID = tuple(np.linspace(9e9, 20e9, 60))
TOLERANCES = (1e-4, 1e-5, 1e-6, 1e-8)
REFERENCE = 1e-12


@pytest.fixture(scope="module")
def hybrid():
    return _guide(
        "hybrid_waveguide.msh",
        {"block": gfem.Material(eps_r=6.0), "air": gfem.Material()},
    )


def _run(model, tolerance):
    options = gfem.SolverOptions(
        order=2, linear_solver="direct", port_mode_tolerance=tolerance
    )
    s = np.asarray(gfem.solve(model, gfem.Driven(GRID), options=options).s.values)
    # the 2D port mode problem takes no linear solver choice
    port_options = gfem.SolverOptions(order=2, port_mode_tolerance=tolerance)
    mode = gfem.solve(model, gfem.PortMode("p1", GRID), options=port_options).e_field
    coefficients = np.asarray(mode._basis @ mode._coords.T)
    return s, coefficients


def test_s_error_follows_the_port_mode_error_at_least_linearly(hybrid):
    s_ref, mode_ref = _run(hybrid, REFERENCE)
    profile, ds = [], []
    for tol in TOLERANCES:
        s, mode = _run(hybrid, tol)
        profile.append(
            (
                np.linalg.norm(mode - mode_ref, axis=0)
                / np.linalg.norm(mode_ref, axis=0)
            ).max()
        )
        ds.append(np.abs(s - s_ref).max())
    print("\nport_mode_tolerance, profile error, |dS|")  # noqa: T201
    for tol, p, d in zip(TOLERANCES, profile, ds, strict=True):
        print(f"  {tol:.0e}  {p:.3e}  {d:.3e}")  # noqa: T201
    slope = np.polyfit(np.log(profile), np.log(ds), 1)[0]
    print(f"  slope {slope:.3f}")  # noqa: T201
    assert slope >= 1.0
    assert ds[TOLERANCES.index(1e-6)] <= 1e-5


HYBRID_GRID = (9e9, 12e9, 16e9, 20e9)


def test_a_hybrid_mode_is_matched_and_transmits_with_order(hybrid):
    s11, s21 = [], []
    for order in (2, 3, 4):
        options = gfem.SolverOptions(order=order, linear_solver="direct")
        result = gfem.solve(hybrid, gfem.Driven(HYBRID_GRID), options=options)
        s = np.asarray(result.s.values)
        gamma = np.asarray(result.port_modes["p1"].gamma)
        s11.append(np.abs(s[:, [0, 1], [0, 1]]).max())
        s21.append(np.abs(s[:, 1, 0] - np.exp(-gamma * HYBRID_LENGTH)).max())
    print(f"\nhybrid guide: max |S11| {s11}, max |S21 - exp(-gamma L)| {s21}")  # noqa: T201
    assert s21[0] > 10 * s21[1] > 100 * s21[2]
    assert s11[0] > s11[1] > s11[2]
    assert s11[1] < 2e-4  # measured 9.8e-5 (20 GHz)
    assert s11[2] < 1e-5
    assert s21[2] < 1e-5


def test_the_adaptive_sweep_projects_the_modal_correction(hybrid):
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    grid = tuple(np.linspace(9e9, 20e9, 23))
    adaptive = gfem.solve(hybrid, gfem.Driven(grid, sweep="adaptive"), options=options)
    solved = np.array(adaptive.sweep.solved)
    rows = [i for i in (3, 10, 17) if not solved[i]]
    uniform = gfem.solve(
        hybrid, gfem.Driven(tuple(grid[i] for i in rows)), options=options
    )
    du = np.abs(np.asarray(adaptive.s.values)[rows] - np.asarray(uniform.s.values))
    assert rows
    assert du.max() <= 1e-6
