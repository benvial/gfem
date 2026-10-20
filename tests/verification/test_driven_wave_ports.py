"""Driven wave ports: convergence, and how port mode errors reach S.

- An empty WR-90 guide (6 mm, TE10 at 8-12 GHz): S21 converges with the
  order to ``exp(-gamma L)`` with the analytic propagation constant, and S11
  to zero.
- WR-90 with an eps_r = 6 block (a strongly dispersive hybrid mode, 9-20
  GHz): S's error is first order in the port mode's error as
  ``SolverOptions.port_mode_tolerance`` tightens from 1e-4 to 1e-8, and is
  below 1e-5 at the default tolerance. Measured: |dS| / (relative mode
  error) stays between 0.04 and 0.2 from 1e-3 to 1e-10, and the log-log
  slope over the four tolerances is 0.93 (scatter of that ratio, not a
  trend), so the check asks for a slope of at least 0.9 and a bounded
  ratio. (Palace's wave port boundary, i Re(k_n) on the port
  face, drops the reflected wave's longitudinal field, so this guide's
  ports are far from matched; that is the formulation, not a port mode
  error, and does not change how port mode errors propagate.)
"""

from pathlib import Path

import numpy as np
import pytest

import gfem

DATA = Path(__file__).parents[1] / "unit" / "data"
C0 = 299792458.0
A, LENGTH = 22.86e-3, 6e-3


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
    mode = gfem.solve(model, gfem.PortMode("p1", GRID), options=options).e_field
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
    assert slope >= 0.9
    assert (np.array(ds) <= 0.5 * np.array(profile)).all()
    assert ds[TOLERANCES.index(1e-6)] <= 1e-5
