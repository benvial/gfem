"""Electrostatic parity with Palace on the two-layer plate stack.

Cases electrostatic/r1 and r05: a TOP plate (m2) over LEFT and RIGHT plates
(m1) across 0.5 um of SiO2, with a grounded m1 strip; order 2. The model is
built from the case's Palace ``config.json`` (material and terminal
attributes) and ``terminals.json`` (terminal names), and the conductor
interiors are excluded, as gsim would. gfem discretizes the same mesh at
the same order as Palace, so every capacitance entry agrees to far better
than the 1e-4 asked here, on either linear solver.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import gfem

CASES = ("r1", "r05")
RTOL = 1e-4


def _csv(path: Path) -> np.ndarray:
    with path.open() as f:
        rows = list(csv.reader(f))
    return np.array([[float(v) for v in row[1:]] for row in rows[1:]])


@pytest.fixture(scope="module", params=CASES)
def case(request, reference_dir):
    return reference_dir / "electrostatic" / request.param


def _model_and_problem(case: Path) -> tuple[gfem.Model, gfem.Electrostatic]:
    config = json.loads((case / "config.json").read_text())
    names = {
        t["index"]: t["name"] for t in json.loads((case / "terminals.json").read_text())
    }
    mesh = gfem.Mesh.from_gmsh(case / "palace.msh")
    by_tag = {(r.dim, r.tag): r.name for r in mesh.regions}
    materials = {
        by_tag[3, tag]: gfem.Material(eps_r=m["Permittivity"])
        for m in config["Domains"]["Materials"]
        for tag in m["Attributes"]
    }
    boundaries = config["Boundaries"]
    terminals = {
        names[t["Index"]]: [by_tag[2, tag] for tag in t["Attributes"]]
        for t in sorted(boundaries["Terminal"], key=lambda t: t["Index"])
    }
    ground = [by_tag[2, tag] for tag in boundaries["Ground"]["Attributes"]]
    shells = [s for group in [*terminals.values(), ground] for s in group]
    excluded = tuple(
        gfem.enclosed_by(s, s.replace("_xy@", "_z@")) for s in shells if "_xy@" in s
    )
    model = gfem.Model(mesh, materials=materials, excluded=excluded)
    return model, gfem.Electrostatic(terminals=terminals, ground=ground)


@pytest.fixture(scope="module")
def setup(case):
    return _model_and_problem(case)


@pytest.mark.parametrize("solver", ["auto", "direct"])
def test_every_capacitance_entry_matches_palace(case, setup, solver):
    model, problem = setup
    cap = gfem.solve(
        model,
        problem,
        options=gfem.SolverOptions(linear_solver=solver, threads=4),
    )
    palace = case / "output" / "palace"
    assert cap.terminals == ("TOP", "LEFT", "RIGHT")
    np.testing.assert_allclose(
        cap.capacitance, _csv(palace / "terminal-C.csv"), rtol=RTOL
    )
    np.testing.assert_allclose(cap.mutual, _csv(palace / "terminal-Cm.csv"), rtol=RTOL)
    np.testing.assert_allclose(
        cap.inverse, _csv(palace / "terminal-Cinv.csv"), rtol=RTOL
    )
    # Palace's domain-E: total energy of excitation i at V_inc[i]
    volts = _csv(palace / "terminal-V.csv")[:, 0]
    stored = _csv(palace / "domain-E.csv")[:, 0]
    total = sum(np.asarray(e) for e in cap.energy.values())
    np.testing.assert_allclose(total * volts**2, stored, rtol=RTOL)
    assert set(cap.energy) == {"sio2", "air"}


def test_quoted_values_in_femtofarads(case, setup):
    expected = {"r1": 30.20, "r05": 29.95}[case.name]
    cap = gfem.solve(*setup, options=gfem.SolverOptions(threads=4))
    top, left = cap.index("TOP"), cap.index("LEFT")
    assert float(cap.mutual[top, left]) * 1e15 == pytest.approx(expected, abs=0.005)
