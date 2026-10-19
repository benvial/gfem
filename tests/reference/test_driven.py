"""Driven parity with Palace on the IHP coplanar line with lumped ports.

Case driven/r8_abc1: an 800 um TopMetal2 CPW on the IHP SG13G2 stack, two
50 ohm two-element lumped CPW ports, surface-impedance conductors and a
first-order absorbing boundary; order 2. The model is built from the case's
Palace ``config.json`` as gsim would build it: the metal volumes are
excluded (their shells carry the `Conductor`), materials take the
permittivity and loss tangent, and the absorbing surfaces, conductors and
ports take Palace's attributes. gfem and Palace discretize the same mesh at
the same order, so S agrees to the discretization gap between the codes
(conductor cracking, the solver tolerance): max |dS| <= 1e-5 at Palace's
own output frequencies.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import gfem
from gfem._engine.driven import DrivenRule

TOLERANCE = 1e-5
#: Palace output frequencies (GHz) checked: rows of port-S.csv near 10, 50 and
#: 100 GHz.
ROWS = (27, 148, 299)


@pytest.fixture(scope="module")
def case():
    ref = os.environ.get("GFEM_REFERENCE_DIR")
    if not ref:
        pytest.skip("GFEM_REFERENCE_DIR not set (needs the driven/r8_abc1 Palace run)")
    return Path(ref) / "driven" / "r8_abc1"


def _model(case: Path) -> gfem.Model:
    config = json.loads((case / "config.json").read_text())
    unit = config["Model"]["L0"]
    mesh = gfem.Mesh.from_gmsh(case / "palace.msh", length_unit="um")
    assert unit == 1e-6
    name = {(r.dim, r.tag): r.name for r in mesh.regions}
    materials = {
        name[3, tag]: gfem.Material(
            eps_r=m["Permittivity"], tan_delta=m.get("LossTan", 0.0)
        )
        for m in config["Domains"]["Materials"]
        for tag in m["Attributes"]
    }
    bnd = config["Boundaries"]
    conductors = [
        gfem.Conductor(name[2, tag], c["Conductivity"], c["Thickness"] * unit)
        for c in bnd["Conductivity"]
        for tag in c["Attributes"]
    ]
    order = bnd["Absorbing"]["Order"]
    absorbing = [
        gfem.Absorbing(name[2, tag], order=order)
        for tag in bnd["Absorbing"]["Attributes"]
    ]
    ports = [
        gfem.LumpedPort(
            f"p{p['Index']}",
            tuple(name[2, e["Attributes"][0]] for e in p["Elements"]),
            tuple(e["Direction"] for e in p["Elements"]),
            impedance=p["R"],
        )
        for p in sorted(bnd["LumpedPort"], key=lambda p: p["Index"])
    ]
    shells = [c.surface for c in conductors]
    excluded = tuple(
        gfem.enclosed_by(s, s.replace("_xy", "_z")) for s in shells if s.endswith("_xy")
    )
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[*conductors, *absorbing],
        ports=ports,
        excluded=excluded,
    )


def _palace(case: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(
        case / "output" / "palace" / "port-S.csv", delimiter=",", skiprows=1
    )
    rows = data[list(ROWS)]
    s = np.zeros((len(rows), 2, 2), complex)
    # columns: f, then |S11| dB, arg S11, |S21|, arg S21, |S12|, arg S12, |S22|, ...
    for k, (i, j) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        db, deg = rows[:, 1 + 2 * k], rows[:, 2 + 2 * k]
        s[:, i, j] = 10 ** (db / 20) * np.exp(1j * np.radians(deg))
    return rows[:, 0] * 1e9, s


def test_driven_parity_s_matches_palace(case):
    model = _model(case)
    frequency, palace = _palace(case)
    options = gfem.SolverOptions(threads=4)
    problem = gfem.Driven(frequency=frequency)
    rule = DrivenRule(model, problem, options)
    assert rule.kind == "bddc_gmres"
    assert rule.operator(model).space.ndof == 162_640
    outputs, _ = rule.forward(model, keep=False)
    s = outputs["s"]
    diff = np.abs(s - palace).max()
    assert diff <= TOLERANCE, (diff, outputs["iterations"])
    print(f"r8_abc1: max |dS| {diff:.3g}, GMRES iterations {outputs['iterations']}")  # noqa: T201
    # Palace and the solver spike: 48-49 GMRES iterations at this tolerance
    assert max(max(its) for its in outputs["iterations"]) <= 54, outputs
    # reciprocal to the GMRES tolerance
    np.testing.assert_allclose(s, np.swapaxes(s, 1, 2), atol=1e-6)
    # the public path (GMRES from zero) gives the same S to solver tolerance
    result = gfem.solve(model, gfem.Driven(frequency=frequency[1]), options=options)
    np.testing.assert_allclose(result.s.values[0], s[1], atol=2e-6)
