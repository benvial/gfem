"""Nonlinear eigenmode problems against Palace on the IHP half-wave resonator.

The 800 um TopMetal2 CPW of the driven reference (IHP SG13G2 stack, 20 um
signal, 15 um gaps) with no ports, so both line ends are open: a half-wave
resonator near 104 GHz whose Q comes from conductor loss (the metal shells
carry gsim's ``Conductivity``, a `Conductor`, whose surface impedance makes
the eigenproblem nonlinear) and radiation through the absorbing walls. The
model is built from each case's Palace ``config.json`` as gsim would build
it: metal volumes excluded, conductors on their shells, absorbing walls
with the case's order.

- Parity (``eigenmode/ihp_hw_r8_abc1``): the driven ``r8`` air box (walls
  50 um beyond the line ends, on the ground edges laterally) with
  first-order absorbing walls, where gfem and Palace discretize the same
  mesh: ``Re f`` and Q agree to 1e-5 and 1e-3 relative.
- Physics (``eigenmode/ihp_hw_box150``): the driven ``r8_box150`` air box
  (150 um margins, 300 um above and below) with second-order absorbing
  walls. gfem's textbook second-order term agrees with Palace's second
  order (the opposite sign) to 1e-4 in ``Re f`` and 1 % in Q, and with
  gfem's own first order to 1 % in Q.
- Wall time within 1.5x Palace's.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from isolated import peak_rss_gib

import gfem

THREADS = 4
#: Same-mesh parity (relative, on Re f and Q).
PARITY_F, PARITY_Q = 1e-5, 1e-3
#: The physics tolerances of the large box (relative).
PHYSICS_F, PHYSICS_Q = 1e-4, 1e-2


@pytest.fixture(scope="module")
def reference(reference_dir):
    return reference_dir / "eigenmode"


def _model(case: Path, absorbing_order: int | None = None) -> tuple[gfem.Model, dict]:
    config = json.loads((case / "config.json").read_text())
    unit = config["Model"]["L0"]
    assert unit == 1e-6
    mesh = gfem.Mesh.from_gmsh(case / "palace.msh", length_unit="um")
    name = {(r.dim, r.tag): r.name for r in mesh.regions}
    materials = {
        name[3, tag]: gfem.Material(
            eps_r=m["Permittivity"], tan_delta=m.get("LossTan", 0.0)
        )
        for m in config["Domains"]["Materials"]
        for tag in m["Attributes"]
    }
    bnd = config["Boundaries"]
    assert not bnd["LumpedPort"]
    assert not bnd["WavePort"]
    conductors = [
        gfem.Conductor(name[2, tag], c["Conductivity"], c["Thickness"] * unit)
        for c in bnd["Conductivity"]
        for tag in c["Attributes"]
    ]
    order = absorbing_order or bnd["Absorbing"]["Order"]
    absorbing = [
        gfem.Absorbing(name[2, tag], order=order)
        for tag in bnd["Absorbing"]["Attributes"]
    ]
    excluded = tuple(
        gfem.enclosed_by(c.surface, c.surface.replace("_xy", "_z"))
        for c in conductors
        if c.surface.endswith("_xy")
    )
    model = gfem.Model(
        mesh,
        materials=materials,
        boundaries=[*conductors, *absorbing],
        excluded=excluded,
    )
    return model, config


def _palace(case: Path) -> np.ndarray:
    """Palace's eig.csv rows: m, Re f (GHz), Im f (GHz), Q, errors."""
    return np.loadtxt(
        case / "output" / "palace" / "eig.csv", delimiter=",", skiprows=1, ndmin=2
    )


def _solve(case: Path, absorbing_order: int | None = None):
    model, config = _model(case, absorbing_order)
    eigen = config["Solver"]["Eigenmode"]
    problem = gfem.Eigenmode(target=eigen["Target"] * 1e9, n_modes=eigen["N"])
    options = gfem.SolverOptions(order=config["Solver"]["Order"], threads=THREADS)
    load = os.getloadavg()[0]
    start = time.perf_counter()
    result = gfem.solve(model, problem, options=options)
    wall = time.perf_counter() - start
    rss = peak_rss_gib()
    palace_run = json.loads((case / "run.json").read_text())
    print(  # noqa: T201
        f"{case.name} (absorbing order {absorbing_order or 'as config'}): "
        f"{wall:.1f} s (Palace {palace_run['wall_s']:.0f} s), peak RSS "
        f"{rss:.2f} GiB (Palace {palace_run['peak_rss_gib']:.2f}), load "
        f"{load:.1f} -> {os.getloadavg()[0]:.1f}\n  f "
        f"{np.asarray(result.frequency) / 1e9} GHz, Q {np.asarray(result.q)}, "
        f"error {np.asarray(result.error)}"
    )
    return result, wall, palace_run


def _gap(result, palace_row) -> tuple[float, float]:
    f = float(result.frequency[0]) / 1e9
    q = float(result.q[0])
    return abs(f / palace_row[1] - 1), abs(q / palace_row[3] - 1)


def test_nonlinear_parity_with_palace_on_the_same_mesh(reference):
    case = reference / "ihp_hw_r8_abc1"
    result, wall, palace_run = _solve(case)
    gap_f, gap_q = _gap(result, _palace(case)[0])
    print(f"ihp_hw_r8_abc1 vs Palace: Re f {gap_f:.2e}, Q {gap_q:.2e}")  # noqa: T201
    assert gap_f <= PARITY_F
    assert gap_q <= PARITY_Q
    assert wall <= 1.5 * palace_run["wall_s"]
    assert math.isfinite(wall)


def test_second_order_absorbing_physics_on_the_large_box(reference):
    case = reference / "ihp_hw_box150"
    second, wall, palace_run = _solve(case)
    gap_f, gap_q = _gap(second, _palace(case)[0])
    first, _, _ = _solve(case, absorbing_order=1)
    q2, q1 = float(second.q[0]), float(first.q[0])
    print(  # noqa: T201
        f"ihp_hw_box150 order 2 vs Palace order 2: Re f {gap_f:.2e}, Q "
        f"{gap_q:.2e}; gfem order 2 vs order 1: Q {abs(q2 / q1 - 1):.2e}"
    )
    assert gap_f <= PHYSICS_F
    assert gap_q <= PHYSICS_Q
    assert abs(q2 / q1 - 1) <= PHYSICS_Q
    assert wall <= 1.5 * palace_run["wall_s"]
