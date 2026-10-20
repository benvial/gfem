"""Eigenmode problems against Palace on the QPDK coupled CPW resonator.

A 4 mm CPW resonator (10 um / 6 um) as a PEC sheet on 500 um anisotropic
sapphire under 500 um of vacuum, its feedline ends loaded by two 50 ohm
two-element lumped ports: a quadratic eigenproblem (port resistance). The
model is built from each case's Palace ``config.json`` as gsim would build
it. ``eigenmode/r20`` is order 2 (one mode near 7.78 GHz), ``r5_p1`` order 1
(three modes).

- The quarter-wave mode near the 7.78 GHz target (7.683 GHz on r20,
  7.254 GHz on r5_p1) is missing from Palace's gsim runs: Palace's SLEPc
  search returns only modes with Re f >= Target. The parity runs set the
  target to 7 GHz and ask for one more mode; gfem solves for one more mode
  than ``config.json``'s ``N`` at its target, and Palace's modes are
  matched by frequency.
- Parity against Palace on the mesh gfem reads, both codes at eigen
  tolerance 1e-10 (``output/palace_nobisect_tol1e-10``; gfem's default).
  Palace's default ``Model.RefineCrackElements`` bisects the elements of
  "under-resolved interior boundaries" next to the PEC sheet before
  solving (1 385 more elements on r20, 562 on r5_p1; ``palace.stdout``):
  those runs, ``output/palace``, differ from gfem by 2e-5..8e-5 in Re f.
  The parity runs turn that refinement off and tighten Palace's eigen and
  linear tolerances from gsim's 1e-6 to 1e-10: at 1e-6 these port-loaded
  modes are not resolved to the parity gates in either code (Palace's Q
  moves by up to 12 % and Re f by up to 1.5e-3, ``output/palace_nobisect``;
  gfem's Re f by 1e-5 and Q by 0.5 %).
- Golden baselines: gfem's own values on these meshes, for regressions.
- Sapphire's loss tangent is per axis (3e-5, 3e-5, 8.6e-5 along its
  axes), as in Palace's ``LossTan``.
- Wall time within 1.5x Palace's gsim-default run (``run.json``); peak RSS
  is printed.
"""

import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import pytest

import gfem

THREADS = 4
#: The ticket's parity tolerances (relative), on Re f and Q.
PARITY_F, PARITY_Q = 1e-5, 1e-3
#: Palace's run on the imported mesh (``Model.RefineCrackElements`` false:
#: no local bisection next to the PEC sheet; eigen and linear tolerances
#: 1e-10), beside the original run.
SAME_MESH = "palace_nobisect_tol1e-10"
#: Relative tolerances (Re f, Q) of the golden check: repeated solves
#: spread by 1e-9 in Re f and 1e-5 in Q (threaded assembly round-off; the Q
#: 2400 mode of r5_p1).
GOLDEN_RTOL = 1e-7, 5e-5
#: gfem on the reference meshes: (Re f in Hz, Q) per mode, nearest the target
#: first.
GOLDEN = {
    "r20": [(7.682751588e9, 1472.297987), (16.671307772e9, 76.770489)],
    "r5_p1": [
        (7.253515900e9, 1803.404767),
        (16.113966300e9, 85.511569),
        (21.795791020e9, 2403.167),
        (31.793035140e9, 47.892295),
    ],
}


@pytest.fixture(scope="module")
def reference():
    ref = os.environ.get("GFEM_REFERENCE_DIR")
    if not ref:
        pytest.skip("GFEM_REFERENCE_DIR not set (needs the eigenmode Palace runs)")
    return Path(ref) / "eigenmode"


def _model(case: Path) -> tuple[gfem.Model, dict]:
    config = json.loads((case / "config.json").read_text())
    assert config["Model"]["L0"] == 1e-6
    mesh = gfem.Mesh.from_gmsh(case / "palace.msh", length_unit="um")
    name = {(r.dim, r.tag): r.name for r in mesh.regions}
    materials = {}
    for m in config["Domains"]["Materials"]:
        material = gfem.Material(
            eps_r=m["Permittivity"],
            mu_r=m.get("Permeability", 1.0),
            tan_delta=_per_axis(m.get("LossTan", 0.0)),
            axes=m.get("MaterialAxes"),
        )
        for tag in m["Attributes"]:
            materials[name[3, tag]] = material
    bnd = config["Boundaries"]
    assert not bnd["Conductivity"]
    assert not bnd["WavePort"]
    ports = [
        gfem.LumpedPort(
            f"p{p['Index']}",
            tuple(name[2, e["Attributes"][0]] for e in p["Elements"]),
            tuple(e["Direction"] for e in p["Elements"]),
            impedance=p["R"],
        )
        for p in sorted(bnd["LumpedPort"], key=lambda p: p["Index"])
    ]
    model = gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC(name[2, t]) for t in bnd["PEC"]["Attributes"]],
        ports=ports,
    )
    return model, config


def _per_axis(value: float | list[float]) -> float | tuple[float, ...]:
    return tuple(value) if isinstance(value, list) else float(value)


def _palace(case: Path) -> tuple[np.ndarray, np.ndarray]:
    """eig.csv and port-Q.csv rows of Palace's run on the imported mesh."""
    out = case / "output" / SAME_MESH
    if not (out / "eig.csv").exists():
        pytest.skip(f"no Palace run without bisection in {out}")
    stdout = (out / "palace.stdout").read_text()
    assert "local bisection" not in stdout
    eig = np.loadtxt(out / "eig.csv", delimiter=",", skiprows=1, ndmin=2)
    q_port = np.loadtxt(out / "port-Q.csv", delimiter=",", skiprows=1, ndmin=2)
    return eig, q_port


@pytest.fixture(scope="module", params=["r20", "r5_p1"])
def solved(request, reference):
    case = reference / request.param
    model, config = _model(case)
    eigen = config["Solver"]["Eigenmode"]
    problem = gfem.Eigenmode(target=eigen["Target"] * 1e9, n_modes=eigen["N"] + 1)
    options = gfem.SolverOptions(
        order=config["Solver"]["Order"],
        threads=THREADS,
    )
    load = os.getloadavg()[0]
    start = time.perf_counter()
    result = gfem.solve(model, problem, options=options)
    wall = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    palace_run = json.loads((case / "run.json").read_text())
    print(  # noqa: T201
        f"{request.param}: {problem.n_modes} modes in {wall:.1f} s (Palace "
        f"{palace_run['wall_s']:.0f} s), peak RSS {rss:.2f} GiB (Palace "
        f"{palace_run['peak_rss_gib']:.2f}), load {load:.1f} -> "
        f"{os.getloadavg()[0]:.1f}\n  {result}"
    )
    return request.param, case, result, wall, palace_run


def test_eigenmode_golden_baseline(solved):
    name, _, result, _, _ = solved
    golden = np.array(GOLDEN[name])
    rtol_f, rtol_q = GOLDEN_RTOL
    np.testing.assert_allclose(result.frequency, golden[:, 0], rtol=rtol_f)
    np.testing.assert_allclose(result.q, golden[:, 1], rtol=rtol_q)
    assert (np.asarray(result.error) <= gfem.SolverOptions().eigen_tolerance).all()
    # the quarter-wave mode Palace missed, nearest the 7.78 GHz target
    assert abs(float(result.frequency[0]) - 7.78e9) < 0.6e9


def test_eigenmode_parity_with_palace(solved):
    name, case, result, _, _ = solved
    eig, q_port = _palace(case)
    f = np.asarray(result.frequency) / 1e9
    q = np.asarray(result.q)
    worst_f = worst_q = worst_port = 0.0
    for row, ports in zip(eig, q_port, strict=True):
        j = int(np.argmin(np.abs(f - row[1])))
        worst_f = max(worst_f, abs(f[j] / row[1] - 1))
        worst_q = max(worst_q, abs(q[j] / row[3] - 1))
        for p, port in enumerate(("p1", "p2")):
            mine = float(result.q_port[port][j])
            worst_port = max(worst_port, abs(mine / ports[1 + 2 * p] - 1))
    print(  # noqa: T201
        f"{name}: Palace modes matched to Re f {worst_f:.2e}, Q {worst_q:.2e}, "
        f"port Q {worst_port:.2e} (relative)"
    )
    assert worst_f <= PARITY_F
    assert worst_q <= PARITY_Q


def test_eigenmode_wall_time_within_palace_budget(solved):
    _, _, _, wall, palace_run = solved
    assert wall <= 1.5 * palace_run["wall_s"]
    assert math.isfinite(wall)
