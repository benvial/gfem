"""The ``driven/qpdk_r5_p1`` reference case: the QPDK resonator, driven.

The ``eigenmode/r5_p1`` mesh and materials (4 mm CPW resonator as a PEC
sheet on anisotropic sapphire under vacuum), order 1, with its two 50 ohm
two-element lumped ports as excitations, swept over 15-23 GHz (801 points,
10 MHz step). Two modes lie in the band: about 16.11 GHz (loaded Q about
85) and 21.80 GHz (Q about 2 400, a 9 MHz linewidth).

The model is built from the case's Palace-style ``config.json`` as the
eigenmode reference tests build theirs, so the driven poles are the
eigenmodes of the same discrete operator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import gfem

CASE = "driven/qpdk_r5_p1"
THREADS = 4
#: Order 1 Nédélec: every DOF is an edge DOF, so BDDC would be a direct solve
#: of the whole system; `direct` also takes the high-Q resonance exactly.
OPTIONS = {"order": 1, "threads": THREADS, "linear_solver": "direct"}


def case_dir() -> Path:
    """The case directory under ``GFEM_REFERENCE_DIR`` (skips without it)."""
    ref = os.environ.get("GFEM_REFERENCE_DIR")
    if not ref:
        pytest.skip(f"GFEM_REFERENCE_DIR not set (needs the {CASE} reference case)")
    case = Path(ref) / CASE
    if not (case / "config.json").exists():
        pytest.skip(f"{case} has no config.json (build the {CASE} reference case)")
    return case


def options(**overrides: object) -> gfem.SolverOptions:
    """The case's solver options."""
    return gfem.SolverOptions(**{**OPTIONS, **overrides})


def model(case: Path) -> tuple[gfem.Model, dict]:
    """The case's model and its config."""
    config = json.loads((case / "config.json").read_text())
    assert config["Model"]["L0"] == 1e-6
    assert config["Solver"]["Order"] == 1
    mesh = gfem.Mesh.from_gmsh(case / "palace.msh", length_unit="um")
    name = {(r.dim, r.tag): r.name for r in mesh.regions}
    materials = {}
    for m in config["Domains"]["Materials"]:
        material = gfem.Material(
            eps_r=m["Permittivity"],
            mu_r=m.get("Permeability", 1.0),
            # gfem.Material takes one loss tangent: the mean of Palace's three
            tan_delta=float(np.mean(m.get("LossTan", 0.0))),
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
    built = gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC(name[2, t]) for t in bnd["PEC"]["Attributes"]],
        ports=ports,
    )
    return built, config


def grid(config: dict) -> np.ndarray:
    """The case's output frequencies (Hz): Palace's linear sample block."""
    (block,) = config["Solver"]["Driven"]["Samples"]
    assert block["Type"] == "Linear"
    lo, hi, step = block["MinFreq"], block["MaxFreq"], block["FreqStep"]
    n = round((hi - lo) / step) + 1
    return np.linspace(lo, hi, n) * 1e9


def confirmation_share(sweep: gfem.SweepInfo) -> float:
    """Share of the samples that only confirmed convergence (trailing errors
    under the tolerance)."""
    errors = sweep.sample_errors
    confirming = next(
        (k for k, e in enumerate(reversed(errors)) if not e < sweep.tolerance),
        len(errors),
    )
    return confirming / len(errors)
