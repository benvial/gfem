"""A bent mode on a large layered section solves at about the straight guide's cost.

The 50k-DOF silicon-on-insulator-like strip of `pml_cases` (core 2.0, cladding
1.45, 1.0 x 0.4 um, 1.2 um of cladding and a 0.6 um layer all round) bent about
the vertical axis at $R$ = 5 um. The bent fundamental is the continuation of the
closed straight guide's fundamental, found by one targeted solve.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pml_cases
import pytest

import gfem

pytestmark = pytest.mark.slow

RADIUS = 5e-6
#: $n_\mathrm{eff}$ of the bent fundamental, 7 digits (the literal of the cost
#: study, written to 4-7 digits; its partner mode of the other polarization
#: and the layer modes sit more than 1e-2 away, so 1e-5 guards the pick).
N_EFF = 1.678258 - 9.901e-3j


def _solve(model, bend):
    options = gfem.SolverOptions(order=2, threads=1)
    start = time.perf_counter()
    result = gfem.solve(
        model, gfem.PortMode("p1", pml_cases.STRIP_F, bend=bend), options=options
    )
    return result, time.perf_counter() - start


def test_a_bent_strip_mode_is_the_continued_fundamental(tmp_path):
    mesh = pml_cases.strip_guide(
        tmp_path / "strip.msh", 1.2, quarter=False, length=0.1, core_h=0.05, clad_h=0.08
    )
    model = pml_cases.strip_model(mesh)
    bent, t_bent = _solve(model, gfem.Bend(RADIUS, outward="+x", at=(0.0, 0.0, 0.0)))
    _, t_straight = _solve(model, None)
    k0 = 2 * math.pi * pml_cases.STRIP_F / 299792458.0
    n_eff = -1j * complex(np.asarray(bent.gamma)[0]) / k0
    # the literal is rounded to 5e-7 in Re and 5e-7 in Im (|rounding| <= 7e-7,
    # 4.2e-7 relative), so the gate 1e-5 is 24x that; the nearest wrong pick
    # (the partner polarization, the layer modes) is more than 1e-2 away
    assert abs(n_eff - N_EFF) <= 1e-5 * abs(N_EFF)
    # an observation, not a gate
    print(  # noqa: T201
        f"bent/straight solve time {t_bent / t_straight:.2f} (bent {t_bent:.1f} s)"
    )
