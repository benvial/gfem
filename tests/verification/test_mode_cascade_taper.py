r"""A linear taper as a staircase of cascaded width steps, against the full FEM.

The PEC box of the width-step test ($y \in [0, 2]$ um, $\lambda = 1.55$ um,
$\varepsilon = 4$ core in $n = 1.444$ cladding) holds a core that widens
linearly from 0.5 to 0.9 um over $L = 16$ um, between 1 um leads. A staircase of
$M$ segments gives each the smooth width at its midpoint, so it is $M + 1$
width steps: each solved on its own between two planes $d$ from it with $N$
kept modes, and joined by `gfem.cascade` through sections $L/M - 2d$ long. The
full device is one solve of the smooth taper at the same element order 4.
The error is on the transmission $S_{\mathrm{out,in}}$ between the leads' first
modes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from cascade_cases import (
    TAPER_L,
    solve_step,
    staircase_widths,
    taper_blocks,
    taper_full_mesh,
    taper_full_model,
    taper_sections,
)

import gfem

OPTIONS = gfem.SolverOptions(order=4, linear_solver="direct", threads=1)
N, D = 6, 0.2  # kept modes (four propagate), um from each step to its planes
STEPS = (4, 8, 16, 32)


def _transmission(result):
    s = result.s
    return np.asarray(s.values)[0][s.ports.index("out"), s.ports.index("in")]


@pytest.fixture(scope="module")
def errors(tmp_path_factory):
    """``{M: |T_staircase - T_full|}`` of the lead modes, each case solved once."""
    directory = tmp_path_factory.mktemp("taper")
    mesh = taper_full_mesh(directory / "full.msh")
    (full,) = solve_step([taper_full_model(mesh)], options=OPTIONS)
    out = {}
    for m in STEPS:
        sub = directory / f"m{m}"
        sub.mkdir()
        solved = solve_step(taper_blocks(sub, m, N, D), options=OPTIONS)
        joined = gfem.cascade(list(solved), taper_sections(m, D))
        out[m] = float(abs(_transmission(joined) - _transmission(full)))
    return out


@pytest.mark.parametrize("m", STEPS)
def test_a_staircase_has_a_block_per_width_step(tmp_path, m):
    # The m + 2 widths (the leads and the m midpoints) are all distinct, so no
    # block repeats: m + 1 blocks, and their planes are L / m - 2 D apart.
    widths = staircase_widths(m)
    assert len(set(np.round(widths, 12))) == m + 2
    assert len(taper_blocks(tmp_path, m, N, D)) == m + 1
    assert TAPER_L / m - 2 * D >= 0.0


def test_the_staircase_converges_in_steps(errors):
    # N = 6, d = 0.2 um, L = 16 um. Measured on MN5 (errors of the
    # transmission): 7.57e-2, 1.14e-2, 1.91e-3, 3.70e-4 at M = 4, 8, 16, 32,
    # the pairwise orders 2.73, 2.58, 2.37 and the order over the range
    # log2(e_4 / e_32) / 3 = 2.56 (midpoint widths: O(M^-2), the rest is the
    # taper's own higher terms). The error does not move with N (6 against 24
    # agree to 3 digits), d (0.1 to 0.25) nor the full mesh's h (0.1 to 0.05),
    # so it is the staircase's. The gates are the order within 0.3 of the
    # measurement and the M = 32 error at 3 times it.
    # The reflection S_in,in is not gated: at M = 16 and 32 the segments, 1 um
    # and 0.5 um long, are about half a guided wavelength (the staircase is a
    # grating near its Bragg condition), so its error stops falling
    # (1.5e-3, 6.4e-4, 1.2e-3 at M = 8, 16, 32): it is physics of the staircase.
    e = [errors[m] for m in STEPS]
    order = np.log2(e[0] / e[-1]) / (len(STEPS) - 1)
    assert abs(order - 2.56) <= 0.3, (order, errors)
    assert errors[32] <= 1.1e-3, errors
    assert all(a > b for a, b in itertools.pairwise(e)), errors
