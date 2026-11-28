r"""The mode cascade of a width step against the full FEM solve.

A PEC-walled box ($y \in [0, 2]$ um, $\lambda = 1.55$ um, $\varepsilon = 4$
core in $n = 1.444$ cladding) holds a lead core $[0.75, 1.25]$ that widens to
$[0.75, 1.65]$ for a section of length $L$ and narrows again. The two steps
are two blocks, solved on their own with $N$ kept modes of the section at a
plane $d$ from their step, and joined by `gfem.cascade`; the full device is
one solve at the same mesh size $h = 0.05$ um and order 4. The error is
$\max|S_\mathrm{cascade} - S_\mathrm{full}|$ over the external S (the lead
modes `in` and `out`); the section's length is plane to plane, $\ell = L - 2d$.

The truncation error falls as $e^{-2\alpha_{N+1} d}$ with the first dropped
mode's decay rate $\alpha_{N+1}$, so each gate is the error measured on the
shipped code times 2 to 4: the measurement is in the comment above it. Every
number is the blocks re-solved at that $N$, never modes sub-selected from a
larger solve.
"""

from __future__ import annotations

import numpy as np
import pytest
from cascade_cases import (
    X_END,
    solve_step,
    step_full_mesh,
    step_full_model,
    step_meshes,
    step_models,
)

import gfem

H = 0.05  # um
OPTIONS = gfem.SolverOptions(order=4, linear_solver="direct", threads=1)
LEADS = ("in", "out")


def _s(result):
    """S of the lead modes ``in`` and ``out``, first frequency."""
    s = result.s
    index = [s.ports.index(n) for n in LEADS]
    return np.asarray(s.values)[0][np.ix_(index, index)]


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    """``full(L, **kw)``: S of the whole device, solved once per case."""
    cache: dict = {}
    directory = tmp_path_factory.mktemp("full")

    def solve(length, **kw):
        key = (length, tuple(sorted(kw.items())))
        if key not in cache:
            mesh = step_full_mesh(directory / f"full{len(cache)}.msh", length, h=H)
            model = step_full_model(mesh, length, **kw)
            (res,) = solve_step([model], options=OPTIONS)
            cache[key] = _s(res)
        return cache[key]

    return solve


@pytest.fixture(scope="module")
def blocks(tmp_path_factory):
    """``blocks(d, n, **kw)``: the two solved blocks, once per case."""
    cache: dict = {}
    directory = tmp_path_factory.mktemp("blocks")

    def solve(d, n, *, shift=0.0, **kw):
        key = (d, n, shift, tuple(sorted(kw.items())))
        if key not in cache:
            sub = directory / f"b{len(cache)}"
            sub.mkdir()
            meshes = step_meshes(sub, h=H, shift=shift, d=d)
            lead_x = (0.0, X_END)
            models = step_models(meshes, n=n, lead_x=lead_x, **kw)
            cache[key] = solve_step(models, options=OPTIONS)
        return cache[key]

    return solve


def _err(blocks_, full_s, length, d):
    """Max error of the cascade at section length ``length`` (um) against ``full_s``."""
    joined = gfem.cascade(list(blocks_), {"s": (length - 2 * d) * 1e-6})
    return float(np.abs(_s(joined) - full_s).max())


def test_the_lossless_step_cascade_matches_the_full_solve(blocks, full):
    # N = 12, d = 0.5 um, L = 1 (a section of length 0: the blocks joined
    # directly) and 2.5 um. Measured on MN5: 4.0e-13 and 1.41e-10, the error of
    # the first dropped mode, e^{-2 alpha_13 d}; the gate is 3 times the larger
    # (the discretization sets it, not round-off, which stays near 1e-13).
    errs = [_err(blocks(0.5, 12), full(L), L, 0.5) for L in (1.0, 2.5)]
    assert max(errs) <= 4e-10, errs


def test_the_error_falls_with_the_kept_modes(blocks, full):
    # N = 4 keeps only the four propagating modes and is refused, so the
    # coarsest cascade has N = 5. Measured on MN5 at d = 0.5 um, max over
    # L = 1 and 2.5 um: 5.26e-6 at N = 5 and 1.41e-10 at N = 12, a ratio of
    # 3.7e4; the gate is a quarter of it.
    errs = {
        n: max(_err(blocks(0.5, n), full(L), L, 0.5) for L in (1.0, 2.5))
        for n in (5, 12)
    }
    assert errs[5] / errs[12] >= 1e4, errs


def test_the_lossy_step_cascade_matches_the_full_solve(blocks, full):
    # N = 8, d = 0.25 um, L = 1 and 2.5 um, tan_delta = 0.05 in the core.
    # Measured on MN5: 1.86e-8 and 2.35e-8; the gate is 3 times the larger.
    lossy = {"tan_delta": 0.05}
    errs = [
        _err(blocks(0.25, 8, **lossy), full(L, **lossy), L, 0.25) for L in (1.0, 2.5)
    ]
    assert max(errs) <= 7e-8, errs


def test_blocks_on_non_conforming_planes_join(blocks, full):
    # B's cladding nodes are moved by 0.3 of the smallest cell, so the two
    # planes carry different nodes; N = 8, d = 0.25 um, L = 1 um. Measured on
    # MN5: error 7.19e-8 (a conforming pair: 2.35e-8 at L = 2.5, the same
    # order); relative difference of the two blocks' gamma 1.02e-10, the
    # discretization's, which `gamma_mismatch` reports in units of k0
    # (2.6e-10). The gates are 3 times these.
    shifted = blocks(0.25, 8, shift=0.3)
    err = _err(shifted, full(1.0), 1.0, 0.25)
    joined = gfem.cascade(list(shifted), {"s": 0.5e-6})
    mismatch = float(np.max(np.asarray(joined.gamma_mismatch["s"])))
    gam_a, gam_b = (
        np.array([np.asarray(b.port_modes[f"s@{m}"].gamma)[0] for m in range(1, 9)])
        for b in shifted
    )
    rel = float(np.max(np.abs(gam_a - gam_b) / np.abs(gam_a)))
    assert err <= 2e-7, err
    assert rel <= 3e-10, rel
    assert mismatch <= 8e-10, mismatch


@pytest.mark.parametrize("n", [5, 8, 16])
def test_the_step_with_absorbing_walls_converges_with_the_kept_modes(blocks, full, n):
    # Absorbing lines on both walls (an open guide, leaky modes), d = 0.25 um,
    # L = 1 um. N = 4 keeps the propagating modes only and is refused. Measured
    # on MN5: 8.43e-5 (N = 5), 2.72e-7 (N = 8), 7.98e-11 (N = 16); the gates are
    # 3 times those, rounded.
    gate = {5: 3e-4, 8: 8e-7, 16: 3e-10}[n]
    open_ = {"absorbing": True}
    err = _err(blocks(0.25, n, **open_), full(1.0, **open_), 1.0, 0.25)
    assert err <= gate, err
