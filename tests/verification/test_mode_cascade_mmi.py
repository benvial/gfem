r"""The mode cascade of a 1x2 multimode interference splitter against the full FEM.

A PEC-walled box ($y \in [0, 3]$ um, $\lambda = 1.55$ um) holds a 2 um wide
core ($\varepsilon = 4$ in $n = 1.444$ cladding) of length $L$, fed by a
0.5 um guide centred in the box, and two of them centred at $\pm 0.5$ um leave
it, each lead 1 um long. The ports are `in1` and the even and odd modes `out1`
and `out2`. The two ends of the MMI are two blocks, solved on their own with
$N$ kept modes of the MMI section at a plane $d$ inside it, and joined by
`gfem.cascade`; the full device is one solve at the same mesh size
$h = 0.05$ um and order 4. The section has 6 propagating modes and then
$\alpha = 0.58, 3.85, 5.76, 7.41, 8.87, 10.2$ /um, so $N \ge 7$.

The error is $\max|S_\mathrm{cascade} - S_\mathrm{full}|$ over the external S;
the section's length is plane to plane, $\ell = L - 2d$. The truncation error
falls as $e^{-2\alpha_{N+1} d}$, and every number here is the blocks re-solved
at that $N$, never modes sub-selected from a larger solve. Each gate is the
value measured on the shipped code times 2 to 4, written beside it.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cascade_cases import (
    EPS_CORE,
    MMI_OPTIONS,
    mmi_block_meshes,
    mmi_block_models,
    mmi_full,
    solve_step,
)

import gfem

LENGTHS = (2.0, 3.5, 5.0)  # um
PORTS = ("in1", "out1", "out2")


def _s(result):
    """S over the external ports ``in1``, ``out1``, ``out2``, first frequency."""
    s = result.s
    index = [s.ports.index(n) for n in PORTS]
    return np.asarray(s.values)[0][np.ix_(index, index)]


def _t(result):
    """The transmission :math:`|S_{out1,in1}|^2` of a result, traceable."""
    s = result.s
    return jnp.abs(s.values[0, s.ports.index("out1"), s.ports.index("in1")]) ** 2


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    """``full(L, eps)``: S of the whole device, solved once per case."""
    cache: dict = {}
    directory = tmp_path_factory.mktemp("full")

    def solve(length, eps=EPS_CORE):
        if (length, eps) not in cache:
            model = mmi_full(directory, length, eps_core=eps)
            (res,) = solve_step([model], options=MMI_OPTIONS)
            cache[length, eps] = _s(res)
        return cache[length, eps]

    return solve


@pytest.fixture(scope="module")
def blocks(tmp_path_factory):
    """``blocks(d, n)``: the two solved blocks, once per case."""
    cache: dict = {}
    directory = tmp_path_factory.mktemp("blocks")

    def solve(d, n):
        if (d, n) not in cache:
            sub = directory / f"b{len(cache)}"
            sub.mkdir()
            models = mmi_block_models(mmi_block_meshes(sub, d), d, n)
            cache[d, n] = solve_step(models, options=MMI_OPTIONS)
        return cache[d, n]

    return solve


def _err(blocks_, full_s, length, d):
    """Max error of the cascade at MMI length ``length`` (um) against ``full_s``."""
    joined = gfem.cascade(list(blocks_), {"s": (length - 2 * d) * 1e-6})
    return float(np.abs(_s(joined) - full_s).max())


def _errs(blocks_, full, d, lengths=LENGTHS):
    return [_err(blocks_, full(L), L, d) for L in lengths]


def test_the_mmi_cascade_matches_the_full_solve(blocks, full):
    # N = 12, d = 0.5 um. Measured on MN5: 7.5e-10, 1.8e-9, 4.2e-9 at L = 2,
    # 3.5, 5 um (the error of the first dropped mode, it grows with the
    # section's length as the propagating modes' phase errors do); the gate
    # is 2.4 times the largest.
    errs = _errs(blocks(0.5, 12), full, 0.5)
    assert max(errs) <= 1e-8, errs


def test_the_error_falls_monotonically_with_the_kept_modes(blocks, full):
    # N = 6 keeps only the propagating modes and is refused, so the ladder
    # starts at 7. Measured on MN5 (max over the three lengths): 5.2e-4,
    # 9.0e-6, 6.5e-8, 4.2e-9 at N = 7, 8, 10, 12, a ratio of 1.2e5 from the
    # first to the last; the gate is a quarter of it.
    ns = (7, 8, 10, 12)
    errs = [max(_errs(blocks(0.5, n), full, 0.5)) for n in ns]
    assert all(b <= a for a, b in itertools.pairwise(errs)), errs
    assert errs[0] / errs[-1] >= 3e4, errs


@pytest.mark.parametrize("n", [8, 10])
def test_the_error_rate_is_the_first_dropped_modes_decay(blocks, full, n):
    # err ~ exp(-2 alpha_{N+1} d): the ratio from d = 0.25 to 0.5 um is
    # exp(alpha / 2); alpha_{N+1} is the decay of the first dropped mode, read
    # from the N = 12 block. Measured on MN5 over L = 2, 3.5, 5 um: the worst
    # fit is 1.9 % off at N = 8 (5.79, 5.87, 5.81 against 5.76 /um) and
    # 3.4 % at N = 10 (8.82, 9.05, 9.18 against 8.87 /um); the gate is 10 %,
    # as the rate law holds to the prefactor's mild drift with d.
    fine = blocks(0.5, 12)[0].port_modes[f"s@{n + 1}"].gamma
    alpha = float(np.real(np.asarray(fine)[0])) * 1e-6
    wide = _errs(blocks(0.25, n), full, 0.25)
    close = _errs(blocks(0.5, n), full, 0.5)
    fits = [2 * np.log(w / c) for w, c in zip(wide, close, strict=True)]
    rel = max(abs(f - alpha) / alpha for f in fits)
    assert rel <= 0.10, (fits, alpha)


def _transmission(blocks_):
    """:math:`T(L)` of the joined blocks, ``L`` the MMI length in um (traceable)."""

    def t(length):
        return _t(gfem.cascade(list(blocks_), {"s": (length - 1.0) * 1e-6}))

    return t


def test_the_cascade_finds_the_best_length_the_full_solve_confirms(blocks, full):
    transmission = _transmission(blocks(0.5, 12))
    grid = np.arange(1.0, 12.0 + 1e-9, 0.005)
    values = np.asarray(jax.vmap(transmission)(jnp.asarray(grid)))
    best = float(grid[np.argmax(values)])
    slope = jax.grad(transmission)
    curvature = jax.grad(slope)
    for _ in range(8):  # Newton on dT/dL from the sweep's best length
        best -= float(slope(best)) / float(curvature(best))
    errs = [_err(blocks(0.5, 12), full(L), L, 0.5) for L in (best, 1.0, 12.0)]
    # Measured on MN5: L* = 3.1953 um, T = 0.95248; the full solve agrees to
    # 2.0e-9, 2.7e-9, 2.4e-9 at L*, 1 and 12 um. The gates: the optimum within
    # 1e-3 um of the 3.195 um the sweep found at the time (3 times the
    # 3e-4 um measured offset), the errors 3.7 times the largest.
    assert abs(best - 3.195) <= 1e-3, best
    assert max(errs) <= 1e-8, errs


def test_the_length_gradient_matches_differences_of_the_full_solve(blocks, full):
    # Measured on MN5: 0.12934066 /um from the cascade, relative difference
    # 2.5e-7 from the fourth-order differences. The step: the error of the
    # differences is (delta^4 / 30) T^(5), and T is a sum of terms in
    # cos((beta_i - beta_j) L) whose largest wave number is
    # beta_1 - beta_6 = 4.5 /um, so |T^(5)| <= 4.5^5 and the truncation
    # is (4.5 delta)^4 4.5 / 30 = 4e-8 at delta = 0.005 um, 3e-7 of the
    # derivative (an estimate of the order, as the measurement shows);
    # round-off of the solves, 1e-13 / delta, is below 1e-10.
    # The gate is 4 times the measurement.
    length, step = 3.0, 0.005
    ad = float(jax.grad(_transmission(blocks(0.5, 12)))(length))
    f = {j: abs(full(length + j * step)[1, 0]) ** 2 for j in (-2, -1, 1, 2)}
    fd = (f[-2] - 8 * f[-1] + 8 * f[1] - f[2]) / (12 * step)
    assert abs(ad - fd) / abs(fd) <= 1e-6, (ad, fd)


def test_the_gradient_through_the_blocks_matches_differences_of_the_full_solve(
    tmp_path,
):
    d, n, length = 0.5, 12, 3.0
    meshes = mmi_block_meshes(tmp_path, d)

    def transmission(eps):
        models = mmi_block_models(meshes, d, n, eps_core=eps, polarity=True)
        res = solve_step(models, options=MMI_OPTIONS)
        return _t(gfem.cascade(list(res), {"s": (length - 2 * d) * 1e-6}))

    ad = float(jax.grad(transmission)(jnp.asarray(EPS_CORE)))
    # The fourth-order differences of the full solve have the truncation
    # (delta^4 / 30) T^(5); T moves fast with the core permittivity (measured
    # against the gradient, the error is 7.2e-4, 2.9e-5, 1.6e-6 at
    # delta = 0.04, 0.02, 0.01, falling as delta^4 with 160 delta^4), so the
    # step is 0.0025: 6e-9, 1.6e-7 of the gradient; round-off of the solves,
    # 1e-13 / delta, is below 1e-10.
    step = 0.0025
    f = {}
    for j in (-2, -1, 1, 2):
        model = mmi_full(tmp_path, length, eps_core=EPS_CORE + j * step)
        (res,) = solve_step([model], options=MMI_OPTIONS)
        f[j] = float(_t(res))
    fd = (f[-2] - 8 * f[-1] + 8 * f[1] - f[2]) / (12 * step)
    # Measured on MN5: -0.039404457 against -0.039404453, relative 1.0e-7, the
    # truncation above plus the cascade's own error (6e-8 of the gradient,
    # the difference of the differences of the cascade and the full solve);
    # the gate is 3 times that.
    assert abs(ad - fd) / abs(fd) <= 3e-7, (ad, fd)
