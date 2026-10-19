"""The adaptive frequency sweep on a small dense affine system.

A(ω) = K - ω² M + iω C with real symmetric K, M, C (so A is complex
symmetric, like the driven operator), two right-hand sides i ω b_p and two
outputs l_p · x. K has four eigenvalues whose resonances fall in the band
and the rest far above it, so the response is a rational function with a
handful of poles in band plus a smooth background: the greedy needs a few
samples, never the whole grid.
"""

import math

import numpy as np
import pytest

import gfem
from gfem._sweep.adaptive import AffineSystem, AffineTerm, adaptive_sweep

N = 60
GRID = np.linspace(1.0, 3.0, 201)


def _system(seed=0, damping=0.02):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    eigenvalues = np.concatenate([[1.4, 2.9, 4.8, 7.3], rng.uniform(40, 90, N - 4)])
    k = q @ np.diag(eigenvalues) @ q.T
    m = np.eye(N)
    c = damping * np.diag(rng.uniform(0.5, 1.5, N))
    b = rng.standard_normal((N, 2))
    pieces = (k, m, c)
    thetas = (lambda _: 1.0, lambda w: -(w**2), lambda w: 1j * w)

    def matrix(w):
        return sum(t(w) * p for t, p in zip(thetas, pieces, strict=True))

    calls = []

    def solve(w, rhs, x0):
        calls.append((w, x0))
        return np.linalg.solve(matrix(w), rhs)

    system = AffineSystem(
        thetas=thetas,
        apply=lambda k_, x: pieces[k_] @ x,
        excitation=(AffineTerm(b, lambda w: 1j * w * np.eye(2)),),
        outputs=(AffineTerm(b, lambda _: np.eye(2)),),
        solve=solve,
    )

    def exact(w):
        x = np.linalg.solve(matrix(w), 1j * w * b)
        return x, b.T @ x

    return system, exact, calls


def _sweep(system, **kwargs):
    kwargs.setdefault("tolerance", 1e-8)
    kwargs.setdefault("max_samples", 40)
    kwargs.setdefault("memory", 2)
    return adaptive_sweep(system, GRID, **kwargs)


def test_greedy_samples_grid_points_from_the_endpoints_until_converged():
    system, _, _ = _system()
    sweep = _sweep(system)
    assert sweep.converged
    assert sweep.samples[:2] == (0, len(GRID) - 1)
    assert len(set(sweep.samples)) == len(sweep.samples)
    # a rational response with four poles in band: a handful of samples
    assert 4 <= len(sweep.samples) <= 16
    assert math.isinf(sweep.errors[0])
    assert math.isinf(sweep.errors[1])
    # memory: the last two samples confirm convergence
    assert all(e < 1e-8 for e in sweep.errors[-2:])
    assert sweep.errors[-3] >= 1e-8
    # two excitations, Re and Im: up to four basis columns per sample
    assert sweep.basis.shape == (N, sweep.model.size)
    assert sweep.model.size <= 4 * len(sweep.samples)
    np.testing.assert_allclose(
        sweep.basis.T @ sweep.basis, np.eye(sweep.model.size), atol=1e-12
    )


def test_reduced_outputs_match_full_solves_over_the_grid():
    system, exact, _ = _system()
    sweep = _sweep(system)
    worst = max(
        np.abs(sweep.outputs[i] - exact(w)[1]).max() / np.abs(exact(w)[1]).max()
        for i, w in enumerate(GRID)
    )
    assert worst < 1e-8
    # at a sample the reduced field is the full solution
    for i in sweep.samples:
        x, _ = exact(GRID[i])
        field = sweep.basis @ sweep.coordinates[i]
        np.testing.assert_allclose(field, x, atol=1e-11 * np.abs(x).max())


def test_each_new_sample_starts_from_the_reduced_prediction():
    system, exact, calls = _system()
    sweep = _sweep(system)
    assert [w for w, _ in calls] == [GRID[i] for i in sweep.samples]
    assert calls[0][1] is None
    x0 = calls[-1][1]
    x, _ = exact(calls[-1][0])
    # the last sample confirmed convergence: its guess is already accurate
    assert np.linalg.norm(x0 - x) <= 1e-8 * np.linalg.norm(x)


def test_frozen_samples_are_solved_as_given_without_greedy():
    system, exact, calls = _system()
    picks = (100, 0, 200, 37)
    sweep = _sweep(system, samples=picks)
    assert sweep.samples == picks
    assert [w for w, _ in calls] == [GRID[i] for i in picks]
    assert math.isinf(sweep.errors[0])
    assert all(np.isfinite(sweep.errors[1:]))
    for i in picks:
        np.testing.assert_allclose(
            sweep.outputs[i], exact(GRID[i])[1], atol=1e-10, rtol=0
        )


def test_the_sample_cap_stops_the_greedy_with_a_warning():
    system, _, _ = _system()
    with pytest.warns(gfem.SweepNotConvergedWarning, match="adaptive_max_samples"):
        sweep = _sweep(system, max_samples=3)
    assert not sweep.converged
    assert len(sweep.samples) == 3


def test_an_exhausted_grid_is_exact_and_converged():
    system, exact, _ = _system()
    grid = GRID[::50]  # five frequencies
    sweep = adaptive_sweep(system, grid, tolerance=1e-12, max_samples=20, memory=5)
    assert sweep.converged
    assert sorted(sweep.samples) == list(range(5))
    for i, w in enumerate(grid):
        np.testing.assert_allclose(sweep.outputs[i], exact(w)[1], rtol=1e-10)


def test_one_frequency_is_one_sample():
    system, exact, _ = _system()
    sweep = adaptive_sweep(system, GRID[:1], tolerance=1e-6, max_samples=20, memory=2)
    assert sweep.samples == (0,)
    assert sweep.converged
    np.testing.assert_allclose(sweep.outputs[0], exact(GRID[0])[1], rtol=1e-10)
