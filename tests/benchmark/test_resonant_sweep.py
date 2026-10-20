"""Adaptive against an 801-point uniform sweep on a high-Q resonator.

``driven/qpdk_r5_p1`` (see `qpdk_resonator`): 801 full solves of the
order-1 resonator (127k DOF, one factorization per frequency), against the
adaptive sweep over the same grid at the default ``adaptive_tolerance=1e-3``:
|dS| <= 1e-5 everywhere, inside the default cap of 20 samples. The uniform
S is written to the test's temporary directory as ``uniform-801.npz``;
copied to the case's ``output/gfem/``, it is what the nightly reference
tier checks the adaptive sweep against.
"""

import os
import resource
import time

import numpy as np
import qpdk_resonator as qr

import gfem

S_TOLERANCE = 1e-5


def test_adaptive_matches_801_point_uniform_sweep(tmp_path):
    case = qr.case_dir()
    model, config = qr.model(case)
    frequency = qr.grid(config)
    assert len(frequency) == 801
    options = qr.options()

    load = os.getloadavg()[0]
    start = time.perf_counter()
    adaptive = gfem.solve(
        model, gfem.Driven(frequency=frequency, sweep="adaptive"), options=options
    )
    adaptive_wall = time.perf_counter() - start
    start = time.perf_counter()
    uniform = gfem.solve(model, gfem.Driven(frequency=frequency), options=options)
    uniform_wall = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    s_uniform = np.asarray(uniform.s.values)
    np.savez(
        tmp_path / "uniform-801.npz",
        frequency=frequency,
        s=s_uniform,
        wall_s=uniform_wall,
        threads=qr.THREADS,
        load=load,
    )
    info = adaptive.sweep
    diff = np.abs(np.asarray(adaptive.s.values) - s_uniform).max(axis=(1, 2))
    print(  # noqa: T201
        f"uniform 801 points {uniform_wall:.0f} s; adaptive {adaptive_wall:.1f} s, "
        f"{len(info.sample_errors)} samples at "
        f"{[round(f / 1e9, 3) for f in info.sampled_frequencies]} GHz, errors "
        f"{[f'{e:.1e}' for e in info.sample_errors]}, basis {info.basis_size}, "
        f"confirmation share {qr.confirmation_share(info):.2f}; max |dS| "
        f"{diff.max():.2e} at {frequency[diff.argmax()] / 1e9:.3f} GHz; peak RSS "
        f"{rss:.2f} GiB, load {load:.1f} -> {os.getloadavg()[0]:.1f}; uniform S in "
        f"{tmp_path / 'uniform-801.npz'}"
    )
    assert info.converged
    assert len(info.sampled_frequencies) <= options.adaptive_max_samples
    assert diff.max() <= S_TOLERANCE
    # reciprocal to the direct solver's accuracy
    np.testing.assert_allclose(s_uniform, np.swapaxes(s_uniform, 1, 2), atol=1e-9)
