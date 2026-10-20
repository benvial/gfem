"""Adaptive sweep on a high-Q resonator, against its eigenmodes (nightly).

``driven/qpdk_r5_p1`` (see `qpdk_resonator`) has two modes in its 15-23 GHz
band. The ports load the driven and the eigenmode problems identically, so
the poles of the driven S are the eigenvalues of the same discrete operator:

- Zoom grids: per mode, an adaptive sweep over 201 points within 5
  linewidths (f / Q, from the eigenmode) of the mode, fitted to a one-pole
  (complex Lorentzian) model shared by the four S entries. The fitted
  resonance frequency matches the eigenmode's Re f to 1e-5 (relative) and
  its loaded Q to 2 % (measured: 1e-11 and 1e-8 against eigenmodes
  converged to a backward error of 1e-10).
- Full band: the adaptive sweep over the 801-point grid against the
  uniform sweep the benchmark tier computes (stored with the case as
  ``output/gfem/uniform-801.npz``), |dS| <= 1e-5 at the default
  ``adaptive_tolerance=1e-3``, converged within the default cap.
"""

import os
import time

import numpy as np
import pytest
import qpdk_resonator as qr
from resonance import fit_resonance

import gfem

#: Band of the case and zoom half-width in linewidths.
BAND = (15e9, 23e9)
LINEWIDTHS = 5.0
ZOOM_POINTS = 201
#: Numerator degree of the one-pole fit: the residue plus a cubic background
#: (the 16 GHz mode's 10 linewidths span 2 GHz, where a linear background
#: biases its fitted Q by 1 %).
FIT_DEGREE = 4
#: Eigenmode backward error for the cross-check. The default (1e-6, Palace's)
#: leaves the eigenvalues off by up to 1e-5 in f and 0.5 % in Q here (the
#: backward error is relative to the norm of the whole curl-curl operator).
EIGEN_TOLERANCE = 1e-10
#: Acceptance: fitted f against the eigenmode (relative), loaded Q, and
#: adaptive against uniform S.
F_TOLERANCE, Q_TOLERANCE, S_TOLERANCE = 1e-5, 0.02, 1e-5


@pytest.fixture(scope="module")
def case(reference_dir):
    return qr.case_dir(reference_dir)


@pytest.fixture(scope="module")
def built(case):
    return qr.model(case)


@pytest.fixture(scope="module")
def modes(built):
    """The gfem eigenmodes in the band: [(Re f Hz, Q)], ascending."""
    model, _ = built
    # the two in-band modes are the nearest ones to the band centre
    problem = gfem.Eigenmode(target=0.5 * sum(BAND), n_modes=2)
    start = time.perf_counter()
    options = qr.options(eigen_tolerance=EIGEN_TOLERANCE)
    result = gfem.solve(model, problem, options=options)
    wall = time.perf_counter() - start
    found = sorted(zip(np.asarray(result.frequency), np.asarray(result.q), strict=True))
    print(f"eigenmodes {found} in {wall:.1f} s")  # noqa: T201
    in_band = [(float(f), float(q)) for f, q in found if BAND[0] < f < BAND[1]]
    assert len(in_band) == 2, found
    return in_band


def _sweep(model, frequency):
    start = time.perf_counter()
    load = os.getloadavg()[0]
    result = gfem.solve(
        model,
        gfem.Driven(frequency=frequency, sweep="adaptive"),
        options=qr.options(),
    )
    wall = time.perf_counter() - start
    info = result.sweep
    print(  # noqa: T201
        f"adaptive over {len(frequency)} points: {len(info.sample_errors)} samples at "
        f"{[round(f / 1e9, 4) for f in info.sampled_frequencies]} GHz, errors "
        f"{[f'{e:.1e}' for e in info.sample_errors]}, basis {info.basis_size}, "
        f"converged {info.converged}, confirmation share "
        f"{qr.confirmation_share(info):.2f}; {wall:.1f} s, load {load:.1f}"
    )
    return result


@pytest.mark.parametrize("mode", [0, 1], ids=["16GHz", "22GHz"])
def test_zoom_sweep_resonance_matches_eigenmode(built, modes, mode):
    model, _ = built
    f0, q0 = modes[mode]
    half = LINEWIDTHS * f0 / q0
    frequency = np.linspace(f0 - half, f0 + half, ZOOM_POINTS)
    result = _sweep(model, frequency)
    assert result.sweep.converged
    s = np.asarray(result.s.values)
    fit = fit_resonance(frequency, s, degree=FIT_DEGREE)
    print(  # noqa: T201
        f"mode {mode}: eigenmode f {f0 / 1e9:.7f} GHz Q {q0:.2f}; fit f "
        f"{fit.frequency / 1e9:.7f} GHz ({fit.frequency / f0 - 1:+.1e}) Q "
        f"{fit.q:.2f} ({fit.q / q0 - 1:+.1e}), misfit {fit.residual:.1e}"
    )
    assert fit.residual < 1e-5
    assert fit.frequency == pytest.approx(f0, rel=F_TOLERANCE)
    assert fit.q == pytest.approx(q0, rel=Q_TOLERANCE)


def test_adaptive_matches_stored_uniform_sweep(case, built):
    stored = case / "output" / "gfem" / "uniform-801.npz"
    if not stored.exists():
        pytest.skip(f"{stored} missing (written by the benchmark tier's run)")
    model, config = built
    frequency = qr.grid(config)
    uniform = np.load(stored)
    np.testing.assert_array_equal(uniform["frequency"], frequency)
    result = _sweep(model, frequency)
    options = qr.options()
    assert result.sweep.converged
    assert len(result.sweep.sampled_frequencies) <= options.adaptive_max_samples
    diff = np.abs(np.asarray(result.s.values) - uniform["s"]).max(axis=(1, 2))
    print(  # noqa: T201
        f"max |dS| adaptive vs uniform {diff.max():.2e} at "
        f"{frequency[diff.argmax()] / 1e9:.3f} GHz"
    )
    assert diff.max() <= S_TOLERANCE
